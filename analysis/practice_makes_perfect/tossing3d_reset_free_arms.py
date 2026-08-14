"""What does a genuinely reset-free arm cost on Tossing3D?

Post-run analysis only; it reads `--results-root` back in and never drives a `Method`.

## Why this measurement did not exist before

#178 ran `--practice-reset-policy never` against `scheduled` at 100 cycles x 10 seeds and
found the two arms byte-identical in every `stats.json` field except
`num_practice_resets`. The flag was a no-op on this domain: `PracticeLoop` sampled the
train task before the reset-policy branch, and `Tossing3DTasks.build_task` could only
build an initial `State` by rebuilding the MuJoCo scene, so the scene was rebuilt every
cycle whatever the flag said. The reset-free condition was never realised, so there is no
earlier number here to compare against -- this is the first measurement of the arm, not a
re-analysis of one.

## What "reset-free" now means on this domain, which is not what it means elsewhere

After the fix, a reset-free run practises in **one scene for its whole length** --
whatever `hard_reset` left behind. That is not an implementation shortcut. On Tossing3D,
handing the robot a new scene and resetting it are the *same physical act*: the only way
to obtain a new initial state is `env.reset(seed=...)`. So the arm is "no reset" and "no
scene variety" inseparably, and a result about it is a result about both together. Any
reading that attributes the whole difference to the missing rescue is over-claiming.

## The rule, taken from #178 rather than invented here

Tossing3D's per-seed score is volatile between adjacent sweeps -- several tasks, with no
learning event in between -- so a single final sweep is one draw of a noisy variable.
#178 fixed a windowed rule before reading its numbers, and it is reused unchanged
because the volatility is a property of the domain, not of that experiment:

- A seed's score is its **mean solved count over the last `WINDOW` sweeps** (`LATE`).
- **Paired across seeds**, because both arms ran the same fixed seed set. An unpaired
  test would discard exactly the structure the design bought.
- Exact paired sign-flip on per-seed `never - scheduled`, and **the MDE beside it,
  always**: a null result without one cannot be told apart from an underpowered test.

Counts are reported `x/y`, never as a bare percentage.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402
from hitl_pmp.core.problem.environment.types import Object, State  # noqa: E402
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment  # noqa: E402
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills  # noqa: E402
from hitl_pmp.sampler_draws import SAMPLER_DRAWS_FILENAME  # noqa: E402

# Same window as #178, for the same reason: wide enough to damp the measured
# sweep-to-sweep volatility, and the two experiments stay comparable.
WINDOW = 10

# The arm directory names under `--results-root`, in the order they are reported and
# plotted. `scheduled` first because it is the incumbent every committed number sits on.
SCHEDULED = "scheduled"
NEVER = "never"

# #133's two non-learning arms on this domain, per seed out of 10 tasks. Horizontal
# REFERENCE LINES, never curves: neither learns, so a curve invites a reader to look for
# a trend in a constant.
ORACLE_PER_SEED = 10.0
RANDOM_SKILLS_PER_SEED = 2.4
ORACLE_LABEL = "skill-oracle ceiling — 100/100 (#133)"
RANDOM_SKILLS_LABEL = "random-skills — 24/100 (#133)"

# CLAUDE.md's training-curve-style section (#188): this figure has the genuine
# reset/scheduled-vs-reset-free/never axis the convention is written for, so `SCHEDULED`
# takes the exact spec blue and `NEVER` the exact spec orange (previously `#2166ac` /
# `#d6604d`, close but not the literal spec hex).
_ARM_COLOURS = {SCHEDULED: "#0072B2", NEVER: "#D55E00"}
# The one neutral reserved for reference/ceiling arms (`skill-oracle`, `random-skills`),
# distinguished from each other only by legend label/y-level, never by hue -- previously
# green `#1a9850` dashed and purple `#762a83` dotted.
_REFERENCE_COLOUR = "#666666"

# The four objects a `Tossing3D-o1` episode binds, named exactly as `KinderBackend` names
# them -- which is how a `sampler_draws.jsonl` `achieved` key (`"<object>.<feature>"`)
# parses back into a `State`. Constructing these imports no simulator: `Tossing3DEnvironment`
# is pure pydantic until its first `reset()`, so this module (and its tests) run on CI,
# which never installs the `tossing3d` extra.
_OBJECTS: tuple[Object, ...] = (
    Object(name="robot", type=Tossing3DEnvironment.robot_type),
    Object(name="cube_0", type=Tossing3DEnvironment.cube_type),
    Object(name="cuboid_barrier", type=Tossing3DEnvironment.barrier_type),
    Object(name="bin_0", type=Tossing3DEnvironment.bin_type),
)
# One object per type in this domain, so a skill's variables bind unambiguously by type.
# Derived rather than written out per skill: the preconditions themselves are then the
# only statement of what each skill needs, and this cannot drift from `skills.py`.
_OBJECT_BY_TYPE = {obj.type: obj for obj in _OBJECTS}
_SKILLS = (Tossing3DSkills.PICK, Tossing3DSkills.MOVE_TO_THROW_POSE, Tossing3DSkills.TOSS)


class Tossing3DResetFree:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load_arms(*, results_root: Path) -> dict[str, dict[int, list[tuple[int, int, int]]]]:
        """`{arm: {seed: evaluations}}`, each entry
        `(num_online_transitions, num_solved, num_total)` for one sweep.

        Read from `stats.json`, so only completed runs appear. A partially-finished run
        has no `stats.json` and is silently absent rather than contributing a truncated
        curve that would shorten every window computed from it.

        **Two layouts, both accepted, because the tree moves between them.**
        `scripts/run_sweep.py` writes `<arm>/<method>/<seed>/`, while a committed
        `docs/experiment-logs/` tree drops the method level and is `<arm>/<seed>/`. A
        glob fixed to one depth finds nothing under the other and `report` then prints
        "No completed runs" and exits 0 -- a silent wrong answer, not a failure. So the
        search is recursive and the **seed is the containing directory's name**, which is
        true of both.

        Deliberately rooted at `results_root / arm` rather than at `results_root`: keying
        on the containing directory alone would collide `scheduled/0` with `never/0` into
        one entry. That collision is invisible for as long as two arms agree, which is
        exactly the condition this whole experiment exists because of.
        """
        arms: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
        for arm in (SCHEDULED, NEVER):
            curves: dict[int, list[tuple[int, int, int]]] = {}
            for path in sorted((results_root / arm).rglob("stats.json")):
                if not path.parent.name.isdigit():
                    continue
                stats = json.loads(path.read_text())
                curves[int(path.parent.name)] = [
                    (int(t), int(s), int(n)) for t, s, n in stats["evaluations"]
                ]
            if curves:
                arms[arm] = curves
        return arms

    @staticmethod
    def transitions_per_cycle(*, evaluations: list[tuple[int, int, int]]) -> list[int]:
        """How many environment steps each practice period actually took.

        `evaluations[i][0]` is `num_online_transitions` as of sweep `i`, and sweep 0
        happens before any practice, so cycle `i`'s cost is the difference between
        consecutive sweeps. A **zero** means the period took no step at all.

        That is the quantity a reset-free arm on this domain lives or dies by, and it is
        not the same question as task success. A tossed cube ends up past an immovable
        barrier and no skill brings it back, while `Toss` deletes `Reachable` -- so after
        a throw nothing is applicable. With a per-period reset the next period starts
        fresh; with none, the robot can simply stop accumulating experience. An arm that
        scores lower because it practised less is a different finding from one that
        scores lower because it learned less, and only this separates them.
        """
        return [evaluations[i + 1][0] - evaluations[i][0] for i in range(len(evaluations) - 1)]

    @staticmethod
    def load_practice_outcomes(
        *, results_root: Path
    ) -> dict[str, dict[int, list[dict[str, dict[str, int]]]]]:
        """`{arm: {seed: practice_outcomes_per_cycle}}` -- per cycle, per skill, the exact
        attempt/success counts `stats.json` already records. This, together with
        `evaluations` (via `transitions_per_cycle`), is the transition-level ground truth
        `toss_transition_index` is derived from -- never the score curve, which can
        plateau or stay flat for reasons unrelated to stranding. Same two-layout handling
        as `load_arms`, for the same reason."""
        outcomes: dict[str, dict[int, list[dict[str, dict[str, int]]]]] = {}
        for arm in (SCHEDULED, NEVER):
            per_seed: dict[int, list[dict[str, dict[str, int]]]] = {}
            for path in sorted((results_root / arm).rglob("stats.json")):
                if not path.parent.name.isdigit():
                    continue
                stats = json.loads(path.read_text())
                per_seed[int(path.parent.name)] = stats["practice_outcomes_per_cycle"]
            if per_seed:
                outcomes[arm] = per_seed
        return outcomes

    @staticmethod
    def toss_transition_index(
        *,
        evaluations: list[tuple[int, int, int]],
        outcomes: list[dict[str, dict[str, int]]],
    ) -> int:
        """The transition index of the run's last real action -- a `Toss` attempt --
        read from the per-cycle transition and skill-attempt record, never from where the
        score curve visually goes flat.

        `Toss` unconditionally deletes both `Holding` and `Reachable`
        (`Tossing3DSkills.TOSS`), so nothing in this domain is applicable in the state it
        leaves behind -- see `stranding_onset`, which this reuses to find the last cycle
        with any activity. That cycle's own `Toss` attempt count is checked, not assumed,
        so a stall with a different cause (e.g. `MoveToThrowPose` never succeeding) raises
        instead of being mislabelled as a toss. The returned index is the cumulative
        transition count as of the end of that cycle -- exactly where a `render_practice`
        curve elbows from rising to flat.
        """
        skills, index = Tossing3DResetFree.last_practice_action(
            evaluations=evaluations, outcomes=outcomes
        )
        if "Toss" not in skills:
            onset = Tossing3DResetFree.stranding_onset(
                transitions=Tossing3DResetFree.transitions_per_cycle(evaluations=evaluations)
            )
            raise ValueError(
                f"cycle {onset if onset is None else onset - 1} (the last active cycle "
                "before stranding) recorded no Toss attempt -- the stranding is not "
                "attributable to a toss on this run, so annotating it as one would "
                "misdescribe the mechanism"
            )
        return index

    @staticmethod
    def last_practice_action(
        *,
        evaluations: list[tuple[int, int, int]],
        outcomes: list[dict[str, dict[str, int]]],
    ) -> tuple[tuple[str, ...], int]:
        """`(skills attempted in the last active cycle, cumulative transition index at its
        end)` -- `toss_transition_index` with the `Toss` assumption removed.

        The assumption was invisible while it held. At the 2026-08-08 pins every one of
        the ten never-arm seeds ended its single practice period on a `Toss`, so a
        derivation that required one described the data exactly. At the 2026-08-13 pins it
        does not: some seeds strand after a failed `Pick` or a dropped cube, having never
        thrown. `toss_transition_index` still raises on those, deliberately -- naming an
        event that did not happen is worse than refusing -- and this is what the figure
        annotates with instead, so a seed is labelled for what it actually did.

        Skill names come back sorted, so the label is stable across runs rather than
        inheriting `stats.json`'s dict order.
        """
        transitions = Tossing3DResetFree.transitions_per_cycle(evaluations=evaluations)
        onset = Tossing3DResetFree.stranding_onset(transitions=transitions)
        if onset is None:
            raise ValueError("run never stranded -- no terminal action to locate")
        last_active_cycle = onset - 1
        attempted = tuple(
            sorted(
                name
                for name, tally in outcomes[last_active_cycle].items()
                if tally.get("num_attempts", 0) >= 1
            )
        )
        return attempted, sum(transitions[: last_active_cycle + 1])

    @staticmethod
    def ended_on_a_toss(
        *,
        evaluations: list[tuple[int, int, int]],
        outcomes: list[dict[str, dict[str, int]]],
    ) -> bool:
        """Did this run's last active practice cycle contain a `Toss` attempt?

        The **route** into stranding, which the score cannot see and the transition count
        cannot separate. A seed that threw the cube past the barrier and a seed that never
        got the cube off the floor both end at zero transitions per cycle forever, and
        only this tells them apart. Read from `stats.json` alone, so it is directly
        comparable against the committed 2026-08-08 runs.
        """
        _skills, _index = Tossing3DResetFree.last_practice_action(
            evaluations=evaluations, outcomes=outcomes
        )
        return "Toss" in _skills

    @staticmethod
    def load_final_practice_features(
        *, results_root: Path
    ) -> dict[str, dict[int, dict[str, float]]]:
        """`{arm: {seed: features}}` -- the newest recorded value of every environment
        feature, merged forward over that run's `sampler_draws.jsonl`.

        **Why merging forward is exact here rather than convenient.** A draw's `achieved`
        carries only the objects its own ground skill binds, so a `MoveToThrowPose` draw
        has no barrier in it. The robot and the cube are bound by all three skills and so
        always come from the newest draw; the barrier and the bin are immovable within an
        episode, and a reset-free run is a single episode from `hard_reset` to the end. So
        the merge is the final state, not an approximation of it.

        Runs without the file (the flag is opt-in) are absent rather than empty, for the
        same reason `load_arms` skips a run with no `stats.json`: an empty feature dict
        would abstract to "nothing holds", which reads as a stranded robot.
        """
        features: dict[str, dict[int, dict[str, float]]] = {}
        for arm in (SCHEDULED, NEVER):
            per_seed: dict[int, dict[str, float]] = {}
            for path in sorted((results_root / arm).rglob(SAMPLER_DRAWS_FILENAME)):
                if not path.parent.name.isdigit():
                    continue
                merged: dict[str, float] = {}
                for line in path.read_text().splitlines():
                    if line.strip():
                        merged.update(json.loads(line)["achieved"])
                if merged:
                    per_seed[int(path.parent.name)] = merged
            if per_seed:
                features[arm] = per_seed
        return features

    @staticmethod
    def build_state(*, features: dict[str, float]) -> State:
        """A `State` over this domain's four objects from a flat `"<object>.<feature>"`
        mapping -- `SamplerDrawRecorder.read_features` run backwards."""
        return State(
            data={
                obj: np.array([features[f"{obj.name}.{name}"] for name in obj.type.feature_names])
                for obj in _OBJECTS
            }
        )

    @staticmethod
    def held_predicates(*, features: dict[str, float]) -> dict[str, bool]:
        """Which of this domain's predicates hold in that state, evaluated with the
        domain's **own** classifiers rather than a copy of their arithmetic.

        That matters more than it looks: `HandEmpty` is `gripper ~ 0`, `Holding` needs
        `gripper > GRASP_THRESHOLD` **and** the cube above `HOLDING_HEIGHT`, and the gap
        between them -- a shut gripper with nothing in it -- is exactly the state this
        analysis exists to name. Re-deriving those thresholds here would put the finding
        at the mercy of a transcription error.

        Keyed by predicate *name*, which is only safe because this domain grounds each
        predicate exactly one way: one robot, one cube, one barrier, one bin. A domain
        with two cubes would need the whole ground atom as the key.
        """
        state = Tossing3DResetFree.build_state(features=features)
        held: dict[str, bool] = {}
        for skill in _SKILLS:
            for atom in skill.preconditions:
                ground = atom.ground(
                    substitution={
                        variable: _OBJECT_BY_TYPE[variable.type] for variable in atom.variables
                    }
                )
                held[ground.predicate.name] = bool(ground.predicate.holds(state, ground.objects))
        return held

    @staticmethod
    def applicable_skills(*, features: dict[str, float]) -> dict[str, bool]:
        """Which of the three skills the robot could still execute -- every precondition
        satisfied -- in that state.

        The preconditions are read off `Tossing3DSkills` itself, so this cannot drift from
        the operators a `Method` actually plans with. All three false is the domain fact
        the reset-free arm dies of: there is no action left, and nothing in this domain
        brings the world back.
        """
        held = Tossing3DResetFree.held_predicates(features=features)
        return {
            skill.name: all(held[atom.predicate.name] for atom in skill.preconditions)
            for skill in _SKILLS
        }

    @staticmethod
    def stranding_onset(*, transitions: list[int]) -> int | None:
        """The first cycle of the **terminal** run of zero-transition cycles, or `None`.

        Terminal-from-here, not "the first gap" -- the same definition
        `pickup_weight_stranding.py` uses on Tossing Room, kept identical so the two
        experiments can be read side by side. A run that takes no step for one period and
        then resumes was never stranded, and calling it stranded would promote ordinary
        exploration noise into the effect being claimed. A run that never moves at all
        strands at cycle 0; a run that moves in its last period reports `None`.
        """
        onset: int | None = None
        for index in range(len(transitions) - 1, -1, -1):
            if transitions[index] != 0:
                break
            onset = index
        return onset

    @staticmethod
    def late_scores(*, curves: dict[int, list[tuple[int, int, int]]]) -> dict[int, float]:
        """Each seed's mean solved count over the last `WINDOW` sweeps.

        A float, not a count, because it is a mean of counts. Callers that need an `x/y`
        pool it back up against the same denominator -- see `pooled`.
        """
        return {
            seed: statistics.fmean(solved for _t, solved, _n in evaluations[-WINDOW:])
            for seed, evaluations in curves.items()
        }

    @staticmethod
    def pooled(*, scores: dict[int, float], num_total: int) -> tuple[float, int]:
        """`(x, y)` for an `x/y` over every seed: the summed mean-solved against the
        summed denominator. `x` stays a float because each seed's contribution is a
        window mean, and rounding it here would hide that."""
        return sum(scores.values()), num_total * len(scores)

    @staticmethod
    def shared_seeds(*, arms: dict[str, dict[int, list[tuple[int, int, int]]]]) -> list[int]:
        """Seeds present in BOTH arms, which is what a paired test may use.

        Intersected rather than assumed equal: if one arm lost a run, pairing on the
        union would silently compare a seed against nothing.
        """
        return sorted(set(arms[SCHEDULED]) & set(arms[NEVER]))

    @staticmethod
    def report(*, results_root: Path) -> None:
        arms = Tossing3DResetFree.load_arms(results_root=results_root)
        missing = [arm for arm in (SCHEDULED, NEVER) if arm not in arms]
        if missing:
            print(f"No completed runs for {missing} under {results_root}")
            return
        seeds = Tossing3DResetFree.shared_seeds(arms=arms)
        num_total = arms[SCHEDULED][seeds[0]][0][2]
        for arm in (SCHEDULED, NEVER):
            sweeps = min(len(c) for c in arms[arm].values())
            print(f"{arm:>9}: {len(arms[arm])} seeds {sorted(arms[arm])}, {sweeps} sweeps each")
        print(f"paired on {len(seeds)}/{len(seeds)} shared seeds, {num_total} test tasks each")

        late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
        print(f"\nLATE window (last {WINDOW} sweeps):")
        for arm in (SCHEDULED, NEVER):
            x, y = Tossing3DResetFree.pooled(
                scores={s: late[arm][s] for s in seeds}, num_total=num_total
            )
            print(f"  {arm:>9}: {x:.1f}/{y}")

        differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)
        better = sum(1 for d in differences if d > 0)
        worse = sum(1 for d in differences if d < 0)
        print(f"\nnever - scheduled, paired over {len(differences)} seeds:")
        print(f"  never higher on {better}/{len(differences)}, lower on {worse}/{len(differences)}")
        print(f"  mean per-seed difference {statistics.fmean(differences):+.2f} tasks")
        print(f"  exact paired sign-flip p = {test.p_value:.6g}")
        print(f"  minimum detectable effect at 80% power: {mde:.2f} tasks per seed")

        print(f"\nper seed ({SCHEDULED} -> {NEVER}, difference):")
        for seed in seeds:
            d = late[NEVER][seed] - late[SCHEDULED][seed]
            print(
                f"  seed {seed}: {late[SCHEDULED][seed]:.1f} -> {late[NEVER][seed]:.1f}  ({d:+.1f})"
            )

        Tossing3DResetFree.report_stranding(arms=arms, seeds=seeds)

    @staticmethod
    def report_stranding(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], seeds: list[int]
    ) -> None:
        """Did the reset-free arm learn less, or did it stop practising altogether?

        Reported before any interpretation of the score gap, because the two support
        very different claims and the score alone cannot tell them apart.
        """
        print("\npractice actually taken (transitions per cycle):")
        for arm in (SCHEDULED, NEVER):
            per_seed = {
                seed: Tossing3DResetFree.transitions_per_cycle(evaluations=arms[arm][seed])
                for seed in seeds
            }
            num_cycles = len(next(iter(per_seed.values())))
            idle = {seed: sum(1 for t in steps if t == 0) for seed, steps in per_seed.items()}
            onsets = {
                seed: Tossing3DResetFree.stranding_onset(transitions=steps)
                for seed, steps in per_seed.items()
            }
            stranded = sum(1 for onset in onsets.values() if onset is not None)
            total = sum(sum(steps) for steps in per_seed.values())
            print(f"  {arm}:")
            print(f"    total transitions      {total} over {len(seeds)} seeds")
            print(f"    seeds ever stranded    {stranded}/{len(seeds)}")
            print(
                f"    idle cycles (0 steps)  "
                f"{sum(idle.values())}/{num_cycles * len(seeds)} across all seeds"
            )
            for seed in seeds:
                onset = onsets[seed]
                where = "never stranded" if onset is None else f"stranded from cycle {onset}"
                print(
                    f"      seed {seed}: {sum(per_seed[seed])} transitions, "
                    f"{idle[seed]}/{num_cycles} idle cycles, {where}"
                )

    @staticmethod
    def render_curves(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], output: Path
    ) -> None:
        """Both arms' learning curves: faint per-seed lines under a bold pooled mean.

        **Cycles, not online transitions, on the x axis.** The cycle is the controlled
        variable -- every seed ran exactly 100 -- while transitions vary per seed because
        a Tossing3D practice period ends early. Against cycles the seeds share a grid and
        the per-seed spread is readable.

        The per-seed haze is the point, not decoration: the measured sweep-to-sweep swing
        is several tasks, so a bold mean alone would imply a smoothness the data does not
        have, and with ten seeds a gap can be one seed wide.
        """
        fig, axes = plt.subplots(1, 1, figsize=(11.0, 6.2))
        num_total = arms[SCHEDULED][sorted(arms[SCHEDULED])[0]][0][2]
        for arm in (SCHEDULED, NEVER):
            curves = arms[arm]
            cycles = list(range(min(len(c) for c in curves.values())))
            colour = _ARM_COLOURS[arm]
            for seed in sorted(curves):
                axes.plot(
                    cycles,
                    [curves[seed][i][1] for i in cycles],
                    color=colour,
                    alpha=0.18,
                    linewidth=0.9,
                )
            pooled_curve = [statistics.fmean(curves[seed][i][1] for seed in curves) for i in cycles]
            x, y = Tossing3DResetFree.pooled(
                scores=Tossing3DResetFree.late_scores(curves=curves), num_total=num_total
            )
            axes.plot(
                cycles,
                pooled_curve,
                color=colour,
                linewidth=2.4,
                # `n=` per #188: neither arm splits into subgroups here -- every seed in
                # each arm behaves the same way -- so each gets one bold line and the
                # legend states the seed count rather than leaving it to be counted off
                # the haze.
                label=f"{arm} — mean, n={len(curves)}; last {WINDOW} sweeps {x:.1f}/{y}",
            )
        axes.axhline(
            ORACLE_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.8,
            label=ORACLE_LABEL,
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.8,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        axes.set_xlabel("practice cycle")
        # No `(x/N)` suffix on the axis label (#188) -- the denominator moves into the title.
        axes.set_ylabel("test tasks solved per seed")
        axes.set_title(
            f"Reset-free against scheduled practice on Tossing3D (of {num_total} test tasks) — "
            "bold pooled mean over faint per-seed lines",
            fontsize=10.5,
        )
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_paired(
        *, arms: dict[str, dict[int, list[tuple[int, int, int]]]], output: Path
    ) -> None:
        """One line per seed joining its `scheduled` LATE score to its `never` LATE score.

        Plotted per seed rather than as two bars because with ten seeds a bar chart of
        two means hides one seed driving the whole movement. Ten near-flat lines is a
        null result; ten lines sloping the same way is a real one.
        """
        seeds = Tossing3DResetFree.shared_seeds(arms=arms)
        num_total = arms[SCHEDULED][seeds[0]][0][2]
        late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
        differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]
        test = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)

        # Per-seed lines are coloured by direction of change (fell vs held/rose), not by arm
        # identity -- the same judgement as tossing3d_plateau.py's render_windows: a paired
        # before/after diff is a different kind of chart from the learning curves the
        # training-curve style section governs, so it keeps its pre-existing direction
        # colours rather than being remapped onto the SCHEDULED/NEVER arm palette, which
        # would misleadingly suggest this axis encodes an arm rather than a sign.
        fig, axes = plt.subplots(1, 1, figsize=(7.4, 6.2))
        for seed in seeds:
            fell = late[NEVER][seed] < late[SCHEDULED][seed]
            axes.plot(
                [0, 1],
                [late[SCHEDULED][seed], late[NEVER][seed]],
                marker="o",
                markersize=5,
                color="#b2182b" if fell else "#2166ac",
                alpha=0.75,
                linewidth=1.4,
            )
        axes.axhline(
            ORACLE_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.6,
            label=ORACLE_LABEL,
        )
        axes.axhline(
            RANDOM_SKILLS_PER_SEED,
            color=_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.6,
            label=RANDOM_SKILLS_LABEL,
        )
        axes.set_xticks([0, 1])
        axes.set_xticklabels([
            f"{SCHEDULED}\n(reset each cycle)",
            f"{NEVER}\n(one scene, no reset)",
        ])
        axes.set_xlim(-0.25, 1.25)
        axes.set_ylim(-num_total * 0.04, num_total * 1.08)
        # No `(x/N)` suffix on the axis label (#188) -- the denominator moves into the title.
        axes.set_ylabel(f"mean test tasks solved per seed over last {WINDOW} sweeps")
        fell_count = sum(1 for d in differences if d < 0)
        axes.set_title(
            f"What does a genuinely reset-free arm cost? (of {num_total} test tasks)\n"
            f"fell on {fell_count}/{len(differences)} seeds, "
            f"mean {statistics.fmean(differences):+.2f} tasks, "
            f"p = {test.p_value:.4g}, MDE {mde:.2f}",
            fontsize=10,
        )
        axes.grid(alpha=0.25, linewidth=0.6, axis="y")
        axes.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_practice(
        *,
        arms: dict[str, dict[int, list[tuple[int, int, int]]]],
        output: Path,
        outcomes: dict[str, dict[int, list[dict[str, dict[str, int]]]]] | None = None,
        annotate_seed: int = 0,
    ) -> None:
        """Cumulative practice transitions against cycle, per seed, for both arms.

        The figure that separates "learned less" from "practised less". A robot that
        keeps practising is a line that keeps rising; a stranded one is a line that goes
        flat and stays flat, and the cycle it flattens at is its stranding onset read
        straight off the axis. Per seed rather than pooled, because a mean over ten seeds
        with different onsets describes no seed.

        **Annotated when `outcomes` is given** (the CLI always supplies it): a marker on
        `annotate_seed`'s `never`-arm curve at its last real action, its transition index
        from `last_practice_action` -- i.e. from the per-cycle transition and skill-attempt
        record, never from where this curve visually goes flat -- plus a shaded region for
        every cycle after it, during which every `never`-arm seed sharing that onset
        recorded zero skill attempts. A second, small panel plots every `never`-arm seed's
        own last-action index, so the one seed singled out on the left is not mistaken for
        a universal number: it varies across seeds, because it depends on how many
        `MoveToThrowPose` draws the sampler needed -- unlike the *cycle* of stranding,
        which is uniform across seeds and is what the shaded region's count reports.

        **The annotation names the skills that cycle actually attempted, rather than
        assuming a `Toss`.** It assumed one until the 2026-08-13 pin bump, where 4/10
        never-arm seeds stranded having never thrown; see `last_practice_action`.
        """
        if outcomes is None:
            fig, axes = plt.subplots(1, 1, figsize=(9.0, 6.0))
            dist_axes = None
        else:
            fig, (axes, dist_axes) = plt.subplots(
                1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [3, 1.1]}
            )
        for arm in (SCHEDULED, NEVER):
            curves = arms[arm]
            colour = _ARM_COLOURS[arm]
            for index, seed in enumerate(sorted(curves)):
                cumulative = [t for t, _s, _n in curves[seed]]
                axes.plot(
                    range(len(cumulative)),
                    cumulative,
                    color=colour,
                    alpha=0.55,
                    linewidth=1.2,
                    label=arm if index == 0 else None,
                )

        if outcomes is not None:
            never_curves = arms[NEVER]
            last_actions = {
                seed: Tossing3DResetFree.last_practice_action(
                    evaluations=never_curves[seed], outcomes=outcomes[NEVER][seed]
                )
                for seed in sorted(never_curves)
            }
            toss_indices = {seed: index for seed, (_skills, index) in last_actions.items()}
            onsets = {
                seed: Tossing3DResetFree.stranding_onset(
                    transitions=Tossing3DResetFree.transitions_per_cycle(
                        evaluations=never_curves[seed]
                    )
                )
                for seed in sorted(never_curves)
            }
            marked_onset = onsets[annotate_seed]
            marked_index = toss_indices[annotate_seed]
            marked_skills = "+".join(last_actions[annotate_seed][0])
            same_onset = sum(1 for o in onsets.values() if o == marked_onset)
            total_seeds = len(onsets)
            tossed = sum(1 for skills, _index in last_actions.values() if "Toss" in skills)

            # This purple (`#762a83`) deliberately stays -- decided, not left by omission.
            # Before this change it was a real instance of "one hue meaning two different
            # things in one report": the same `#762a83` also coloured the NEVER-arm
            # `random-skills` reference line just above. That collision is what #188's rule
            # exists to cure, and it is gone now that the reference lines moved to
            # `_REFERENCE_COLOUR`. What remains is a single, now-unique use marking a
            # specific *event* (the stranding onset / last Toss), not an *arm* -- a
            # different semantic the training-curve-style section does not govern. Recolouring
            # it to the reference grey would blend a "look here, something happened" marker
            # into the "this is a flat, uninteresting ceiling" role grey is reserved for, and
            # `_REFERENCE_COLOUR` (`#666666`) is already doing a third, different job in this
            # same figure (`dist_axes`'s median line, below) -- so a genuinely distinct
            # highlight colour carries more information here than reuse would.
            axes.axvspan(
                marked_onset,
                len(next(iter(never_curves.values()))) - 1,
                color="#762a83",
                alpha=0.08,
                zorder=0,
                label=(
                    f"stranded from cycle {marked_onset}: 0 transitions/cycle onward "
                    f"({same_onset}/{total_seeds} never-arm seeds)"
                ),
            )
            axes.axvline(marked_onset, color="#762a83", linestyle="--", linewidth=1.3, zorder=1)
            axes.plot(
                [marked_onset],
                [marked_index],
                marker="o",
                markersize=9,
                markerfacecolor="#762a83",
                markeredgecolor="white",
                markeredgewidth=1.2,
                linestyle="none",
                zorder=5,
                label=(
                    f"seed {annotate_seed}: last action ({marked_skills}) "
                    f"at transition {marked_index}"
                ),
            )
            axes.annotate(
                f"seed {annotate_seed}: {marked_skills},\ntransition {marked_index}",
                xy=(marked_onset, marked_index),
                xytext=(marked_onset + 7, marked_index + 30),
                fontsize=8,
                color="#762a83",
                arrowprops={"arrowstyle": "->", "color": "#762a83", "linewidth": 1.0},
            )

            ordered_seeds = sorted(toss_indices)
            ys = list(range(len(ordered_seeds)))
            xs = [toss_indices[seed] for seed in ordered_seeds]
            dot_colours = [
                "#762a83" if seed == annotate_seed else "#c2a5cf" for seed in ordered_seeds
            ]
            median_index = statistics.median(toss_indices.values())
            assert dist_axes is not None  # narrows for mypy: only None branch skips this block
            dist_axes.scatter(xs, ys, c=dot_colours, s=42, zorder=3, edgecolors="none")
            dist_axes.axvline(median_index, color="#666666", linestyle=":", linewidth=1.0)
            dist_axes.set_yticks(ys)
            # The skill set, not just the seed number: whether a seed ever threw is the
            # thing that changed between pins, and a bare seed label would hide it.
            dist_axes.set_yticklabels(
                [f"seed {seed}: {'+'.join(last_actions[seed][0])}" for seed in ordered_seeds],
                fontsize=7.0,
            )
            dist_axes.set_xlabel("last-action transition index", fontsize=8.5)
            dist_axes.set_title(
                f"last practice action, every\nnever-arm seed (min {min(xs)}, "
                f"max {max(xs)},\nmedian {median_index:.1f}; "
                f"Toss on {tossed}/{total_seeds})",
                fontsize=8.2,
            )
            dist_axes.grid(alpha=0.2, linewidth=0.5, axis="x")
            dist_axes.tick_params(axis="both", labelsize=7.5)

        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_xlabel("practice cycle")
        axes.set_ylabel("cumulative practice transitions")
        axes.set_title(
            "Did the reset-free arm learn less, or practise less?\n"
            "one line per seed; a flat line is a robot that has stopped acting",
            fontsize=10.5,
        )
        axes.legend(fontsize=8, loc="upper left", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def render_stranding(
        *,
        arms: dict[str, dict[int, list[tuple[int, int, int]]]],
        outcomes: dict[str, dict[int, list[dict[str, dict[str, int]]]]],
        features: dict[str, dict[int, dict[str, float]]],
        output: Path,
    ) -> None:
        """*Why* each never-arm seed stopped: the route in, and the abstract state it
        stopped in.

        The left panel is one row per seed, ticked for every predicate that holds in the
        run's final recorded state; the right panel is the same seeds' applicable skills,
        which is what "stranded" actually means. Rows are ordered by route -- ended on a
        `Toss` first -- because the split between the routes is the finding this figure
        exists for, and a seed-ordered strip would scatter it.

        Not a curve, so the training-curve style's line rules do not apply; the two arm
        hues still do, and only `NEVER` appears here because the `SCHEDULED` arm never
        strands.
        """
        seeds = sorted(arms[NEVER])
        routes = {
            seed: Tossing3DResetFree.ended_on_a_toss(
                evaluations=arms[NEVER][seed], outcomes=outcomes[NEVER][seed]
            )
            for seed in seeds
        }
        ordered = sorted(seeds, key=lambda seed: (not routes[seed], seed))
        held = {
            seed: Tossing3DResetFree.held_predicates(features=features[NEVER][seed])
            for seed in ordered
        }
        applicable = {
            seed: Tossing3DResetFree.applicable_skills(features=features[NEVER][seed])
            for seed in ordered
        }
        predicate_names = sorted(next(iter(held.values())))
        skill_names = [skill.name for skill in _SKILLS]

        fig, (left, right) = plt.subplots(
            1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [2.1, 1]}
        )
        for axes, columns, table, title in (
            (left, predicate_names, held, "predicates holding in the final recorded state"),
            (right, skill_names, applicable, "skills still applicable"),
        ):
            for row, seed in enumerate(ordered):
                for column, name in enumerate(columns):
                    value = table[seed][name]
                    axes.scatter(
                        [column],
                        [row],
                        s=150,
                        marker="o" if value else "x",
                        color=_ARM_COLOURS[NEVER] if value else "#999999",
                        linewidths=1.6,
                        zorder=3,
                    )
            axes.set_xticks(range(len(columns)))
            axes.set_xticklabels(columns, rotation=30, ha="right", fontsize=8.5)
            axes.set_yticks(range(len(ordered)))
            axes.set_yticklabels(
                [
                    f"seed {seed} — {'ended on Toss' if routes[seed] else 'never threw'}"
                    for seed in ordered
                ],
                fontsize=8.0,
            )
            axes.set_xlim(-0.6, len(columns) - 0.4)
            axes.set_ylim(-0.6, len(ordered) - 0.4)
            axes.grid(alpha=0.25, linewidth=0.6)
            axes.set_title(title, fontsize=9.5)

        tossed = sum(1 for value in routes.values() if value)
        stuck = sum(1 for seed in ordered if not any(applicable[seed].values()))
        fig.suptitle(
            "Why the reset-free arm stops: two routes into the same dead end "
            f"(Toss on {tossed}/{len(ordered)} seeds, "
            f"no skill applicable on {stuck}/{len(ordered)})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output}")

    @staticmethod
    def report_stranding_routes(
        *,
        arms: dict[str, dict[int, list[tuple[int, int, int]]]],
        outcomes: dict[str, dict[int, list[dict[str, dict[str, int]]]]],
        features: dict[str, dict[int, dict[str, float]]],
    ) -> None:
        """The route split and the final abstract state, per never-arm seed."""
        seeds = sorted(arms[NEVER])
        tossed = 0
        stuck = 0
        print("\nhow the never arm stopped (route, then final abstract state):")
        for seed in seeds:
            route = Tossing3DResetFree.ended_on_a_toss(
                evaluations=arms[NEVER][seed], outcomes=outcomes[NEVER][seed]
            )
            tossed += route
            skills, index = Tossing3DResetFree.last_practice_action(
                evaluations=arms[NEVER][seed], outcomes=outcomes[NEVER][seed]
            )
            held = Tossing3DResetFree.held_predicates(features=features[NEVER][seed])
            applicable = Tossing3DResetFree.applicable_skills(features=features[NEVER][seed])
            stuck += not any(applicable.values())
            print(
                f"  seed {seed}: last cycle attempted {'+'.join(skills)} "
                f"(transition {index}), ended on Toss = {route}"
            )
            print(f"      holds:      {sorted(name for name, value in held.items() if value)}")
            print(
                f"      applicable: "
                f"{sorted(name for name, value in applicable.items() if value) or 'nothing'}"
            )
        print(f"  ended on a Toss on {tossed}/{len(seeds)} seeds")
        print(f"  no skill applicable in the final recorded state on {stuck}/{len(seeds)} seeds")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--curves-output", type=Path, default=None)
    parser.add_argument("--paired-output", type=Path, default=None)
    parser.add_argument("--practice-output", type=Path, default=None)
    parser.add_argument(
        "--stranding-output",
        type=Path,
        default=None,
        help=(
            "Why each never-arm seed stopped: its route in (did the last active cycle "
            "contain a Toss?) and the abstract state it stopped in, read off that run's "
            "own sampler_draws.jsonl. Needs --record-sampler-draws to have been passed."
        ),
    )
    parser.add_argument(
        "--annotate-seed",
        type=int,
        default=0,
        help=(
            "Which never-arm seed's last-action/stranding point --practice-output "
            "annotates with actual transition numbers. Default 0, the seed used as the "
            "worked example in this experiment's log, so annotating it keeps one seed "
            "consistent throughout rather than introducing a second arbitrary pick."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Tossing3DResetFree.report(results_root=args.results_root)
    outputs = (
        args.curves_output,
        args.paired_output,
        args.practice_output,
        args.stranding_output,
    )
    if any(output is not None for output in outputs):
        arms = Tossing3DResetFree.load_arms(results_root=args.results_root)
        if args.curves_output is not None:
            Tossing3DResetFree.render_curves(arms=arms, output=args.curves_output)
        if args.paired_output is not None:
            Tossing3DResetFree.render_paired(arms=arms, output=args.paired_output)
        if args.practice_output is not None:
            outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=args.results_root)
            Tossing3DResetFree.render_practice(
                arms=arms,
                output=args.practice_output,
                outcomes=outcomes,
                annotate_seed=args.annotate_seed,
            )
        if args.stranding_output is not None:
            outcomes = Tossing3DResetFree.load_practice_outcomes(results_root=args.results_root)
            features = Tossing3DResetFree.load_final_practice_features(
                results_root=args.results_root
            )
            Tossing3DResetFree.report_stranding_routes(
                arms=arms, outcomes=outcomes, features=features
            )
            Tossing3DResetFree.render_stranding(
                arms=arms, outcomes=outcomes, features=features, output=args.stranding_output
            )


if __name__ == "__main__":
    main()
