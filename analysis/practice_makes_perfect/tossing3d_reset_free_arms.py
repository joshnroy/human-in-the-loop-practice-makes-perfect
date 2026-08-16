"""What does a genuinely reset-free arm cost on Tossing3D?

> **WRITTEN FOR THE THREE-SKILL DOMAIN, AND IT FAILS QUIETLY ON A TWO-SKILL RUN.** The
> skill names this module keys on -- `Pick`, `MoveToThrowPose`, `Toss` -- no longer exist:
> the domain is now `PickCube` (no continuous parameters, so no sampler and no informed
> draws at all) and `MoveToTossLocationAndToss`. The per-skill tallies are read with
> `.get(skill_name)`, so a two-skill `stats.json` yields `0/0` everywhere and this module
> reports "never practiced" rather than raising. Do not run it against a two-skill sweep
> and read the output as a finding. Repointing the names is not enough -- this module's
> whole argument is about `MoveToThrowPose`'s informed draws against a `Pick` control, and
> neither side of that comparison survives -- so it is left as the record of what it
> measured. Its published results stand: see `docs/experiment-logs/`.

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

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
        transitions = Tossing3DResetFree.transitions_per_cycle(evaluations=evaluations)
        onset = Tossing3DResetFree.stranding_onset(transitions=transitions)
        if onset is None:
            raise ValueError("run never stranded -- no terminal toss to locate")
        last_active_cycle = onset - 1
        if outcomes[last_active_cycle].get("Toss", {}).get("num_attempts", 0) < 1:
            raise ValueError(
                f"cycle {last_active_cycle} (the last active cycle before stranding) "
                "recorded no Toss attempt -- the stranding is not attributable to a toss "
                "on this run, so annotating it as one would misdescribe the mechanism"
            )
        return sum(transitions[: last_active_cycle + 1])

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
                label=f"{arm} — last {WINDOW} sweeps {x:.1f}/{y}",
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
        `annotate_seed`'s `never`-arm curve at its one real action -- a `Toss` attempt, its
        transition index from `toss_transition_index`, i.e. from the per-cycle transition
        and skill-attempt record, never from where this curve visually goes flat -- plus a
        shaded region for every cycle after it, during which every `never`-arm seed sharing
        that onset recorded zero skill attempts. A second, small panel plots every
        `never`-arm seed's own toss-transition index, so the one seed singled out on the
        left is not mistaken for a universal number: it varies (3-13 across the ten real
        seeds), because it depends on how many `MoveToThrowPose` draws the sampler needed
        before succeeding -- unlike the *cycle* of stranding, which is uniform across seeds
        and is what the shaded region's count reports.
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
            toss_indices = {
                seed: Tossing3DResetFree.toss_transition_index(
                    evaluations=never_curves[seed], outcomes=outcomes[NEVER][seed]
                )
                for seed in sorted(never_curves)
            }
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
            same_onset = sum(1 for o in onsets.values() if o == marked_onset)
            total_seeds = len(onsets)

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
                label=f"seed {annotate_seed}: last action (Toss) at transition {marked_index}",
            )
            axes.annotate(
                f"seed {annotate_seed}: Toss,\ntransition {marked_index}",
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
            dist_axes.set_yticklabels([f"seed {seed}" for seed in ordered_seeds], fontsize=7.5)
            dist_axes.set_xlabel("toss transition index", fontsize=8.5)
            dist_axes.set_title(
                f"toss transition index, every\nnever-arm seed (min {min(xs)}, "
                f"max {max(xs)},\nmedian {median_index:.1f}; seed {annotate_seed} marked)",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--curves-output", type=Path, default=None)
    parser.add_argument("--paired-output", type=Path, default=None)
    parser.add_argument("--practice-output", type=Path, default=None)
    parser.add_argument(
        "--annotate-seed",
        type=int,
        default=0,
        help=(
            "Which never-arm seed's toss/stranding point --practice-output annotates "
            "with actual transition numbers. Default 0: its toss-transition index (7) "
            "is tied for closest to the median across all ten never-arm seeds (6.0), "
            "and it is the seed already used as the worked example in this experiment's "
            "log, so annotating it keeps one seed consistent throughout rather than "
            "introducing a second arbitrary pick."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Tossing3DResetFree.report(results_root=args.results_root)
    outputs = (args.curves_output, args.paired_output, args.practice_output)
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


if __name__ == "__main__":
    main()
