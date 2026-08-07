"""Post-run analysis for the reset-free-practice A/B on Tossing Room (split throws):
is the free reset at the top of every interaction period load-bearing, or was it only
tidiness?

`PracticeLoop` has always put the environment back to the freshly-sampled train task's
initial state at the top of each practice period (`--practice-reset-policy scheduled`,
the only behaviour that existed before this stack). `never` removes exactly that one
write: practice state runs continuously across period boundaries and across the train
task changing underneath it. The train-task *distribution* is untouched -- a task is
still drawn per period and still handed to `get_practice_policy` -- so the two arms
differ in one privileged state-write and nothing else.

Why the domain matters. Tossing Room's ledge is one-way: once the robot drops past it,
the pile is unreachable for the rest of the run. Under `scheduled` that costs at most
the remainder of one period; under `never` it is permanent, so a period that strands
itself takes every later period with it and practice stops generating throw experience
altogether. `docs/experiment-logs/2026-08-06-reset-free-practice-ab.md` holds the
pre-registration written before either sweep ran, including the prediction (`scheduled`
wins, by a large margin) and the one probe that pointed the other way.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never
runs a simulation or drives a `Method`). Four modes in two pairs, each pair the
condense-then-render shape of the sibling `tossingroom_reset_interval.py`, whose
`PairedTests` this imports rather than copies.

**The outcome side** -- what the two arms could do afterwards, on the held-out test set:

* `--arm NAME=DIR ... --aggregate-output JSON` condenses each sweep's
  `DIR/ees/<seed>/stats.json` into the committed per-family aggregate. Raw sweep
  directories live outside the repo and do not travel between machines, so the
  aggregate is the record that survives.
* `--arms-json JSON --output PNG --curves-output PNG --families-output PNG`
  regenerates every figure and every number in the experiment log from that aggregate
  alone.

**The practice side** -- what practice actually *did* differently, which is the half
that says *why* the outcome moved:

* `--trace NAME=JSON ... --practice-aggregate-output JSON` condenses the per-period
  lifted-skill traces (one file per arm, ~550 KB each and full of per-attempt sampler
  parameters) down to the per-seed, per-period, per-skill `[attempts, successes]`
  counts, which is all any number below is computed from.
* `--practice-json JSON --practice-output PNG` regenerates the practice-side figure and
  report from that condensed record.

Both arms spend **exactly the same practice budget** -- 14900 skill executions, 1490 per
seed, checked rather than assumed -- so the composition of that budget is the whole
comparison. The stranding mechanism predicts a specific composition shift, and this is
where it is visible: throw attempts collapse in `never` after the early cycles, and the
`PressTrash` down / `PressRecycling` up asymmetry is *positional* evidence for where the
robot ended up (the trash button is at room 6, past the one-way ledge and therefore
unreachable once stranded; the recycling button is at room 1, where a stranded robot
is). The robot's room is **not** directly instrumented in these traces, so that last
point is strong indirect evidence and is reported as such, never as an observation.

**The checks that decide whether any of it is worth reading, printed first.**
`num_practice_resets` is a *measurement* of resets as they happened, not a restatement
of the flag, so it is the manipulation check: 10 per `scheduled` run, 0 per `never`
run. The realised per-family test-set composition is asserted to be 14 TRASH / 14
RECYCLING / 2 EMPTY, not assumed. Achieved transitions are reported so "equal
experience" is measured too -- a period that ends early on `InteractionComplete` is not
charged the steps it did not take, and removing resets could plausibly have changed
that.

**Statistics.** Both arms ran the same fixed seeds 0..9, so the comparison is *paired*
and the headline test is `PairedTests.sign_flip` -- exact by enumerating the sign-flip
null in full, which needs no normal approximation, no continuity or tie correction and
no scipy (not a dependency here). Alongside it, `TwoProportionSensitivity` reports a
minimum detectable effect computed **per comparison from that comparison's own two
denominators**, because they differ by a factor of seven: 140 tasks per arm for a throw
family against 20 for EMPTY. A single project-wide MDE would flatter the small one.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import EvaluationBreakdown
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402
from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402

# Arm -> (--practice-reset-policy value, how a figure names it). Ordering is the
# ordering of every table and figure below: the incumbent first, the new behaviour
# second, so a slope drawn left-to-right reads as "what removing the reset did".
_ARM_LABELS = {
    "scheduled": "scheduled\n(reset at every period start)",
    "never": "never\n(practice state runs continuously)",
}
# The incumbent -- the only behaviour that existed before this stack, and therefore the
# reference every difference below is measured *from*.
_BASELINE_ARM = "scheduled"

# The cycle count and period length both arms ran with. `scheduled` takes exactly one
# free reset per cycle and `never` takes none, so these fix what the manipulation check
# expects; they are named rather than inlined because that expectation is derived from
# them and from nothing else.
_NUM_CYCLES = 10
_PERIOD_STEPS = 150
_EXPECTED_RESETS = {"scheduled": _NUM_CYCLES, "never": 0}

# --num-test-tasks both arms ran with, and therefore the composition the domain
# allocates for it (14 TRASH / 14 RECYCLING / 2 EMPTY).
_NUM_TEST_TASKS = 30

# Okabe-Ito blue/orange/green, in the fixed report order. Blue and orange are the
# widest-separated pair in that palette under deuteranopia and protanopia alike, which
# is why they carry the two ARMS (the comparison the reader has to make); green is only
# ever the third family panel and never sits adjacent to an arm colour. Every arm also
# gets its own marker, so identity is never colour-alone.
_ARM_STYLE = {"scheduled": ("#0072B2", "o"), "never": ("#D55E00", "s")}
_FAMILY_STYLE = {"TRASH": "#D55E00", "RECYCLING": "#0072B2", "EMPTY": "#009E73"}
# Report/figure order for the families: the two throw families first (where the
# mechanism under test lives), the no-throw control last.
_FAMILIES = ("TRASH", "RECYCLING", "EMPTY")

# The domain's seven lifted skills, in the report order used everywhere below: the two
# pickups, the mover, the two throws, the two buttons. Written out rather than derived
# from whichever names happen to appear in a trace file, so a skill that an arm never
# once attempted still gets a row reading 0 instead of silently vanishing -- "never
# attempted" is the single most informative cell in that table.
_LIFTED_SKILLS = (
    "PickupTrash",
    "PickupRecycling",
    "MoveRoom",
    "ThrowTrash",
    "ThrowRecycling",
    "PressTrash",
    "PressRecycling",
)
# The two skills whose disappearance is the mechanism under test. Pooled rather than
# tracked apart in the timeline, because stranding removes access to the pile and so
# kills both at once; the per-skill table below keeps them separate.
_THROW_SKILLS = ("ThrowTrash", "ThrowRecycling")
# The room each button sits in, and the fact that decides whether it is reachable. Only
# ever printed -- the traces do not record the robot's room, so this is what makes the
# Press asymmetry *interpretable*, not something derived from the data.
_BUTTON_ROOMS = {"PressTrash": 6, "PressRecycling": 1}

# (z_{0.025} + z_{0.20}): the standard two-sided, 80%-power constant, spelled out from
# its two halves so the 0.05/0.80 choice is visible rather than a magic 2.8.
_MDE_CONSTANT = 1.959963985 + 0.841621234

# Figures are saved on an explicit white canvas rather than matplotlib's transparent
# default, so a PNG dropped into a dark-themed PR or Notion page keeps readable axes
# instead of black text on black.
_FIGURE_FACECOLOR = "white"


def expected_denominators(*, num_test_tasks: int = _NUM_TEST_TASKS) -> dict[str, int]:
    """The deterministic per-family test-set composition of a `tossingroomsplit` run at
    30 test tasks -- 14 TRASH / 14 RECYCLING / 2 EMPTY.

    Asked of the domain (`TossingRoomSplitTasks.test_goal_type_counts`, public for
    exactly this) rather than hardcoded here, because a hardcoded copy is a second
    source of truth that goes stale silently. Nothing about the allocation depends on
    the seed or the layout, so a throwaway instance answers it.
    """
    counts = TossingRoomSplitTasks(
        env=TossingRoomSplitEnvironment(), num_test_tasks=num_test_tasks
    ).test_goal_type_counts()
    return {goal_type.name: count for goal_type, count in counts.items()}


class ResetPolicyReport:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    # ------------------------------------------------------------------ aggregation

    @staticmethod
    def aggregate(*, arm_dirs: dict[str, Path], method: str = "ees") -> dict:
        """Condenses raw sweep directories into the committed aggregate,
        `{arm: {seed: {"resets": int, "families": {family: [[transitions, solved,
        total], ...]}}}}`.

        Per-family rather than one overall curve because the prediction was
        per-family (RECYCLING damaged first, EMPTY least), and the realised reset count
        is carried alongside because without it the committed record could not answer
        "did the manipulation happen?" -- the first question this experiment has to
        survive.

        Reads each `stats.json` back through `Metrics.model_validate_json` rather than
        parsing the JSON by hand, per analysis/README.md.
        """
        aggregate: dict = {}
        for arm, root in arm_dirs.items():
            seeds: dict = {}
            for stats_path in sorted((root / method).glob("*/stats.json")):
                metrics = Metrics.model_validate_json(stats_path.read_text())
                if not metrics.breakdowns:
                    raise ValueError(
                        f"{stats_path} has no per-task breakdowns -- it predates "
                        f"Metrics.breakdowns, so per-family numbers cannot be recovered"
                    )
                curves: dict[str, list[list[int]]] = {family: [] for family in _FAMILIES}
                for breakdown in metrics.breakdowns:
                    counts = ResetPolicyReport._counts(breakdown=breakdown)
                    for family, (solved, total) in counts.items():
                        curves[family].append([breakdown.num_online_transitions, solved, total])
                seeds[stats_path.parent.name] = {
                    "resets": metrics.num_practice_resets,
                    "families": curves,
                }
            if not seeds:
                raise ValueError(f"no stats.json under {root / method}")
            aggregate[arm] = seeds
        return aggregate

    @staticmethod
    def _counts(*, breakdown: EvaluationBreakdown) -> dict[str, tuple[int, int]]:
        """family -> (num_solved, num_total) for one evaluation sweep."""
        counts: dict[str, list[int]] = {family: [0, 0] for family in _FAMILIES}
        for outcome in breakdown.outcomes:
            family = GoalFamilies.classify(goal=outcome.goal)
            counts[family][0] += int(outcome.solved)
            counts[family][1] += 1
        return {family: (solved, total) for family, (solved, total) in counts.items()}

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_arms(*, json_path: Path) -> dict:
        arms = json.loads(json_path.read_text())
        missing = sorted(set(_ARM_LABELS) - set(arms))
        if missing:
            raise ValueError(f"aggregate JSON is missing arms: {missing}")
        return arms

    @staticmethod
    def seeds(*, arms: dict) -> list[str]:
        """The seeds both arms share, sorted numerically. Pairing is only valid over
        these, so the intersection is taken rather than assumed."""
        shared: set[str] | None = None
        for seeds in arms.values():
            shared = set(seeds) if shared is None else shared & set(seeds)
        return sorted(shared or set(), key=int)

    @staticmethod
    def reset_counts(*, arms: dict, arm: str) -> list[int]:
        """Per-seed count of the free resets that actually happened -- the manipulation
        check, and the first thing the report prints. A design that silently did not
        vary what it claimed reports a clean-looking null."""
        return [arms[arm][seed]["resets"] for seed in ResetPolicyReport.seeds(arms=arms)]

    @staticmethod
    def num_sweeps(*, arms: dict) -> int:
        """Evaluation sweeps every run of every arm has. Taken as the minimum so an
        index is always valid; the runs here all have 11."""
        return min(
            len(arms[arm][seed]["families"]["TRASH"])
            for arm in _ARM_LABELS
            for seed in ResetPolicyReport.seeds(arms=arms)
        )

    @staticmethod
    def counts_at(
        *, arms: dict, arm: str, family: str, index: int | None = None
    ) -> list[tuple[int, int]]:
        """Per-seed `(solved, total)` for one family at one evaluation sweep --
        `index=None` meaning the LAST one -- ordered by seed, so the lists across arms
        are index-aligned for pairing.

        The counts are what `Metrics.breakdowns` actually recorded; every rate in this
        file is derived *from* these and never the other way round. That is not
        cosmetic: at 14 tasks a family rate can only land on multiples of ~7.1pp, and at
        2 tasks per seed EMPTY's perfect score is 20/20 over the arm, which is a very
        different claim from TRASH's 140.
        """
        counts = []
        for seed in ResetPolicyReport.seeds(arms=arms):
            triples = arms[arm][seed]["families"][family]
            triple = max(triples, key=lambda t: t[0]) if index is None else triples[index]
            counts.append((triple[1], triple[2]))
        return counts

    @staticmethod
    def overall_counts_at(
        *, arms: dict, arm: str, index: int | None = None
    ) -> list[tuple[int, int]]:
        """Per-seed `(solved, total)` over ALL families at one sweep -- the 30-task
        headline, summed from the three family counts rather than re-read, so it cannot
        disagree with the per-family breakdown it is printed beside."""
        totals = [(0, 0)] * len(ResetPolicyReport.seeds(arms=arms))
        for family in _FAMILIES:
            counts = ResetPolicyReport.counts_at(arms=arms, arm=arm, family=family, index=index)
            totals = [
                (running_solved + solved, running_total + total)
                for (running_solved, running_total), (solved, total) in zip(
                    totals, counts, strict=True
                )
            ]
        return totals

    @staticmethod
    def final_solved(*, arms: dict, arm: str) -> list[int]:
        """Per-seed tasks solved at the final sweep, out of 30 -- the paired unit the
        headline test runs on."""
        return [
            solved for solved, _total in ResetPolicyReport.overall_counts_at(arms=arms, arm=arm)
        ]

    @staticmethod
    def pooled_counts(
        *, arms: dict, arm: str, family: str | None = None, index: int | None = None
    ) -> tuple[int, int]:
        """`(solved, total)` at one sweep summed over every shared seed -- the
        cross-seed aggregate in its lossless form (10 seeds x 14 tasks = 140).

        `family=None` pools all three families (10 x 30 = 300). Descriptive only:
        pooling destroys the pairing every test here relies on. It is reported because
        a summed count is the honest rendering of "how much evidence is behind this
        number", which a percentage discards.
        """
        counts = (
            ResetPolicyReport.overall_counts_at(arms=arms, arm=arm, index=index)
            if family is None
            else ResetPolicyReport.counts_at(arms=arms, arm=arm, family=family, index=index)
        )
        return (sum(solved for solved, _ in counts), sum(total for _, total in counts))

    @staticmethod
    def rate(*, counts: tuple[int, int]) -> float:
        """A count rendered as a percentage. Never a substitute for the count -- every
        line in this report prints `x/y` and puts this in brackets after it."""
        solved, total = counts
        return 100.0 * solved / total if total else 0.0

    @staticmethod
    def paired_differences(*, arms: dict, family: str | None = None) -> list[float]:
        """Per-seed (never - scheduled) difference in final tasks solved, in TASKS.

        The sign convention is fixed here once and used everywhere: negative means
        removing the reset cost the agent tasks. Kept in tasks rather than percentage
        points because the paired unit is a count on a fixed 30-task denominator, so
        the count is exact and the percentage is a rendering of it.
        """
        per_arm = {
            arm: (
                ResetPolicyReport.overall_counts_at(arms=arms, arm=arm)
                if family is None
                else ResetPolicyReport.counts_at(arms=arms, arm=arm, family=family)
            )
            for arm in _ARM_LABELS
        }
        return [
            float(never - scheduled)
            for (scheduled, _), (never, _) in zip(
                per_arm["scheduled"], per_arm["never"], strict=True
            )
        ]

    @staticmethod
    def direction_counts(*, differences: list[float]) -> tuple[int, int, int]:
        """`(worse, tied, better)` seed counts for `never` against `scheduled`.

        Reported next to every p-value because a mean difference cannot tell "every
        seed moved a little" from "one seed moved a lot", and on this project the
        second has repeatedly been the true story.
        """
        worse = sum(1 for d in differences if d < 0)
        better = sum(1 for d in differences if d > 0)
        return (worse, len(differences) - worse - better, better)

    @staticmethod
    def curve(*, arms: dict, arm: str, family: str | None = None) -> list[tuple[float, list[int]]]:
        """`(transitions, per-seed solved counts)` per checkpoint.

        Returns the per-seed counts rather than a mean and an error bar, because the
        figure has to draw the per-seed spread (CLAUDE.md: a bar chart of two means
        hides one seed driving the whole effect) and the caller should not have to go
        back to the aggregate for it. x is the mean achieved transitions at that
        checkpoint -- `num_online_transitions` is data-driven, so seeds need not land
        on identical x-values even though these runs all do.
        """
        points = []
        for index in range(ResetPolicyReport.num_sweeps(arms=arms)):
            counts = (
                ResetPolicyReport.overall_counts_at(arms=arms, arm=arm, index=index)
                if family is None
                else ResetPolicyReport.counts_at(arms=arms, arm=arm, family=family, index=index)
            )
            transitions = [
                arms[arm][seed]["families"]["TRASH"][index][0]
                for seed in ResetPolicyReport.seeds(arms=arms)
            ]
            points.append((statistics.mean(transitions), [solved for solved, _ in counts]))
        return points

    # ------------------------------------------------------------------ the checks

    @staticmethod
    def achieved_transitions(*, arms: dict, arm: str) -> list[int]:
        """Per-seed total online transitions actually taken.

        The design intends 1500 in every run of both arms, and that is not automatic:
        a period ending early on `InteractionComplete` is not charged the steps it did
        not take, and a reset can revive a robot whose practice planner had nothing
        applicable left. So the arms could in principle have bought different amounts
        of experience. Measured, not assumed.
        """
        return [
            max(triple[0] for triple in arms[arm][seed]["families"]["TRASH"])
            for seed in ResetPolicyReport.seeds(arms=arms)
        ]

    @staticmethod
    def family_denominators(*, arms: dict, arm: str, family: str) -> list[int]:
        """How many test tasks of one family each seed actually held."""
        return [
            arms[arm][seed]["families"][family][0][2] for seed in ResetPolicyReport.seeds(arms=arms)
        ]

    @staticmethod
    def composition_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed, family) whose realised test-task count is not the designed
        one. Empty is the only acceptable result, and `_print_manipulation_checks`
        raises on anything else: a wrong denominator makes every rate below it wrong,
        and the EMPTY-first classification rule exists precisely because getting this
        wrong produces plausible-looking numbers rather than an error."""
        violations = []
        for arm in _ARM_LABELS:
            for family, expected in expected_denominators().items():
                counts = ResetPolicyReport.family_denominators(arms=arms, arm=arm, family=family)
                for seed, count in zip(ResetPolicyReport.seeds(arms=arms), counts, strict=True):
                    if count != expected:
                        violations.append(f"{arm} seed {seed} {family}: {count} != {expected}")
        return violations

    @staticmethod
    def reset_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed) whose realised `num_practice_resets` is not what the reset
        policy implies -- 10 for `scheduled` (one per cycle), 0 for `never`."""
        violations = []
        for arm, expected in _EXPECTED_RESETS.items():
            counts = ResetPolicyReport.reset_counts(arms=arms, arm=arm)
            for seed, count in zip(ResetPolicyReport.seeds(arms=arms), counts, strict=True):
                if count != expected:
                    violations.append(f"{arm} seed {seed}: {count} resets != {expected}")
        return violations

    # ------------------------------------------------------------------ figures

    @staticmethod
    def render_per_seed_figure(*, arms: dict, output: Path) -> None:
        """The primary figure: the 10 paired seeds, drawn so the PAIRING is visible.

        Left is a slope plot -- one line per seed connecting its `scheduled` and
        `never` final counts. A bar chart of two means would show the same difference
        and hide the thing that matters here, which is that the two seeds contributing
        nothing to the effect are the two that were already at the bottom under both
        policies. Where several seeds land on the same count the line stack is
        indistinguishable from one line, so the multiplicity is written next to it
        rather than jittered (jitter would misstate the data).

        Right is the paired difference itself, one dot per seed, with the mean and the
        exact sign-flip p. That is the statistic the headline claim rests on, so it is
        drawn rather than left in the table.
        """
        seeds = ResetPolicyReport.seeds(arms=arms)
        arm_order = list(_ARM_LABELS)
        fig, (slope_ax, diff_ax) = plt.subplots(1, 2, figsize=(13.0, 5.6))

        per_arm = {arm: ResetPolicyReport.final_solved(arms=arms, arm=arm) for arm in arm_order}
        positions = list(range(len(arm_order)))
        for index in range(len(seeds)):
            slope_ax.plot(
                positions,
                [per_arm[arm][index] for arm in arm_order],
                color="#666666",
                alpha=0.55,
                linewidth=1.3,
                zorder=2,
            )
        for position, arm in zip(positions, arm_order, strict=True):
            color, marker = _ARM_STYLE[arm]
            slope_ax.scatter(
                [position] * len(seeds),
                per_arm[arm],
                s=55,
                color=color,
                marker=marker,
                zorder=3,
                label=f"{arm} seeds",
            )
            # How many seeds share each y, written out: three overlapping dots at 18
            # and one dot at 18 are the same picture and very different evidence.
            for value in sorted(set(per_arm[arm])):
                shared = per_arm[arm].count(value)
                if shared > 1:
                    slope_ax.annotate(
                        f"x{shared}",
                        xy=(position, value),
                        xytext=(10 if position == 0 else -10, 0),
                        textcoords="offset points",
                        ha="left" if position == 0 else "right",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
            solved, total = ResetPolicyReport.pooled_counts(arms=arms, arm=arm)
            slope_ax.errorbar(
                [position],
                [statistics.mean(per_arm[arm])],
                yerr=[[statistics.stdev(per_arm[arm]) / len(seeds) ** 0.5]] * 2,
                marker="D",
                markersize=11,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.4,
                capsize=5,
                linewidth=2.0,
                zorder=4,
                label=f"{arm} mean ({solved}/{total} over all seeds)",
            )
        slope_ax.set_xticks(positions)
        slope_ax.set_xticklabels([_ARM_LABELS[arm] for arm in arm_order], fontsize=9)
        slope_ax.set_xlim(-0.45, len(arm_order) - 0.55)
        slope_ax.set_ylim(0, 31)
        slope_ax.set_ylabel(f"Final-sweep tasks solved, out of {_NUM_TEST_TASKS} per seed")
        slope_ax.set_title(
            f"Every seed, paired: one line per seed ({len(seeds)} seeds)\n"
            "same seeds, same budget, one privileged state-write apart"
        )
        slope_ax.grid(True, axis="y", alpha=0.3)
        # Upper right: the slopes run top-left to bottom-right, so that corner is the
        # only one a legend does not sit on top of a seed in.
        slope_ax.legend(loc="upper right", fontsize=8)

        differences = ResetPolicyReport.paired_differences(arms=arms)
        flip = PairedTests.sign_flip(differences=differences)
        worse, tied, better = ResetPolicyReport.direction_counts(differences=differences)
        diff_ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.3, zorder=1)
        # Fanned across x purely so equal differences do not overprint; x carries no
        # meaning and is unlabelled for that reason.
        for index, (seed, difference) in enumerate(zip(seeds, differences, strict=True)):
            diff_ax.scatter(
                [index],
                [difference],
                s=60,
                color=_ARM_STYLE["never"][0] if difference < 0 else "#666666",
                marker=_ARM_STYLE["never"][1] if difference < 0 else "o",
                zorder=3,
            )
            diff_ax.annotate(
                f"s{seed}",
                xy=(index, difference),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#444444",
            )
        mean = statistics.mean(differences)
        diff_ax.axhline(
            mean,
            color=_ARM_STYLE["never"][0],
            linewidth=2.0,
            label=f"mean {mean:+.1f} tasks",
        )
        diff_ax.set_xticks([])
        diff_ax.set_xlim(-0.7, len(seeds) - 0.3)
        # Room below the lowest seed for its own label, which hangs under its marker.
        diff_ax.set_ylim(min(differences) - 1.7, max(differences) + 1.0)
        diff_ax.set_ylabel(f"never minus scheduled (tasks, out of {_NUM_TEST_TASKS})")
        diff_ax.set_title(
            "The paired difference, per seed\n"
            f"never is worse in {worse}/{len(seeds)}, tied in {tied}/{len(seeds)}, "
            f"better in {better}/{len(seeds)}\n"
            f"exact two-sided sign-flip p = {flip.p_value:.4f}"
        )
        diff_ax.grid(True, axis="y", alpha=0.3)
        diff_ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            "Tossing Room (split throws), EES: is the per-period free reset "
            "load-bearing?\n"
            f"{_NUM_CYCLES} cycles x {_PERIOD_STEPS} steps, "
            f"{len(seeds)} paired seeds, {_NUM_TEST_TASKS} test tasks per seed",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)

    @staticmethod
    def render_curve_figure(*, arms: dict, output: Path) -> None:
        """Learning curves across the 11 evaluation sweeps, with every seed drawn.

        One shared panel, not one per arm: the question is when the two arms diverge,
        and that is a comparison a reader should not have to make across panels. The
        thin lines are the individual seeds -- required rather than decorative, since
        two mean curves cannot show whether the gap is the whole population or one
        collapsed seed.
        """
        seeds = ResetPolicyReport.seeds(arms=arms)
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for arm in _ARM_LABELS:
            color, marker = _ARM_STYLE[arm]
            points = ResetPolicyReport.curve(arms=arms, arm=arm)
            xs = [transitions for transitions, _ in points]
            for index in range(len(seeds)):
                ax.plot(
                    xs,
                    [counts[index] for _, counts in points],
                    color=color,
                    alpha=0.22,
                    linewidth=1.0,
                    zorder=2,
                )
            means = [statistics.mean(counts) for _, counts in points]
            solved, total = ResetPolicyReport.pooled_counts(arms=arms, arm=arm)
            ax.plot(
                xs,
                means,
                color=color,
                linewidth=2.6,
                marker=marker,
                markersize=6,
                zorder=3,
                label=f"{arm} (mean of {len(seeds)} seeds; final {solved}/{total})",
            )
            ax.annotate(
                f"{solved}/{total}",
                xy=(xs[-1], means[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
        ax.set_xlabel("Online practice transitions")
        ax.set_ylabel(f"Evaluation tasks solved, out of {_NUM_TEST_TASKS} per seed")
        ax.set_ylim(0, 31)
        ax.set_xlim(-60, 1720)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(
            "Tossing Room (split throws), EES: evaluation success against practice "
            "budget\n"
            "thin lines are individual seeds; end labels are tasks solved over all "
            f"{len(seeds)} seeds\n"
            "both arms share checkpoint 0 by construction -- neither has practiced yet",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)

    @staticmethod
    def render_family_figure(*, arms: dict, output: Path) -> None:
        """Per-family breakdown, one panel per family, per-seed spread in each.

        Panels rather than one grouped bar chart because the three families have
        different denominators (14 / 14 / 2 per seed) and a shared y-axis in tasks
        would silently rescale them. Each panel keeps the slope-plot idiom of the
        primary figure, so the same reading applies: a line is a seed.
        """
        seeds = ResetPolicyReport.seeds(arms=arms)
        denominators = expected_denominators()
        arm_order = list(_ARM_LABELS)
        fig, axes = plt.subplots(1, len(_FAMILIES), figsize=(14.0, 5.0))
        positions = list(range(len(arm_order)))
        for ax, family in zip(axes, _FAMILIES, strict=True):
            per_arm = {
                arm: [
                    solved
                    for solved, _ in ResetPolicyReport.counts_at(arms=arms, arm=arm, family=family)
                ]
                for arm in arm_order
            }
            for index in range(len(seeds)):
                ax.plot(
                    positions,
                    [per_arm[arm][index] for arm in arm_order],
                    color="#666666",
                    alpha=0.5,
                    linewidth=1.2,
                    zorder=2,
                )
            for position, arm in zip(positions, arm_order, strict=True):
                color, marker = _ARM_STYLE[arm]
                ax.scatter(
                    [position] * len(seeds),
                    per_arm[arm],
                    s=50,
                    color=color,
                    marker=marker,
                    zorder=3,
                    label=arm if family == _FAMILIES[0] else None,
                )
                for value in sorted(set(per_arm[arm])):
                    shared = per_arm[arm].count(value)
                    if shared > 1:
                        ax.annotate(
                            f"x{shared}",
                            xy=(position, value),
                            xytext=(9 if position == 0 else -9, 0),
                            textcoords="offset points",
                            ha="left" if position == 0 else "right",
                            va="center",
                            fontsize=8,
                            color=color,
                        )
                ax.plot(
                    [position],
                    [statistics.mean(per_arm[arm])],
                    marker="D",
                    markersize=10,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=1.3,
                    zorder=4,
                )
            pooled = {
                arm: ResetPolicyReport.pooled_counts(arms=arms, arm=arm, family=family)
                for arm in arm_order
            }
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [f"{arm}\n{pooled[arm][0]}/{pooled[arm][1]}" for arm in arm_order], fontsize=9
            )
            ax.set_xlim(-0.5, len(arm_order) - 0.5)
            ax.set_ylim(-0.6, denominators[family] + 0.9)
            ax.set_ylabel(f"Tasks solved, out of {denominators[family]} per seed")
            mde = TwoProportionSensitivity.minimum_detectable_effect(
                counts_a=pooled["scheduled"], counts_b=pooled["never"]
            )
            sensitivity = (
                "no sensitivity: both arms at the ceiling"
                if mde is None
                else f"MDE {mde:.1f}pp at {pooled['scheduled'][1]} vs {pooled['never'][1]} tasks"
            )
            ax.set_title(
                f"{family}  ({denominators[family]}/seed)\n{sensitivity}",
                color=_FAMILY_STYLE[family],
                fontsize=10,
            )
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].legend(loc="center left", fontsize=8)
        fig.suptitle(
            "Tossing Room (split throws), EES: final-sweep success by goal family, "
            f"{len(seeds)} paired seeds\n"
            "one line per seed; x-axis labels are tasks solved over all seeds; "
            "diamonds are arm means",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)


class PracticeSideReport:
    """The practice half of the same A/B: not what the two arms could do afterwards, but
    what their practice periods actually *did*, from the per-period lifted-skill tallies.

    A static-method container, never instantiated, same as `ResetPolicyReport` above.

    The condensed record it produces and reads is
    `{arm: {seed: {"periods": [{skill: [attempts, successes]}, ...]}}}` -- one entry per
    interaction period, in cycle order, and `[attempts, successes]` pairs for the same
    reason the outcome aggregate stores `[transitions, solved, total]` triples: the
    counts are what was recorded, and every number below is derived from them rather
    than stored twice and allowed to disagree. A skill absent from a period's dict was
    not attempted in it; every accessor reads that as 0 rather than raising, since
    "never attempted" is a real and important measurement here.
    """

    # ------------------------------------------------------------------ aggregation

    @staticmethod
    def condense(*, trace_paths: dict[str, Path]) -> dict:
        """Condenses the raw per-period trace files into that record.

        The raw files are ~550 KB each -- most of it per-attempt sampler parameters
        (`greedy_forces`, `greedy_targets`, ...) that nothing here reads -- so they are
        deliberately not committed, and this is what survives in their place.

        Each file's own recorded `practice_reset_policy` is checked against the arm name
        it was passed under. Swapping the two files produces a perfectly plausible
        report with the conclusion reversed, which is exactly the class of error that
        gets published.
        """
        practice: dict = {}
        for arm, path in trace_paths.items():
            raw = json.loads(path.read_text())
            policy = raw.get("practice_reset_policy")
            if policy != arm:
                raise ValueError(
                    f"{path} records practice_reset_policy={policy!r} but was passed as arm {arm!r}"
                )
            practice[arm] = {
                str(entry["seed"]): {
                    "periods": [
                        {
                            skill: [tally["attempts"], tally["successes"]]
                            for skill, tally in period["skills"].items()
                        }
                        for period in entry["periods"]
                    ]
                }
                for entry in raw["seeds"]
            }
        return practice

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_practice(*, json_path: Path) -> dict:
        practice = json.loads(json_path.read_text())
        missing = sorted(set(_ARM_LABELS) - set(practice))
        if missing:
            raise ValueError(f"practice JSON is missing arms: {missing}")
        return practice

    @staticmethod
    def seeds(*, practice: dict) -> list[str]:
        """The seeds both arms share, sorted numerically -- same rule as the outcome
        side, since the same pairing applies."""
        shared: set[str] | None = None
        for seeds in practice.values():
            shared = set(seeds) if shared is None else shared & set(seeds)
        return sorted(shared or set(), key=int)

    @staticmethod
    def num_cycles(*, practice: dict) -> int:
        """Interaction periods every run of every arm has, taken as the minimum so an
        index is always valid."""
        return min(
            len(practice[arm][seed]["periods"])
            for arm in practice
            for seed in PracticeSideReport.seeds(practice=practice)
        )

    @staticmethod
    def attempts_at(*, practice: dict, arm: str, seed: str, cycle: int, skill: str | None) -> int:
        """Attempts of one skill (or, with `skill=None`, of every skill) in one seed's
        one period -- the single primitive every count below is summed from."""
        period = practice[arm][seed]["periods"][cycle]
        if skill is None:
            return sum(attempts for attempts, _successes in period.values())
        return period.get(skill, [0, 0])[0]

    @staticmethod
    def per_seed_attempts(*, practice: dict, arm: str, skill: str | None = None) -> list[int]:
        """Per-seed attempts over the whole run, ordered by seed."""
        return [
            sum(
                PracticeSideReport.attempts_at(
                    practice=practice, arm=arm, seed=seed, cycle=cycle, skill=skill
                )
                for cycle in range(PracticeSideReport.num_cycles(practice=practice))
            )
            for seed in PracticeSideReport.seeds(practice=practice)
        ]

    @staticmethod
    def pooled_attempts(*, practice: dict, arm: str, skill: str | None = None) -> int:
        """Attempts pooled over every shared seed. Descriptive, and honest only because
        the budget check guarantees the denominator is the same in both arms."""
        return sum(PracticeSideReport.per_seed_attempts(practice=practice, arm=arm, skill=skill))

    @staticmethod
    def pooled_successes(*, practice: dict, arm: str, skill: str) -> int:
        """Successes pooled the same way. Reported beside attempts because "practised a
        lot and kept failing" and "practised a lot and succeeded" are the same attempt
        count and completely different stories."""
        return sum(
            practice[arm][seed]["periods"][cycle].get(skill, [0, 0])[1]
            for seed in PracticeSideReport.seeds(practice=practice)
            for cycle in range(PracticeSideReport.num_cycles(practice=practice))
        )

    @staticmethod
    def throws_per_seed_per_cycle(*, practice: dict, arm: str) -> list[list[int]]:
        """`[seed][cycle]` throw attempts, both throw skills pooled -- the per-seed
        spread the timeline panel draws, kept per-seed for the same reason the outcome
        curves are: a pooled curve cannot tell a population from one surviving seed."""
        return [
            [
                sum(
                    PracticeSideReport.attempts_at(
                        practice=practice, arm=arm, seed=seed, cycle=cycle, skill=skill
                    )
                    for skill in _THROW_SKILLS
                )
                for cycle in range(PracticeSideReport.num_cycles(practice=practice))
            ]
            for seed in PracticeSideReport.seeds(practice=practice)
        ]

    @staticmethod
    def throws_per_cycle(*, practice: dict, arm: str) -> list[int]:
        """Throw attempts per cycle pooled over seeds."""
        rows = PracticeSideReport.throws_per_seed_per_cycle(practice=practice, arm=arm)
        return [sum(row[cycle] for row in rows) for cycle in range(len(rows[0]))]

    @staticmethod
    def live_throw_cycles(*, practice: dict, arm: str) -> list[int]:
        """Per-seed count of periods containing at least one throw attempt.

        A deliberately generous definition: one attempt in 149 skill executions counts
        the period as live. It is the *presence* of any throw experience that the
        stranding claim is about, and a threshold above 1 would be a free parameter
        chosen after seeing the data.
        """
        return [
            sum(1 for attempts in row if attempts > 0)
            for row in PracticeSideReport.throws_per_seed_per_cycle(practice=practice, arm=arm)
        ]

    @staticmethod
    def contributing_seeds(*, practice: dict, arm: str, cycle: int) -> list[str]:
        """Which seeds supplied a cycle's throw attempts.

        The attribution the pooled timeline cannot carry: a flat pooled tail reads as
        "the arm still practises a little" and can equally be one unstranded seed
        practising as much as ever while every other seed contributes nothing.
        """
        seeds = PracticeSideReport.seeds(practice=practice)
        rows = PracticeSideReport.throws_per_seed_per_cycle(practice=practice, arm=arm)
        return [seed for seed, row in zip(seeds, rows, strict=True) if row[cycle] > 0]

    # ------------------------------------------------------------------ the checks

    @staticmethod
    def budget_violations(*, practice: dict) -> list[str]:
        """Every seed whose two arms did not spend the same number of skill executions.

        This is the check the whole practice-side comparison depends on. Composition is
        only interpretable against a fixed budget: an arm that attempted `MoveRoom` more
        often could otherwise simply be an arm that acted more often. Empty is the only
        acceptable result, and the report raises on anything else.
        """
        violations = []
        baseline = _BASELINE_ARM
        reference = PracticeSideReport.per_seed_attempts(practice=practice, arm=baseline)
        for arm in practice:
            if arm == baseline:
                continue
            counts = PracticeSideReport.per_seed_attempts(practice=practice, arm=arm)
            for seed, count, expected in zip(
                PracticeSideReport.seeds(practice=practice), counts, reference, strict=True
            ):
                if count != expected:
                    violations.append(
                        f"seed {seed}: {arm} {count} attempts != {baseline} {expected}"
                    )
        return violations

    @staticmethod
    def period_budgets(*, practice: dict) -> list[int]:
        """The distinct per-period attempt totals over both arms, every seed and every
        period, sorted. A single value means every period bought exactly that many skill
        executions, which is what makes a per-period count readable as `x/y`."""
        return sorted({
            PracticeSideReport.attempts_at(
                practice=practice, arm=arm, seed=seed, cycle=cycle, skill=None
            )
            for arm in practice
            for seed in PracticeSideReport.seeds(practice=practice)
            for cycle in range(PracticeSideReport.num_cycles(practice=practice))
        })

    # ------------------------------------------------------------------ figure

    @staticmethod
    def render_practice_figure(*, practice: dict, output: Path) -> None:
        """The practice-side figure: what the identical budget was spent on.

        Three panels, left to right in the order the argument runs. (a) throw attempts
        per period, every seed drawn -- this is where `never`'s collapse after cycle 2
        is visible, and where a pooled-only plot would hide that its flat tail is a
        single seed. (b) the same thing reduced to one number per seed and PAIRED, so a
        reader can check the effect is the population rather than a mean. (c) the whole
        budget's composition per skill, per seed, which is where the positional
        `PressTrash`/`PressRecycling` asymmetry lives.
        """
        seeds = PracticeSideReport.seeds(practice=practice)
        cycles = PracticeSideReport.num_cycles(practice=practice)
        budgets = PracticeSideReport.period_budgets(practice=practice)
        per_period = budgets[0] if len(budgets) == 1 else None
        fig, (timeline_ax, live_ax, skill_ax) = plt.subplots(
            1, 3, figsize=(17.5, 5.9), gridspec_kw={"width_ratios": [1.25, 0.72, 1.35]}
        )

        PracticeSideReport._draw_timeline(
            ax=timeline_ax, practice=practice, cycles=cycles, per_period=per_period
        )
        PracticeSideReport._draw_live_cycles(ax=live_ax, practice=practice, cycles=cycles)
        PracticeSideReport._draw_skill_composition(ax=skill_ax, practice=practice)

        total = PracticeSideReport.pooled_attempts(practice=practice, arm=_BASELINE_ARM)
        fig.suptitle(
            "Tossing Room (split throws), EES: what the SAME practice budget was spent "
            "on\n"
            f"{len(seeds)} paired seeds x {cycles} periods, "
            f"{total} skill executions per arm -- identical in both, so the composition "
            "is the comparison",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)

    @staticmethod
    def _draw_timeline(*, ax, practice: dict, cycles: int, per_period: int | None) -> None:
        """(a) Throw attempts per period, thin per-seed lines behind a bold arm mean."""
        xs = list(range(cycles))
        for arm in _ARM_LABELS:
            color, marker = _ARM_STYLE[arm]
            rows = PracticeSideReport.throws_per_seed_per_cycle(practice=practice, arm=arm)
            for row in rows:
                ax.plot(xs, row, color=color, alpha=0.25, linewidth=1.0, zorder=2)
            means = [statistics.mean([row[cycle] for row in rows]) for cycle in xs]
            pooled = PracticeSideReport.pooled_attempts(practice=practice, arm=arm, skill=None)
            throws = sum(
                PracticeSideReport.pooled_attempts(practice=practice, arm=arm, skill=skill)
                for skill in _THROW_SKILLS
            )
            ax.plot(
                xs,
                means,
                color=color,
                linewidth=2.6,
                marker=marker,
                markersize=6,
                zorder=3,
                label=f"{arm} (mean of {len(rows)} seeds; {throws}/{pooled} attempts are throws)",
            )
        ax.set_xticks(xs)
        ax.set_xlabel("Interaction period (cycle)")
        ax.set_ylabel(
            "Throw attempts in that period, per seed"
            + (f",\nout of {per_period} skill executions per period" if per_period else "")
        )
        ax.set_xlim(-0.3, cycles - 0.7)
        # Headroom above the tallest seed line (a period is at most ~19 throws) so the
        # legend sits in empty space rather than on top of the data it describes.
        ax.set_ylim(-1.0, 25.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", fontsize=8, framealpha=1.0)
        ax.set_title(
            "(a) Throw practice over the run\n"
            "thin lines are individual seeds; never starts AHEAD and collapses",
            fontsize=10,
        )

    @staticmethod
    def _draw_live_cycles(*, ax, practice: dict, cycles: int) -> None:
        """(b) The same thing per seed and paired: periods with any throw attempt."""
        seeds = PracticeSideReport.seeds(practice=practice)
        arm_order = list(_ARM_LABELS)
        positions = list(range(len(arm_order)))
        per_arm = {
            arm: PracticeSideReport.live_throw_cycles(practice=practice, arm=arm)
            for arm in arm_order
        }
        for index in range(len(seeds)):
            ax.plot(
                positions,
                [per_arm[arm][index] for arm in arm_order],
                color="#666666",
                alpha=0.55,
                linewidth=1.3,
                zorder=2,
            )
        for position, arm in zip(positions, arm_order, strict=True):
            color, marker = _ARM_STYLE[arm]
            ax.scatter(
                [position] * len(seeds),
                per_arm[arm],
                s=55,
                color=color,
                marker=marker,
                zorder=3,
            )
            # Written out rather than jittered: six `never` seeds sit on 1/10, and six
            # overlapping dots look exactly like one.
            for value in sorted(set(per_arm[arm])):
                shared = per_arm[arm].count(value)
                if shared > 1:
                    ax.annotate(
                        f"x{shared}",
                        xy=(position, value),
                        xytext=(10 if position == 0 else -10, 0),
                        textcoords="offset points",
                        ha="left" if position == 0 else "right",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
            ax.plot(
                [position],
                [statistics.mean(per_arm[arm])],
                marker="D",
                markersize=11,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.4,
                zorder=4,
            )
        ax.set_xticks(positions)
        # Bare arm names plus the pooled count, the same idiom as the per-family figure
        # above -- and not the two-line policy descriptions the other figures use, which
        # collide in a panel this narrow. Panel (a)'s legend carries the description.
        ax.set_xticklabels(
            [f"{arm}\n{sum(per_arm[arm])}/{cycles * len(seeds)}" for arm in arm_order],
            fontsize=10,
        )
        ax.set_xlim(-0.5, len(arm_order) - 0.5)
        ax.set_ylim(-0.5, cycles + 1.4)
        ax.set_ylabel(f"Periods with >=1 throw attempt, out of {cycles} per seed")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(
            "(b) Live throw periods, paired by seed\n"
            "one line per seed; labels are periods over all seeds",
            fontsize=10,
        )

    @staticmethod
    def _draw_skill_composition(*, ax, practice: dict) -> None:
        """(c) The whole budget by skill, one dot per seed, both arms on one row pair.

        A symlog x-axis, because the same budget covers 11356 `MoveRoom` attempts and 9
        `PickupRecycling` ones and a linear axis renders the second as nothing; symlog
        rather than log because zero is a value several seeds genuinely take, and it is
        the most informative value in the panel. Vertical offset within a skill's band
        separates the two arms and carries no other meaning.
        """
        arm_order = list(_ARM_LABELS)
        for index, skill in enumerate(_LIFTED_SKILLS):
            for offset, arm in zip((0.17, -0.17), arm_order, strict=True):
                color, marker = _ARM_STYLE[arm]
                counts = PracticeSideReport.per_seed_attempts(
                    practice=practice, arm=arm, skill=skill
                )
                ax.scatter(
                    counts,
                    [index + offset] * len(counts),
                    s=32,
                    color=color,
                    marker=marker,
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                    label=arm if index == 0 else None,
                )
                pooled = PracticeSideReport.pooled_attempts(practice=practice, arm=arm, skill=skill)
                total = PracticeSideReport.pooled_attempts(practice=practice, arm=arm)
                ax.annotate(
                    f"{pooled}/{total}",
                    xy=(1.005, (index + offset)),
                    xycoords=("axes fraction", "data"),
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color=color,
                    annotation_clip=False,
                )
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(-0.4, 30000)
        ax.set_yticks(range(len(_LIFTED_SKILLS)))
        ax.set_yticklabels(
            [
                skill
                + (f"\n(button in room {_BUTTON_ROOMS[skill]})" if skill in _BUTTON_ROOMS else "")
                for skill in _LIFTED_SKILLS
            ],
            fontsize=8,
        )
        ax.set_ylim(-0.6, len(_LIFTED_SKILLS) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Practice attempts by that seed (symlog; 0 is shown)"
            "\nlabels at right are attempts pooled over all seeds"
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(
            "(c) The identical budget, by skill: one dot per seed\n"
            "PressTrash (room 6, past the ledge) down, PressRecycling (room 1) up",
            fontsize=10,
        )


class TwoProportionSensitivity(BaseModel):
    """The minimum detectable effect of one comparison, computed from **that
    comparison's own two denominators**.

    A single project-wide MDE would be wrong here by a factor of nearly three: a throw
    family pools 140 tasks per arm and EMPTY pools 20, so the sensitivity they support
    is not remotely the same. The formula is the standard two-sided, 80%-power
    normal-approximation one, evaluated at the OBSERVED rates rather than at a
    worst-case p = 0.5, so it says what this design could resolve where the data
    actually landed:

        MDE = (z_0.025 + z_0.20) * sqrt( p1(1-p1)/n1 + p2(1-p2)/n2 )

    **It degenerates, and the degenerate case must not be printed as a number.** When
    both arms sit at an extreme -- here EMPTY is 20/20 in both -- every variance term is
    zero and the formula returns 0.0pp, which reads as "this design could detect an
    arbitrarily small effect" and means the exact opposite: the normal approximation has
    no standard error to work with, and 20 tasks per arm support no inference at all.
    `minimum_detectable_effect` returns `None` there and every caller says so in words.
    """

    model_config = ConfigDict(frozen=True)

    @staticmethod
    def minimum_detectable_effect(
        *, counts_a: tuple[int, int], counts_b: tuple[int, int]
    ) -> float | None:
        """MDE in percentage points, or `None` when the normal approximation
        degenerates (both arms at 0 or at their ceiling, so the pooled variance is
        exactly zero)."""
        variance = 0.0
        for solved, total in (counts_a, counts_b):
            if total == 0:
                return None
            rate = solved / total
            variance += rate * (1.0 - rate) / total
        if variance == 0.0:
            return None
        return 100.0 * _MDE_CONSTANT * math.sqrt(variance)

    @staticmethod
    def observed_difference(*, counts_a: tuple[int, int], counts_b: tuple[int, int]) -> float:
        """(b - a) difference of the two pooled rates, in percentage points -- the
        number the MDE has to be read against. A difference of two rates, so pp is its
        correct unit; the counts it comes from are printed beside it everywhere."""
        return ResetPolicyReport.rate(counts=counts_b) - ResetPolicyReport.rate(counts=counts_a)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help='Repeatable, "scheduled=/path/to/sweep/root". Aggregation mode.',
    )
    parser.add_argument("--aggregate-output", type=Path, default=None)
    parser.add_argument("--arms-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Primary per-seed figure.")
    parser.add_argument("--curves-output", type=Path, default=None, help="Learning curves.")
    parser.add_argument("--families-output", type=Path, default=None, help="Per-family breakdown.")
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help='Repeatable, "scheduled=/path/to/traces-scheduled.json". Practice-side '
        "condensing mode.",
    )
    parser.add_argument("--practice-aggregate-output", type=Path, default=None)
    parser.add_argument("--practice-json", type=Path, default=None)
    parser.add_argument("--practice-output", type=Path, default=None, help="Practice-side figure.")
    return parser.parse_args()


def _parse_named_paths(*, entries: list[str], flag: str) -> dict[str, Path]:
    """`["scheduled=DIR", ...]` -> `{"scheduled": Path("DIR")}`."""
    parsed = {}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator:
            raise ValueError(f"{flag} must look like scheduled=PATH, got {entry!r}")
        parsed[name] = Path(path)
    return parsed


def _print_manipulation_checks(*, arms: dict) -> None:
    """Everything that decides whether the numbers below are worth reading, printed
    first and on purpose -- and RAISING rather than warning when it fails.

    A reset count that did not move means the flag did nothing and the whole
    comparison is a null by construction. A wrong family denominator means the goal
    classification mis-bucketed EMPTY into a throw family, which produces
    plausible-looking rates rather than an error.
    """
    seeds = ResetPolicyReport.seeds(arms=arms)
    print(f"Shared seeds ({len(seeds)}): {', '.join(seeds)}\n")

    print("MANIPULATION CHECK -- free resets that actually happened (Metrics.num_practice_resets):")
    print(f"{'arm':>10} {'expected':>9} {'observed (min..max)':>21} {'status':>9}")
    for arm, expected in _EXPECTED_RESETS.items():
        counts = ResetPolicyReport.reset_counts(arms=arms, arm=arm)
        ok = all(count == expected for count in counts)
        print(
            f"{arm:>10} {expected:>9} {f'{min(counts)}..{max(counts)}':>21} "
            f"{'OK' if ok else 'MISMATCH':>9}"
        )
    reset_problems = ResetPolicyReport.reset_violations(arms=arms)
    if reset_problems:
        raise ValueError(
            "the reset manipulation did not happen as designed: " + "; ".join(reset_problems)
        )

    print("\nCOMPOSITION CHECK -- test tasks per family per seed (designed 14/14/2):")
    for family, expected in expected_denominators().items():
        observed = sorted(
            set(ResetPolicyReport.family_denominators(arms=arms, arm=_BASELINE_ARM, family=family))
        )
        print(f"  {family:>10}: {observed} (expected {expected})")
    violations = ResetPolicyReport.composition_violations(arms=arms)
    print(f"  violations across both arms and every seed: {len(violations)}")
    if violations:
        raise ValueError(
            "the realised test-set composition is not the designed one, so every rate "
            "below would be computed on a wrong denominator: " + "; ".join(violations[:10])
        )

    print("\nEXPERIENCE CHECK -- online transitions actually taken (design: 1500 in every run):")
    for arm in _ARM_LABELS:
        achieved = ResetPolicyReport.achieved_transitions(arms=arms, arm=arm)
        matched = sum(1 for value in achieved if value == _NUM_CYCLES * _PERIOD_STEPS)
        print(
            f"  {arm:>10}: min {min(achieved)}, max {max(achieved)}, "
            f"exactly {_NUM_CYCLES * _PERIOD_STEPS} in {matched}/{len(achieved)} runs"
        )
    print(
        f"  Evaluation sweeps per run: {ResetPolicyReport.num_sweeps(arms=arms)} "
        "(checkpoint 0 is before any practice step, so both arms share it by "
        "construction)"
    )


def _print_final_success(*, arms: dict) -> None:
    seeds = ResetPolicyReport.seeds(arms=arms)
    denominators = expected_denominators()

    print(
        f"\nFINAL SWEEP, per-seed tasks solved out of {_NUM_TEST_TASKS} "
        "(the paired unit the headline test runs on):"
    )
    print(f"{'arm':>10} " + " ".join(f"{('s' + seed):>4}" for seed in seeds) + f" {'mean':>7}")
    for arm in _ARM_LABELS:
        solved = ResetPolicyReport.final_solved(arms=arms, arm=arm)
        print(
            f"{arm:>10} "
            + " ".join(f"{value:>4}" for value in solved)
            + f" {statistics.mean(solved):>7.1f}"
        )

    print(f"\nFINAL SWEEP, pooled over the {len(seeds)} shared seeds (solved / attempted):")
    print(f"{'family':>10} {'per seed':>9} {'scheduled':>16} {'never':>16} {'diff (pp)':>10}")
    for family in (*_FAMILIES, None):
        label = "OVERALL" if family is None else family
        per_seed = _NUM_TEST_TASKS if family is None else denominators[family]
        cells = []
        for arm in _ARM_LABELS:
            counts = ResetPolicyReport.pooled_counts(arms=arms, arm=arm, family=family)
            cells.append(f"{counts[0]}/{counts[1]} ({ResetPolicyReport.rate(counts=counts):.1f}%)")
        difference = TwoProportionSensitivity.observed_difference(
            counts_a=ResetPolicyReport.pooled_counts(arms=arms, arm="scheduled", family=family),
            counts_b=ResetPolicyReport.pooled_counts(arms=arms, arm="never", family=family),
        )
        print(f"{label:>10} {per_seed:>9} {cells[0]:>16} {cells[1]:>16} {difference:>+10.1f}")

    print("\nUNTRAINED BASELINE -- checkpoint 0, before either arm has taken a practice step:")
    for arm in _ARM_LABELS:
        counts = ResetPolicyReport.pooled_counts(arms=arms, arm=arm, index=0)
        print(
            f"  {arm:>10}: {counts[0]}/{counts[1]} ({ResetPolicyReport.rate(counts=counts):.1f}%)"
        )
    print(
        "  Identical by construction (no practice yet), so it is the floor both curves "
        "rise from\n  rather than a result."
    )


def _print_paired_tests(*, arms: dict) -> None:
    """The headline: an exact paired sign-flip test on the per-seed final counts.

    Exact by enumerating the whole sign-flip null (`PairedTests.sign_flip`, imported
    from the sibling reset-interval analysis rather than reimplemented -- a second copy
    of a hand-rolled significance test is exactly how a sign error gets published).
    """
    seeds = ResetPolicyReport.seeds(arms=arms)
    differences = ResetPolicyReport.paired_differences(arms=arms)
    worse, tied, better = ResetPolicyReport.direction_counts(differences=differences)
    flip = PairedTests.sign_flip(differences=differences)
    wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
    nonzero = [d for d in differences if d != 0.0]
    # Dropping the ties changes the fraction's denominator (2**8 rather than 2**10) but
    # not the p-value: a tied seed contributes the same factor to both the numerator and
    # the denominator. Reported in the compact form, and cross-checked against the full
    # enumeration so the claim "they agree" is measured rather than asserted.
    compact = PairedTests.sign_flip(differences=nonzero)
    if abs(compact.p_value - flip.p_value) > 1e-12:
        raise ValueError(
            f"dropping tied seeds changed the exact p-value ({compact.p_value} vs "
            f"{flip.p_value}); one of the two enumerations is wrong"
        )
    print(
        f"\nHEADLINE PAIRED TEST -- never minus scheduled, final tasks solved, {len(seeds)} seeds"
    )
    print(f"  per-seed differences (tasks): {[int(d) for d in differences]}")
    print(
        f"  never is worse in {worse}/{len(seeds)} seeds, tied in {tied}/{len(seeds)}, "
        f"better in {better}/{len(seeds)}"
    )
    print(
        f"  mean {statistics.mean(differences):+.1f} tasks "
        f"(sd {statistics.stdev(differences):.1f}) out of {_NUM_TEST_TASKS}"
    )
    print(
        f"  exact two-sided sign-flip p = {round(compact.p_value * 2 ** len(nonzero))}/"
        f"{2 ** len(nonzero)} = {compact.p_value:.4f}  "
        f"(over all {len(differences)} seeds including ties: "
        f"{round(flip.p_value * 2 ** len(differences))}/{2 ** len(differences)}, "
        "the same value)"
    )
    print(f"  exact Wilcoxon signed-rank p = {wilcoxon.p_value:.4f} (distribution-free companion)")

    print("\nPER-FAMILY paired tests (never minus scheduled, final tasks solved):")
    for family in _FAMILIES:
        family_differences = ResetPolicyReport.paired_differences(arms=arms, family=family)
        family_flip = PairedTests.sign_flip(differences=family_differences)
        family_worse, family_tied, family_better = ResetPolicyReport.direction_counts(
            differences=family_differences
        )
        print(
            f"  {family:>10}: mean {statistics.mean(family_differences):+5.1f} tasks, "
            f"worse {family_worse}/{len(seeds)}, tied {family_tied}/{len(seeds)}, "
            f"better {family_better}/{len(seeds)}, sign-flip p = {family_flip.p_value:.4f}"
        )


def _print_sensitivity(*, arms: dict) -> None:
    """What this design could have resolved, per comparison, from its own denominators.

    Printed for the significant comparisons as well as the null one: an observed effect
    smaller than the MDE is not a robust finding even at p < 0.05, and the reader can
    only check that if both numbers are on the page.
    """
    print("\nSENSITIVITY -- minimum detectable effect, each from its OWN two denominators")
    print(f"  constant (z_0.025 + z_0.20) = {_MDE_CONSTANT:.6f}; evaluated at the observed rates")
    print(f"{'comparison':>10} {'n per arm':>10} {'observed diff':>14} {'MDE':>9} {'verdict':>28}")
    for family in (*_FAMILIES, None):
        label = "OVERALL" if family is None else family
        scheduled = ResetPolicyReport.pooled_counts(arms=arms, arm="scheduled", family=family)
        never = ResetPolicyReport.pooled_counts(arms=arms, arm="never", family=family)
        difference = TwoProportionSensitivity.observed_difference(
            counts_a=scheduled, counts_b=never
        )
        mde = TwoProportionSensitivity.minimum_detectable_effect(counts_a=scheduled, counts_b=never)
        if mde is None:
            # Never "0.0pp": a zero here is the normal approximation collapsing, not a
            # design that can detect anything. Say so in words instead.
            print(
                f"{label:>10} {scheduled[1]:>10} {difference:>+13.1f}pp {'n/a':>9} "
                f"{'degenerate: see note below':>28}"
            )
            continue
        verdict = "above MDE" if abs(difference) >= mde else "BELOW MDE -- not resolvable"
        print(f"{label:>10} {scheduled[1]:>10} {difference:>+13.1f}pp {mde:>8.1f}pp {verdict:>28}")
    empty_scheduled = ResetPolicyReport.pooled_counts(arms=arms, arm="scheduled", family="EMPTY")
    empty_never = ResetPolicyReport.pooled_counts(arms=arms, arm="never", family="EMPTY")
    print(
        f"\n  EMPTY is {empty_scheduled[0]}/{empty_scheduled[1]} against "
        f"{empty_never[0]}/{empty_never[1]} -- both arms at the ceiling, so every "
        "variance term in the\n  MDE formula is exactly zero and it returns 0.0pp. That "
        "is the approximation breaking down,\n  not a sensitivity floor: a "
        f"{empty_scheduled[0]}/{empty_scheduled[1]}-vs-{empty_never[0]}/{empty_never[1]} "
        f"comparison at {empty_scheduled[1]} tasks per arm supports no inference in "
        "either\n  direction. It is reported as a control that did not move, and "
        "nothing more."
    )


def _print_report(*, arms: dict) -> None:
    _print_manipulation_checks(arms=arms)
    _print_final_success(arms=arms)
    _print_paired_tests(arms=arms)
    _print_sensitivity(arms=arms)


def _print_practice_checks(*, practice: dict) -> None:
    """The equal-budget check, printed first and RAISING on failure.

    Every practice-side number is a share of a budget, so a budget that differed between
    the arms would make "never attempted `MoveRoom` more" unfalsifiable -- it could just
    mean "never acted more". Measured per seed, not assumed from the flags.
    """
    seeds = PracticeSideReport.seeds(practice=practice)
    cycles = PracticeSideReport.num_cycles(practice=practice)
    print(f"Shared seeds ({len(seeds)}): {', '.join(seeds)}; {cycles} interaction periods each\n")

    print("BUDGET CHECK -- skill executions actually attempted (composition is only")
    print("comparable against a fixed budget):")
    for arm in _ARM_LABELS:
        per_seed = PracticeSideReport.per_seed_attempts(practice=practice, arm=arm)
        print(
            f"  {arm:>10}: {sum(per_seed)} attempts over {len(seeds)} seeds "
            f"(per seed min {min(per_seed)}, max {max(per_seed)})"
        )
    violations = PracticeSideReport.budget_violations(practice=practice)
    print(f"  seeds whose two arms differ: {len(violations)}")
    if violations:
        raise ValueError(
            "the two arms did not buy the same amount of practice, so their skill "
            "compositions are not comparable: " + "; ".join(violations[:10])
        )
    budgets = PracticeSideReport.period_budgets(practice=practice)
    print(
        f"  distinct per-period attempt totals: {budgets} "
        + (
            f"(every period bought exactly {budgets[0]})"
            if len(budgets) == 1
            else "(periods differ -- read per-period counts against their own total)"
        )
    )


def _print_practice_composition(*, practice: dict) -> None:
    """The per-skill table: the same budget, spent differently."""
    print("\nPRACTICE COMPOSITION -- attempts (successes) pooled over every shared seed:")
    print(f"{'skill':>16} {'scheduled':>22} {'never':>22} {'never - scheduled':>18}")
    for skill in _LIFTED_SKILLS:
        cells = []
        counts = {}
        for arm in _ARM_LABELS:
            attempts = PracticeSideReport.pooled_attempts(practice=practice, arm=arm, skill=skill)
            successes = PracticeSideReport.pooled_successes(practice=practice, arm=arm, skill=skill)
            total = PracticeSideReport.pooled_attempts(practice=practice, arm=arm)
            counts[arm] = attempts
            cells.append(f"{attempts}/{total} ({successes} ok)")
        difference = counts["never"] - counts["scheduled"]
        print(f"{skill:>16} {cells[0]:>22} {cells[1]:>22} {difference:>+18}")
    print(
        "\n  The robot's ROOM is not instrumented in these traces, so the button rows "
        "are strong\n  indirect evidence rather than an observation: PressTrash "
        f"(button in room {_BUTTON_ROOMS['PressTrash']}, past the\n  one-way ledge) falls "
        f"while PressRecycling (button in room {_BUTTON_ROOMS['PressRecycling']}) rises, "
        "which is what a robot\n  stranded on the recycling side of the ledge would "
        "produce."
    )


def _print_throw_timeline(*, practice: dict) -> None:
    """The timeline, and -- the part a pooled curve cannot carry -- which seeds are
    still supplying it."""
    seeds = PracticeSideReport.seeds(practice=practice)
    cycles = PracticeSideReport.num_cycles(practice=practice)
    throws = {
        arm: sum(
            PracticeSideReport.pooled_attempts(practice=practice, arm=arm, skill=skill)
            for skill in _THROW_SKILLS
        )
        for arm in _ARM_LABELS
    }
    print(
        f"\nTHROW PRACTICE OVER THE RUN ({' + '.join(_THROW_SKILLS)}), pooled over "
        f"{len(seeds)} seeds:"
    )
    print(f"{'arm':>10} " + " ".join(f"{('c' + str(cycle)):>4}" for cycle in range(cycles)))
    for arm in _ARM_LABELS:
        per_cycle = PracticeSideReport.throws_per_cycle(practice=practice, arm=arm)
        total = PracticeSideReport.pooled_attempts(practice=practice, arm=arm)
        print(
            f"{arm:>10} "
            + " ".join(f"{value:>4}" for value in per_cycle)
            + f"   total {throws[arm]}/{total}"
        )

    print(f"\nSEEDS STILL THROWING, per cycle (out of {len(seeds)}):")
    for arm in _ARM_LABELS:
        contributing = [
            PracticeSideReport.contributing_seeds(practice=practice, arm=arm, cycle=cycle)
            for cycle in range(cycles)
        ]
        print(
            f"{arm:>10} "
            + " ".join(f"{len(entry):>4}" for entry in contributing)
            + "   last cycle's seeds: "
            + (", ".join(f"s{seed}" for seed in contributing[-1]) or "none")
        )

    print(f"\nLIVE THROW PERIODS per seed (periods with >=1 throw attempt, out of {cycles}):")
    print(f"{'arm':>10} " + " ".join(f"{('s' + seed):>4}" for seed in seeds) + f" {'total':>9}")
    for arm in _ARM_LABELS:
        live = PracticeSideReport.live_throw_cycles(practice=practice, arm=arm)
        print(
            f"{arm:>10} "
            + " ".join(f"{value:>4}" for value in live)
            + f" {f'{sum(live)}/{cycles * len(seeds)}':>9}"
        )


def _print_practice_report(*, practice: dict) -> None:
    _print_practice_checks(practice=practice)
    _print_practice_composition(practice=practice)
    _print_throw_timeline(practice=practice)


def main() -> None:
    args = _parse_args()
    if args.trace:
        if args.practice_aggregate_output is None:
            raise ValueError("--trace requires --practice-aggregate-output")
        practice = PracticeSideReport.condense(
            trace_paths=_parse_named_paths(entries=args.trace, flag="--trace")
        )
        args.practice_aggregate_output.write_text(json.dumps(practice, sort_keys=True))
        print(f"wrote {args.practice_aggregate_output}")
        return

    if args.arm:
        if args.aggregate_output is None:
            raise ValueError("--arm requires --aggregate-output")
        aggregate = ResetPolicyReport.aggregate(
            arm_dirs=_parse_named_paths(entries=args.arm, flag="--arm")
        )
        # Compact, one line: this is recorded data, not source, and an indented
        # rendering is many times the bytes for no added readability.
        args.aggregate_output.write_text(json.dumps(aggregate, sort_keys=True))
        print(f"wrote {args.aggregate_output}")
        return

    if args.practice_json is not None:
        practice = PracticeSideReport.load_practice(json_path=args.practice_json)
        if args.practice_output is not None:
            PracticeSideReport.render_practice_figure(
                practice=practice, output=args.practice_output
            )
            print(f"wrote {args.practice_output}")
        _print_practice_report(practice=practice)
        return

    if args.arms_json is None:
        raise ValueError(
            "pass one of: --arm ... --aggregate-output, --trace ... "
            "--practice-aggregate-output, --practice-json, or --arms-json"
        )
    arms = ResetPolicyReport.load_arms(json_path=args.arms_json)
    for output, render in (
        (args.output, ResetPolicyReport.render_per_seed_figure),
        (args.curves_output, ResetPolicyReport.render_curve_figure),
        (args.families_output, ResetPolicyReport.render_family_figure),
    ):
        if output is not None:
            render(arms=arms, output=output)
            print(f"wrote {output}")
    _print_report(arms=arms)


if __name__ == "__main__":
    main()
