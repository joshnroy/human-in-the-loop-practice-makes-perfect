"""Post-run analysis for the 2x2 two-way-ledge experiment on Tossing Room (split
throws): is the cost of reset-free practice *caused* by the world's irreversible action,
or does it appear wherever the free reset is removed?

**Background.** PR #115 measured one A/B on this domain -- `--practice-reset-policy
scheduled` against `never`, ten paired seeds -- and found `never` much worse, attributing
it to stranding: the ledge out of room 2 is one-way, the pile (the only item source) is
in room 3, so a practice period that walks left once can never pick anything up again,
and under `never` that damage carries into every later period. That is a *mechanism*
claim, and the A/B alone cannot separate it from the weaker claim that continuous
practice state is simply worse. `--two-way-ledge` is the intervention that separates
them: it makes the ledge traversable rightward too, which removes the domain's only
irreversible action and nothing else about the reset policy.

So this reads back a full 2x2 -- world in {`one-way`, `two-way`} x policy in
{`scheduled`, `never`} -- and the quantity of interest is the **interaction**: the
reset-free penalty in the one-way world minus the reset-free penalty in the two-way one.
If stranding is the mechanism, the penalty collapses when the ledge opens.

**The two-way world is a different, easier domain, and that is why the interaction is the
statistic rather than any cross-world comparison of raw counts.** Turning the flag on
also stops EMPTY being an ordering task (its shortest solve drops 10 -> 9, so the
evaluation horizon drops 12 -> 11) and stops RECYCLING being one-attempt-per-period. A
two-way number placed beside a one-way one is therefore not a like-for-like comparison.
The penalty is a within-world difference, so both of its terms carry the same domain
difficulty and it cancels; the interaction is a difference of those differences. Nothing
below ever subtracts a two-way count from a one-way one.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs
a simulation or drives a `Method`). Two modes:

* `--arm NAME=DIR ... --aggregate-output JSON` condenses each sweep's
  `DIR/ees/<seed>/stats.json` into the committed aggregate. Raw sweep directories live
  outside the repo and do not travel between machines, so the aggregate is the record
  that survives.
* `--arms-json JSON --output PNG --penalty-output PNG --stranding-output PNG`
  regenerates every figure and every number in the experiment log from that aggregate
  alone.

**The arms are a square, not four strings.** Every arm name is derived from its (world,
policy) pair by `arm_name`, and `load_arms` raises when any of the four cells is absent.
A three-armed 2x2 does not merely lose precision -- the interaction is not defined at
all, and a report that silently prints the comparisons it *can* make would read as a
result.

**The checks that decide whether any of it is worth reading, printed first and raising.**
`num_practice_resets` is a *measurement* of resets as they happened rather than a
restatement of the flag, so it is the manipulation check: 10 per `scheduled` run, 0 per
`never` run, in both worlds. Achieved transitions are checked at 1500 so "equal
experience" is measured too. The realised per-family test composition is asserted to be
14 TRASH / 14 RECYCLING / 2 EMPTY (the flag does not change it, which is what makes the
two worlds' evaluation sets comparable in composition even though they differ in
difficulty).

**Stranding is measured, not assumed.** `Metrics.practice_outcomes_per_cycle` records
per-lifted-skill attempt tallies per window, so a period in which the robot attempted no
`Pickup*` and no `Throw*` at all is direct evidence that it could not reach the pile.
That is the definition used here, and the per-arm stranded-cycle counts are the
mechanism's own measurement rather than an inference from the outcome numbers.

**Statistics.** All four arms ran the same fixed seeds, so every comparison is *paired*
and the tests are `PairedTests.sign_flip` and `PairedTests.wilcoxon_signed_rank` --
exact by enumerating their nulls in full, needing no normal approximation, no continuity
or tie correction and no scipy (not a dependency here). They are imported from
`tossingroom_reset_interval` rather than reimplemented: a second copy of a hand-rolled
significance test is exactly how a sign error gets published. `TwoProportionSensitivity`
is imported from the sibling `tossingroomsplit_reset_policy` for the same reason, and
reports a minimum detectable effect computed **per comparison from that comparison's own
two denominators** -- 300 pooled tasks per arm against 20 for EMPTY, so a single
project-wide MDE would flatter the small one.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
from pydantic import BaseModel

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import EvaluationBreakdown
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from analysis.practice_makes_perfect.tossingroom_reset_interval import (  # noqa: E402
    PairedTests,
)
from analysis.practice_makes_perfect.tossingroomsplit_reset_policy import (  # noqa: E402
    TwoProportionSensitivity,
)

# The two factors of the 2x2. Ordering is the ordering of every table and figure below:
# the incumbent first in each (the one-way world is the domain every banked number on
# this project was measured in; `scheduled` is the only reset policy that existed before
# PR #115's stack), so a slope drawn left-to-right reads as "what the change did".
_WORLDS = ("one-way", "two-way")
_POLICIES = ("scheduled", "never")
_BASELINE_POLICY = "scheduled"

_WORLD_LABELS = {
    "one-way": "one-way ledge\n(the pile is unreachable once crossed)",
    "two-way": "two-way ledge (--two-way-ledge)\n(no irreversible action at all)",
}
_POLICY_LABELS = {
    "scheduled": "scheduled\n(reset at every period start)",
    "never": "never\n(practice state runs continuously)",
}

# The cycle count and period length every arm ran with. `scheduled` takes exactly one
# free reset per cycle and `never` takes none, so these fix what the manipulation check
# expects; they are named rather than inlined because that expectation is derived from
# them and from nothing else.
_NUM_CYCLES = 10
_PERIOD_STEPS = 150
_EXPECTED_TRANSITIONS = _NUM_CYCLES * _PERIOD_STEPS
_EXPECTED_RESETS = {"scheduled": _NUM_CYCLES, "never": 0}

# --num-test-tasks every arm ran with, and therefore the composition the domain
# allocates for it (14 TRASH / 14 RECYCLING / 2 EMPTY). `--two-way-ledge` does not move
# it, which is what keeps the two worlds' evaluation sets comparable in COMPOSITION even
# though the flag makes the two-way world easier.
_NUM_TEST_TASKS = 30

# Goal-family classification, as an ORDERED rule list, and the order is load-bearing.
# `Goal.describe()` renders the EMPTY family as
# "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)" -- it names BOTH bins, so
# a naive "does it mention recycling?" test swallows it and silently reports 16
# RECYCLING / 0 EMPTY. Matching the `BinEmpty` predicate first is what keeps the two
# throw families' denominators honest. `composition_violations` is the backstop: a
# misclassification shows up there as a wrong denominator rather than as a plausible
# wrong answer.
_FAMILY_RULES = (("BinEmpty", "EMPTY"), ("Trash", "TRASH"), ("Recycling", "RECYCLING"))
# Report/figure order for the families: the two throw families first (where the mechanism
# under test lives), the no-throw control last.
_FAMILIES = ("TRASH", "RECYCLING", "EMPTY")

# A skill whose name starts with one of these needs the robot to be at the pile, so a
# practice period recording zero attempts across all of them is a period in which the
# robot could not reach the only item source -- the stranding measurement. Prefixes
# rather than an enumerated skill list on purpose: the split-throw domain names them
# PickupTrash/PickupRecycling/ThrowTrash/ThrowRecycling, and a future item kind should be
# counted without editing this file.
_ITEM_SKILL_PREFIXES = ("Pickup", "Throw")

# Okabe-Ito blue/orange/green, in the fixed report order. Blue and orange are the
# widest-separated pair in that palette under deuteranopia and protanopia alike, which is
# why they carry the two WORLDS (the comparison the reader has to make); green is only
# ever the third family panel. Policy is carried by MARKER rather than by a third colour,
# so identity in the four-arm panels is never colour-alone and the two factors of the 2x2
# are visually separable from each other.
_WORLD_COLOR = {"one-way": "#0072B2", "two-way": "#D55E00"}
_WORLD_MARKER = {"one-way": "o", "two-way": "s"}
_POLICY_MARKER = {"scheduled": "o", "never": "X"}
_FAMILY_STYLE = {"TRASH": "#D55E00", "RECYCLING": "#0072B2", "EMPTY": "#009E73"}
# The stranded-cell highlight on the heatmap, and the one place a colour carries meaning
# on its own -- so it is also drawn as a box outline rather than a fill, and the cell's
# count (0) is printed inside it regardless.
_STRANDED_EDGE = "#D55E00"

# (z_{0.025} + z_{0.20}): the standard two-sided, 80%-power constant, spelled out from
# its two halves so the 0.05/0.80 choice is visible rather than a magic 2.8. Printed in
# the sensitivity header; `TwoProportionSensitivity` evaluates with its own copy, and
# `test_the_printed_constant_is_the_one_the_formula_actually_uses` pins the two together
# so the header cannot drift from the computation.
_MDE_CONSTANT = 1.959963985 + 0.841621234

# Figures are saved on an explicit white canvas rather than matplotlib's transparent
# default, so a PNG dropped into a dark-themed PR or Notion page keeps readable axes
# instead of black text on black.
_FIGURE_FACECOLOR = "white"


def arm_name(*, world: str, policy: str) -> str:
    """The arm name for one cell of the 2x2, derived from the pair that defines it.

    Deriving rather than listing four strings is what makes a missing or misspelled arm
    an error: `TwoWayLedgeReport.arms()` is the full square by construction, so
    `load_arms` can check completeness against something that cannot itself be
    incomplete. Four independent literals would let `two-way-nver` create a fifth arm
    and leave a cell empty, and the interaction is not defined on three arms.
    """
    if world not in _WORLDS:
        raise ValueError(f"{world!r} is not a world; expected one of {list(_WORLDS)}")
    if policy not in _POLICIES:
        raise ValueError(f"{policy!r} is not a reset policy; expected one of {list(_POLICIES)}")
    return f"{world}-{policy}"


def expected_denominators(*, num_test_tasks: int = _NUM_TEST_TASKS) -> dict[str, int]:
    """The deterministic per-family test-set composition of a `tossingroomsplit` run at
    30 test tasks -- 14 TRASH / 14 RECYCLING / 2 EMPTY.

    Asked of the domain (`TossingRoomSplitTasks.test_goal_type_counts`, public for
    exactly this) rather than hardcoded here, because a hardcoded copy is a second source
    of truth that goes stale silently. Nothing about the allocation depends on the seed,
    the layout or `--two-way-ledge`, so a throwaway instance answers it for all four arms.
    """
    counts = TossingRoomSplitTasks(
        env=TossingRoomSplitEnvironment(), num_test_tasks=num_test_tasks
    ).test_goal_type_counts()
    return {goal_type.name: count for goal_type, count in counts.items()}


class TwoWayLedgeReport:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project.

    The aggregate it produces and reads is

        {arm: {seed: {"resets": int,
                      "transitions": int,
                      "families": {family: [[transitions, solved, total], ...]},
                      "practice": [{skill: [attempts, successes]}, ...]}}}

    Counts rather than rates throughout, and `[attempts, successes]` pairs rather than a
    rate, for the same reason every line of the report prints `x/y`: the counts are what
    was recorded, and everything below is derived from them rather than stored twice and
    allowed to disagree. A skill absent from a practice window was not attempted in it;
    every accessor reads that as 0 rather than raising, since "never attempted" is the
    single most informative value in this experiment.
    """

    # ------------------------------------------------------------------ the 2x2 itself

    @staticmethod
    def arms() -> tuple[str, ...]:
        """Every cell of the square, worlds outer and policies inner -- the fixed order
        of every table and figure below."""
        return tuple(
            arm_name(world=world, policy=policy) for world in _WORLDS for policy in _POLICIES
        )

    @staticmethod
    def world_arms(*, world: str) -> tuple[str, str]:
        """One world's (scheduled, never) pair, in that order."""
        return (
            arm_name(world=world, policy=_BASELINE_POLICY),
            arm_name(world=world, policy="never"),
        )

    # ------------------------------------------------------------------ aggregation

    @staticmethod
    def aggregate(*, arm_dirs: dict[str, Path], method: str = "ees") -> dict:
        """Condenses raw sweep directories into the committed aggregate.

        Per-family curves because the outcome prediction is per-family; the realised
        reset count and transition total alongside them because without those the
        committed record could not answer "did the manipulation happen, and did both arms
        buy the same experience?" -- the first two questions this experiment has to
        survive. The per-window practice tallies because the stranding claim is measured
        from them and from nothing else.

        Reads each `stats.json` back through `Metrics.model_validate_json` rather than
        parsing the JSON by hand, per analysis/README.md.
        """
        unknown = sorted(set(arm_dirs) - set(TwoWayLedgeReport.arms()))
        if unknown:
            raise ValueError(
                f"{unknown} is not a cell of the 2x2; expected some of "
                f"{list(TwoWayLedgeReport.arms())}"
            )
        missing = sorted(set(TwoWayLedgeReport.arms()) - set(arm_dirs))
        if missing:
            raise ValueError(
                f"missing arms: {missing} -- the interaction is not defined on a partial "
                "square, so an incomplete set is an error rather than a smaller report"
            )
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
                if not metrics.practice_outcomes_per_cycle:
                    raise ValueError(
                        f"{stats_path} has no per-cycle practice tallies -- it predates "
                        f"Metrics.practice_outcomes_per_cycle, so stranding cannot be "
                        f"measured and would have to be inferred from the outcome"
                    )
                curves: dict[str, list[list[int]]] = {family: [] for family in _FAMILIES}
                for breakdown in metrics.breakdowns:
                    counts = TwoWayLedgeReport._counts(breakdown=breakdown)
                    for family, (solved, total) in counts.items():
                        curves[family].append([breakdown.num_online_transitions, solved, total])
                seeds[stats_path.parent.name] = {
                    "resets": metrics.num_practice_resets,
                    "transitions": max(
                        breakdown.num_online_transitions for breakdown in metrics.breakdowns
                    ),
                    "families": curves,
                    "practice": [
                        {
                            skill: [tally.num_attempts, tally.num_successes]
                            for skill, tally in window.items()
                        }
                        for window in metrics.practice_outcomes_per_cycle
                    ],
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
            family = TwoWayLedgeReport.classify(goal=outcome.goal)
            counts[family][0] += int(outcome.solved)
            counts[family][1] += 1
        return {family: (solved, total) for family, (solved, total) in counts.items()}

    @staticmethod
    def classify(*, goal: str) -> str:
        """The task family one `Goal.describe()` string belongs to.

        Walks `_FAMILY_RULES` in order, so EMPTY is tested for first -- see that constant
        for why the order is the whole point. An unmatched goal raises rather than
        bucketing as "other": it means a domain change or the wrong sweep directory, and
        a silent "other" bucket would quietly shrink a denominator.
        """
        for token, family in _FAMILY_RULES:
            if token in goal:
                return family
        raise ValueError(f"unrecognised goal description: {goal!r}")

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_arms(*, json_path: Path) -> dict:
        arms = json.loads(json_path.read_text())
        missing = sorted(set(TwoWayLedgeReport.arms()) - set(arms))
        if missing:
            raise ValueError(
                f"aggregate JSON is missing arms: {missing} -- the interaction between "
                "world and reset policy is undefined without all four cells"
            )
        return arms

    @staticmethod
    def seeds(*, arms: dict) -> list[str]:
        """The seeds every arm shares, sorted numerically. Pairing is only valid over
        these, so the intersection is taken rather than assumed."""
        shared: set[str] | None = None
        for seeds in arms.values():
            shared = set(seeds) if shared is None else shared & set(seeds)
        return sorted(shared or set(), key=int)

    @staticmethod
    def num_sweeps(*, arms: dict) -> int:
        """Evaluation sweeps every run of every arm has. Taken as the minimum so an index
        is always valid; the runs here all have 11."""
        return min(
            len(arms[arm][seed]["families"]["TRASH"])
            for arm in TwoWayLedgeReport.arms()
            for seed in TwoWayLedgeReport.seeds(arms=arms)
        )

    @staticmethod
    def practice_windows(*, arms: dict, arm: str, seed: str) -> int:
        """How many of one run's practice windows are real interaction periods.

        `Metrics.practice_outcomes_per_cycle` is index-aligned with `evaluations`, so a
        10-cycle run has **11** windows, not 10: windows 0-9 are the practice periods and
        window 10 is a trailing bookkeeping window recorded after the final evaluation
        sweep, with no practice in it. Measured on these sweeps, each of windows 0-9
        records 149 total skill attempts and window 10 records 0, for every seed of every
        arm.

        That trailing window is the difference between a stranding denominator of 100 and
        one of 110, and -- far worse -- it is stranded by definition (no `Pickup*` and no
        `Throw*` attempt), so counting it scores every seed of every arm as stranding
        once and turns a true 0/100 into 10/110.

        Derived from the record rather than by dropping the last element: a run that
        recorded a different number of trailing windows, or none, would silently lose or
        gain a real period under a hardcoded `-1`. A genuine practice period always
        records *something* (a stranded robot still walks and presses buttons -- 149
        attempts either way), so "the last window with any attempt at all" is the honest
        boundary, and `practice_window_violations` checks that what it drops really is a
        single empty trailing window.
        """
        windows = arms[arm][seed]["practice"]
        attempts = [sum(attempts for attempts, _successes in window.values()) for window in windows]
        return max((index for index, total in enumerate(attempts) if total > 0), default=-1) + 1

    @staticmethod
    def num_cycles(*, arms: dict) -> int:
        """Interaction periods every run of every arm has, taken as the minimum so an
        index is always valid. See `practice_windows` for why this is not simply the
        length of the practice record."""
        return min(
            TwoWayLedgeReport.practice_windows(arms=arms, arm=arm, seed=seed)
            for arm in TwoWayLedgeReport.arms()
            for seed in TwoWayLedgeReport.seeds(arms=arms)
        )

    @staticmethod
    def counts_at(
        *, arms: dict, arm: str, family: str, index: int | None = None
    ) -> list[tuple[int, int]]:
        """Per-seed `(solved, total)` for one family at one evaluation sweep --
        `index=None` meaning the LAST one -- ordered by seed, so the lists across arms are
        index-aligned for pairing.

        The counts are what `Metrics.breakdowns` actually recorded; every rate in this
        file is derived *from* these and never the other way round. That is not cosmetic:
        at 14 tasks a family rate can only land on multiples of ~7.1pp, and at 2 tasks per
        seed EMPTY pools to 20 over an arm, which is a very different claim from TRASH's
        140.
        """
        counts = []
        for seed in TwoWayLedgeReport.seeds(arms=arms):
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
        totals = [(0, 0)] * len(TwoWayLedgeReport.seeds(arms=arms))
        for family in _FAMILIES:
            counts = TwoWayLedgeReport.counts_at(arms=arms, arm=arm, family=family, index=index)
            totals = [
                (running_solved + solved, running_total + total)
                for (running_solved, running_total), (solved, total) in zip(
                    totals, counts, strict=True
                )
            ]
        return totals

    @staticmethod
    def final_solved(*, arms: dict, arm: str) -> list[int]:
        """Per-seed tasks solved at the final sweep, out of 30 -- the paired unit every
        test below runs on."""
        return [
            solved for solved, _total in TwoWayLedgeReport.overall_counts_at(arms=arms, arm=arm)
        ]

    @staticmethod
    def pooled_counts(
        *, arms: dict, arm: str, family: str | None = None, index: int | None = None
    ) -> tuple[int, int]:
        """`(solved, total)` at one sweep summed over every shared seed -- the cross-seed
        aggregate in its lossless form (10 seeds x 30 tasks = 300).

        `family=None` pools all three families. Descriptive only: pooling destroys the
        pairing every test here relies on. It is reported because a summed count is the
        honest rendering of "how much evidence is behind this number", which a percentage
        discards.
        """
        counts = (
            TwoWayLedgeReport.overall_counts_at(arms=arms, arm=arm, index=index)
            if family is None
            else TwoWayLedgeReport.counts_at(arms=arms, arm=arm, family=family, index=index)
        )
        return (sum(solved for solved, _ in counts), sum(total for _, total in counts))

    @staticmethod
    def rate(*, counts: tuple[int, int]) -> float:
        """A count rendered as a percentage. Never a substitute for the count -- every
        line in this report prints `x/y` and puts this in brackets after it."""
        solved, total = counts
        return 100.0 * solved / total if total else 0.0

    @staticmethod
    def reset_counts(*, arms: dict, arm: str) -> list[int]:
        """Per-seed count of the free resets that actually happened -- the manipulation
        check, and the first thing the report prints. A design that silently did not vary
        what it claimed reports a clean-looking null."""
        return [arms[arm][seed]["resets"] for seed in TwoWayLedgeReport.seeds(arms=arms)]

    @staticmethod
    def achieved_transitions(*, arms: dict, arm: str) -> list[int]:
        """Per-seed total online transitions actually taken.

        The design intends 1500 in every run of all four arms, and that is not automatic:
        a period ending early on `InteractionComplete` is not charged the steps it did not
        take, and `--two-way-ledge` changes what is reachable during practice. So the arms
        could in principle have bought different amounts of experience. Measured, not
        assumed.
        """
        return [arms[arm][seed]["transitions"] for seed in TwoWayLedgeReport.seeds(arms=arms)]

    @staticmethod
    def family_denominators(*, arms: dict, arm: str, family: str) -> list[int]:
        """How many test tasks of one family each seed actually held."""
        return [
            arms[arm][seed]["families"][family][0][2] for seed in TwoWayLedgeReport.seeds(arms=arms)
        ]

    # ------------------------------------------------------------------ the hard checks

    @staticmethod
    def reset_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed) whose realised `num_practice_resets` is not what its reset
        policy implies -- 10 for `scheduled` (one per cycle), 0 for `never`, in both
        worlds."""
        violations = []
        for arm in TwoWayLedgeReport.arms():
            expected = _EXPECTED_RESETS[arm.rsplit("-", maxsplit=1)[1]]
            counts = TwoWayLedgeReport.reset_counts(arms=arms, arm=arm)
            for seed, count in zip(TwoWayLedgeReport.seeds(arms=arms), counts, strict=True):
                if count != expected:
                    violations.append(f"{arm} seed {seed}: {count} resets != {expected}")
        return violations

    @staticmethod
    def transition_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed) that did not reach the full practice budget. Equal
        experience across the four arms is what makes their outcomes comparable at all."""
        violations = []
        for arm in TwoWayLedgeReport.arms():
            counts = TwoWayLedgeReport.achieved_transitions(arms=arms, arm=arm)
            for seed, count in zip(TwoWayLedgeReport.seeds(arms=arms), counts, strict=True):
                if count != _EXPECTED_TRANSITIONS:
                    violations.append(
                        f"{arm} seed {seed}: {count} transitions != {_EXPECTED_TRANSITIONS}"
                    )
        return violations

    @staticmethod
    def composition_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed, family) whose realised test-task count is not the designed
        one. Empty is the only acceptable result, and `_print_manipulation_checks` raises
        on anything else: a wrong denominator makes every rate below it wrong, and the
        EMPTY-first classification rule exists precisely because getting this wrong
        produces plausible-looking numbers rather than an error."""
        violations = []
        for arm in TwoWayLedgeReport.arms():
            for family, expected in expected_denominators().items():
                counts = TwoWayLedgeReport.family_denominators(arms=arms, arm=arm, family=family)
                for seed, count in zip(TwoWayLedgeReport.seeds(arms=arms), counts, strict=True):
                    if count != expected:
                        violations.append(f"{arm} seed {seed} {family}: {count} != {expected}")
        return violations

    @staticmethod
    def practice_window_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed) whose practice record is not shaped the way the cycle
        indexing assumes: one window per evaluation sweep, of which exactly one -- the
        trailing bookkeeping window -- is empty.

        Structural rather than scientific, and it raises for that reason. If the trailing
        window ever carried practice, or if more than one window were empty, `num_cycles`
        would count a different number of periods than the run actually ran and every
        cycle index in the stranding timeline would shift, in a way no reader of the
        figure could see.
        """
        violations = []
        for arm in TwoWayLedgeReport.arms():
            for seed in TwoWayLedgeReport.seeds(arms=arms):
                entry = arms[arm][seed]
                windows = entry["practice"]
                sweeps = len(entry["families"]["TRASH"])
                if len(windows) != sweeps:
                    violations.append(
                        f"{arm} seed {seed}: {len(windows)} practice windows != {sweeps} "
                        f"evaluation sweeps"
                    )
                    continue
                trailing = sum(attempts for attempts, _successes in windows[-1].values())
                if trailing != 0:
                    violations.append(
                        f"{arm} seed {seed}: trailing practice window has {trailing} "
                        f"attempts, expected 0"
                    )
                    continue
                cycles = TwoWayLedgeReport.practice_windows(arms=arms, arm=arm, seed=seed)
                if cycles != sweeps - 1:
                    violations.append(
                        f"{arm} seed {seed}: {cycles} practice periods with any recorded "
                        f"attempt != {sweeps - 1} expected"
                    )
        return violations

    # ------------------------------------------------------------------ the penalty

    @staticmethod
    def penalties(*, arms: dict, world: str) -> list[float]:
        """Per-seed reset-free penalty in one world: `scheduled` minus `never` final
        tasks solved, in TASKS.

        The sign convention is fixed here once and used everywhere: a POSITIVE penalty
        means removing the free reset cost the agent tasks. Kept in tasks rather than
        percentage points because the paired unit is a count on a fixed 30-task
        denominator, so the count is exact and the percentage is a rendering of it.

        Within-world by construction, which is what makes it comparable across worlds
        even though the two worlds are not equally difficult: both of its terms carry
        that world's difficulty, so it cancels.
        """
        scheduled_arm, never_arm = TwoWayLedgeReport.world_arms(world=world)
        scheduled = TwoWayLedgeReport.final_solved(arms=arms, arm=scheduled_arm)
        never = TwoWayLedgeReport.final_solved(arms=arms, arm=never_arm)
        return [float(a - b) for a, b in zip(scheduled, never, strict=True)]

    @staticmethod
    def pooled_penalty(*, arms: dict, world: str, family: str | None = None) -> float:
        """One world's penalty pooled over every seed, in percentage points -- the
        descriptive headline the per-seed vector is the evidence for."""
        scheduled_arm, never_arm = TwoWayLedgeReport.world_arms(world=world)
        scheduled = TwoWayLedgeReport.pooled_counts(arms=arms, arm=scheduled_arm, family=family)
        never = TwoWayLedgeReport.pooled_counts(arms=arms, arm=never_arm, family=family)
        return TwoWayLedgeReport.rate(counts=scheduled) - TwoWayLedgeReport.rate(counts=never)

    @staticmethod
    def interaction_differences(*, arms: dict) -> list[float]:
        """Per-seed `penalty(one-way) - penalty(two-way)` -- the experiment's headline
        quantity, and the only statistic here that combines the two worlds.

        It is a difference of two within-world differences, so the two-way world being
        easier does not contaminate it. Positive means the reset-free arm loses MORE when
        the world contains an irreversible action, which is what the stranding mechanism
        predicts; zero means removing the reset costs the same either way, which would
        leave the mechanism unsupported.
        """
        one_way = TwoWayLedgeReport.penalties(arms=arms, world="one-way")
        two_way = TwoWayLedgeReport.penalties(arms=arms, world="two-way")
        return [a - b for a, b in zip(one_way, two_way, strict=True)]

    @staticmethod
    def direction_counts(*, penalties: list[float]) -> tuple[int, int, int]:
        """`(worse, tied, better)` seed counts for `never` against `scheduled`, where
        `penalties` follows this file's sign convention (positive = the reset-free arm
        solved fewer).

        Reported next to every p-value because a mean difference cannot tell "every seed
        moved a little" from "one seed moved a lot", and on this project the second has
        repeatedly been the true story.
        """
        worse = sum(1 for value in penalties if value > 0)
        better = sum(1 for value in penalties if value < 0)
        return (worse, len(penalties) - worse - better, better)

    # ------------------------------------------------------------------ stranding

    @staticmethod
    def is_stranded(*, window: dict[str, list[int]]) -> bool:
        """Whether one practice period recorded zero attempts at every skill that needs
        the pile -- every `Pickup*` and every `Throw*`.

        The mechanism's own measurement, and deliberately generous in the other
        direction: a single attempt in a whole 150-step period makes the period live. It
        is the total ABSENCE of item experience that stranding predicts, and any
        threshold above 1 would be a free parameter chosen after seeing the data. A skill
        recorded with 0 attempts counts the same as one absent from the window entirely.
        """
        return not any(
            attempts > 0
            for skill, (attempts, _successes) in window.items()
            if skill.startswith(_ITEM_SKILL_PREFIXES)
        )

    @staticmethod
    def item_attempts(*, arms: dict, arm: str) -> list[list[int]]:
        """`[seed][cycle]` attempts at every `Pickup*`/`Throw*` skill pooled -- what the
        stranding heatmap draws, kept per-seed because the whole point is that a stranded
        run is a RUN of zeros in one seed's row, which a pooled per-cycle curve cannot
        show."""
        return [
            [
                sum(
                    attempts
                    for skill, (attempts, _successes) in arms[arm][seed]["practice"][cycle].items()
                    if skill.startswith(_ITEM_SKILL_PREFIXES)
                )
                for cycle in range(TwoWayLedgeReport.num_cycles(arms=arms))
            ]
            for seed in TwoWayLedgeReport.seeds(arms=arms)
        ]

    @staticmethod
    def stranded_cycles(*, arms: dict, arm: str) -> list[list[bool]]:
        """`[seed][cycle]` stranded flags, over the practice cycles only (see
        `num_cycles` for why the trailing window is excluded)."""
        return [
            [
                TwoWayLedgeReport.is_stranded(window=arms[arm][seed]["practice"][cycle])
                for cycle in range(TwoWayLedgeReport.num_cycles(arms=arms))
            ]
            for seed in TwoWayLedgeReport.seeds(arms=arms)
        ]

    @staticmethod
    def num_stranded_cycles(*, arms: dict, arm: str) -> int:
        """Stranded cycles over the whole arm -- the numerator of the `x/100` the report
        prints (10 seeds x 10 cycles)."""
        return sum(sum(row) for row in TwoWayLedgeReport.stranded_cycles(arms=arms, arm=arm))

    @staticmethod
    def seeds_that_strand(*, arms: dict, arm: str) -> list[str]:
        """Which seeds strand at least once.

        Reported beside the cycle count because the two answer different questions: 30
        stranded cycles out of 100 is "three seeds died early" or "every seed lost three
        periods", and only this tells them apart.
        """
        rows = TwoWayLedgeReport.stranded_cycles(arms=arms, arm=arm)
        return [
            seed
            for seed, row in zip(TwoWayLedgeReport.seeds(arms=arms), rows, strict=True)
            if any(row)
        ]

    @staticmethod
    def first_stranded_cycle(*, arms: dict, arm: str) -> list[int | None]:
        """Per-seed index of the first stranded cycle, or `None` for a seed that never
        strands -- `None` rather than -1 or the cycle count, either of which would sort
        and average as if it were a real cycle."""
        return [
            next((cycle for cycle, stranded in enumerate(row) if stranded), None)
            for row in TwoWayLedgeReport.stranded_cycles(arms=arms, arm=arm)
        ]

    # ------------------------------------------------------------------ figures

    @staticmethod
    def render_outcome_figure(*, arms: dict, output: Path) -> None:
        """The 2x2 outcome figure: four panels, one per family plus the 30-task overall,
        each showing all four arms with every seed drawn.

        Panels rather than one shared axis because the families have different
        denominators (30 / 14 / 14 / 2 per seed) and a shared y-axis in tasks would
        silently rescale them. Within a panel the four arms sit at four x-positions
        grouped by world, and a grey line joins each seed's `scheduled` and `never` points
        WITHIN a world -- the pairing that every test here runs on. No line crosses the
        gap between the worlds, because no comparison here ever does: the two-way world is
        easier for reasons that have nothing to do with the reset policy.
        """
        seeds = TwoWayLedgeReport.seeds(arms=arms)
        denominators = expected_denominators()
        arm_order = TwoWayLedgeReport.arms()
        positions = {
            arm: index + (0.0 if index < 2 else 0.6) for index, arm in enumerate(arm_order)
        }
        panels = (None, *_FAMILIES)
        fig, axes = plt.subplots(1, len(panels), figsize=(19.0, 5.8))

        for ax, family in zip(axes, panels, strict=True):
            per_seed = 30 if family is None else denominators[family]
            per_arm = {
                arm: [
                    solved
                    for solved, _ in (
                        TwoWayLedgeReport.overall_counts_at(arms=arms, arm=arm)
                        if family is None
                        else TwoWayLedgeReport.counts_at(arms=arms, arm=arm, family=family)
                    )
                ]
                for arm in arm_order
            }
            for world in _WORLDS:
                pair = TwoWayLedgeReport.world_arms(world=world)
                for index in range(len(seeds)):
                    ax.plot(
                        [positions[arm] for arm in pair],
                        [per_arm[arm][index] for arm in pair],
                        color="#666666",
                        alpha=0.5,
                        linewidth=1.2,
                        zorder=2,
                    )
            for arm in arm_order:
                world, policy = arm.rsplit("-", maxsplit=1)
                color = _WORLD_COLOR[world]
                ax.scatter(
                    [positions[arm]] * len(seeds),
                    per_arm[arm],
                    s=52,
                    color=color,
                    marker=_POLICY_MARKER[policy],
                    zorder=3,
                    label=(f"{world}, {policy}" if family is None else None),
                )
                # How many seeds share each y, written out: three overlapping dots at 18
                # and one dot at 18 are the same picture and very different evidence.
                for value in sorted(set(per_arm[arm])):
                    shared = per_arm[arm].count(value)
                    if shared > 1:
                        ax.annotate(
                            f"x{shared}",
                            xy=(positions[arm], value),
                            xytext=(0, 9),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7.5,
                            color=color,
                        )
                ax.plot(
                    [positions[arm]],
                    [statistics.mean(per_arm[arm])],
                    marker="D",
                    markersize=10,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=1.3,
                    zorder=4,
                )
            pooled = {
                arm: TwoWayLedgeReport.pooled_counts(arms=arms, arm=arm, family=family)
                for arm in arm_order
            }
            ax.set_xticks([positions[arm] for arm in arm_order])
            ax.set_xticklabels(
                [
                    f"{arm.rsplit('-', maxsplit=1)[0]}\n{arm.rsplit('-', maxsplit=1)[1]}\n"
                    f"{pooled[arm][0]}/{pooled[arm][1]}"
                    for arm in arm_order
                ],
                fontsize=8,
            )
            ax.axvline(
                (positions[arm_order[1]] + positions[arm_order[2]]) / 2.0,
                color="#bbbbbb",
                linewidth=1.0,
                linestyle="--",
                zorder=1,
            )
            ax.set_xlim(positions[arm_order[0]] - 0.55, positions[arm_order[3]] + 0.55)
            ax.set_ylim(-0.6, per_seed + max(1.0, per_seed * 0.12))
            ax.set_ylabel(f"Tasks solved / {per_seed} per seed")
            ax.grid(True, axis="y", alpha=0.3)
            title = "OVERALL" if family is None else family
            color = "#333333" if family is None else _FAMILY_STYLE[family]
            ax.set_title(f"{title}  ({per_seed}/seed)", color=color, fontsize=11)
        axes[0].legend(loc="lower left", fontsize=7.5)
        fig.suptitle(
            "Tossing Room (split throws), EES: the 2x2 of world x practice-reset policy\n"
            f"{len(seeds)} paired seeds, {_NUM_CYCLES} cycles x {_PERIOD_STEPS} steps, "
            f"{_NUM_TEST_TASKS} test tasks per seed; one grey line per seed, WITHIN a "
            "world only\n"
            "the two-way world is an easier domain, so only the within-world gaps are "
            "comparable across it",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)

    @staticmethod
    def render_penalty_figure(*, arms: dict, output: Path) -> None:
        """The headline figure: the reset-free penalty in each world, per seed.

        Left is the penalty itself as a slope plot -- one line per seed from its one-way
        penalty to its two-way penalty -- with zero marked, because "the penalty
        collapses when the ledge opens" is a claim about where those lines end up. Right
        is the interaction (the difference of those two penalties) per seed with the exact
        sign-flip p, which is the statistic the headline claim actually rests on.

        Per-seed points throughout and never a bare mean: the one-way `never`
        distribution is bimodal by construction -- a seed either strands or it does not --
        and a mean over a bimodal distribution describes no seed in the experiment.
        """
        seeds = TwoWayLedgeReport.seeds(arms=arms)
        per_world = {
            world: TwoWayLedgeReport.penalties(arms=arms, world=world) for world in _WORLDS
        }
        fig, (slope_ax, interaction_ax) = plt.subplots(1, 2, figsize=(13.5, 6.0))

        positions = list(range(len(_WORLDS)))
        slope_ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.4, zorder=1)
        for index in range(len(seeds)):
            slope_ax.plot(
                positions,
                [per_world[world][index] for world in _WORLDS],
                color="#666666",
                alpha=0.55,
                linewidth=1.3,
                zorder=2,
            )
        for position, world in zip(positions, _WORLDS, strict=True):
            values = per_world[world]
            slope_ax.scatter(
                [position] * len(seeds),
                values,
                s=58,
                color=_WORLD_COLOR[world],
                marker=_WORLD_MARKER[world],
                zorder=3,
            )
            for value in sorted(set(values)):
                shared = values.count(value)
                if shared > 1:
                    slope_ax.annotate(
                        f"x{shared}",
                        xy=(position, value),
                        xytext=(11 if position == 0 else -11, 0),
                        textcoords="offset points",
                        ha="left" if position == 0 else "right",
                        va="center",
                        fontsize=8,
                        color=_WORLD_COLOR[world],
                    )
            slope_ax.errorbar(
                [position],
                [statistics.mean(values)],
                yerr=[[statistics.stdev(values) / len(seeds) ** 0.5]] * 2,
                marker="D",
                markersize=11,
                color=_WORLD_COLOR[world],
                markeredgecolor="white",
                markeredgewidth=1.4,
                capsize=5,
                linewidth=2.0,
                zorder=4,
            )
        labels = []
        for world in _WORLDS:
            scheduled_arm, never_arm = TwoWayLedgeReport.world_arms(world=world)
            scheduled = TwoWayLedgeReport.pooled_counts(arms=arms, arm=scheduled_arm)
            never = TwoWayLedgeReport.pooled_counts(arms=arms, arm=never_arm)
            mde = TwoProportionSensitivity.minimum_detectable_effect(
                counts_a=scheduled, counts_b=never
            )
            sensitivity = (
                "degenerate; supports no inference"
                if mde is None
                else f"MDE {mde:.1f}pp at {scheduled[1]} vs {never[1]} tasks"
            )
            labels.append(
                f"{world}\n{scheduled[0]}/{scheduled[1]} vs {never[0]}/{never[1]}\n"
                f"pooled {TwoWayLedgeReport.pooled_penalty(arms=arms, world=world):+.1f}pp\n"
                f"{sensitivity}"
            )
        slope_ax.set_xticks(positions)
        slope_ax.set_xticklabels(labels, fontsize=8)
        slope_ax.set_xlim(-0.5, len(_WORLDS) - 0.5)
        slope_ax.set_ylabel(
            f"Reset-free penalty: scheduled minus never (tasks, / {_NUM_TEST_TASKS})"
        )
        slope_ax.grid(True, axis="y", alpha=0.3)
        slope_ax.set_title(
            f"The reset-free penalty, per seed ({len(seeds)} seeds)\n"
            "one line per seed; above zero means removing the reset COST tasks",
            fontsize=10,
        )

        differences = TwoWayLedgeReport.interaction_differences(arms=arms)
        flip = PairedTests.sign_flip(differences=differences)
        worse, tied, better = TwoWayLedgeReport.direction_counts(penalties=differences)
        interaction_ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.4, zorder=1)
        for index, (seed, difference) in enumerate(zip(seeds, differences, strict=True)):
            interaction_ax.scatter(
                [index],
                [difference],
                s=62,
                color=_WORLD_COLOR["one-way"] if difference > 0 else "#666666",
                marker=_WORLD_MARKER["one-way"] if difference > 0 else "X",
                zorder=3,
            )
            interaction_ax.annotate(
                f"s{seed}",
                xy=(index, difference),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#444444",
            )
        mean = statistics.mean(differences)
        interaction_ax.axhline(
            mean,
            color=_WORLD_COLOR["one-way"],
            linewidth=2.0,
            label=f"mean {mean:+.1f} tasks",
        )
        interaction_ax.set_xticks([])
        interaction_ax.set_xlim(-0.7, len(seeds) - 0.3)
        interaction_ax.set_ylim(min(differences) - 1.7, max(differences) + 1.2)
        interaction_ax.set_ylabel(
            f"penalty(one-way) minus penalty(two-way) (tasks, / {_NUM_TEST_TASKS})"
        )
        interaction_ax.grid(True, axis="y", alpha=0.3)
        interaction_ax.legend(loc="lower right", fontsize=8)
        interaction_ax.set_title(
            "The interaction, per seed\n"
            f"larger one-way penalty in {worse}/{len(seeds)}, tied in {tied}/{len(seeds)}, "
            f"larger two-way penalty in {better}/{len(seeds)}\n"
            f"exact two-sided sign-flip p = {flip.p_value:.4f}",
            fontsize=10,
        )

        fig.suptitle(
            "Tossing Room (split throws), EES: does opening the one-way ledge remove the "
            "reset-free penalty?\n"
            f"{len(seeds)} paired seeds; the penalty is a WITHIN-world difference, so the "
            "two worlds' unequal difficulty cancels out of it",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)

    @staticmethod
    def render_stranding_figure(*, arms: dict, output: Path) -> None:
        """The mechanism figure: every seed's per-cycle `Pickup*`/`Throw*` attempt count,
        one heatmap panel per arm.

        A heatmap rather than a curve because the claim is about a RUN of zeros in one
        seed's row -- a seed that strands at cycle 3 and never recovers -- and a pooled
        curve averages exactly that away. Every stranded cell is also outlined, so the
        pattern is legible without reading the colour scale, and the count is printed in
        every cell so the figure never asks the reader to estimate a number from a colour.
        """
        seeds = TwoWayLedgeReport.seeds(arms=arms)
        cycles = TwoWayLedgeReport.num_cycles(arms=arms)
        arm_order = TwoWayLedgeReport.arms()
        grids = {arm: TwoWayLedgeReport.item_attempts(arms=arms, arm=arm) for arm in arm_order}
        # One shared colour scale across the four panels, or a panel with little practice
        # would render as brightly as one with a lot.
        vmax = max(max(max(row) for row in grid) for grid in grids.values()) or 1
        fig, axes = plt.subplots(1, len(arm_order), figsize=(20.0, 5.6))

        for ax, arm in zip(axes, arm_order, strict=True):
            world, policy = arm.rsplit("-", maxsplit=1)
            grid = grids[arm]
            stranded = TwoWayLedgeReport.stranded_cycles(arms=arms, arm=arm)
            ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            for row_index, row in enumerate(grid):
                for cycle, value in enumerate(row):
                    ax.annotate(
                        str(value),
                        xy=(cycle, row_index),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < vmax * 0.6 else "black",
                    )
                    if stranded[row_index][cycle]:
                        ax.add_patch(
                            Rectangle(
                                (cycle - 0.5, row_index - 0.5),
                                1.0,
                                1.0,
                                fill=False,
                                edgecolor=_STRANDED_EDGE,
                                linewidth=1.8,
                            )
                        )
            ax.set_xticks(range(cycles))
            ax.set_xticklabels([f"c{cycle}" for cycle in range(cycles)], fontsize=7)
            ax.set_yticks(range(len(seeds)))
            ax.set_yticklabels([f"s{seed}" for seed in seeds], fontsize=8)
            ax.set_xlabel("Interaction period (cycle)")
            strandings = TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=arm)
            stranding_seeds = TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=arm)
            ax.set_title(
                f"{world}, {policy}\n"
                f"stranded cycles {strandings}/{cycles * len(seeds)}, "
                f"seeds that ever strand {len(stranding_seeds)}/{len(seeds)}",
                color=_WORLD_COLOR[world],
                fontsize=10,
            )
        axes[0].set_ylabel("Seed")
        fig.suptitle(
            "Tossing Room (split throws), EES: Pickup*/Throw* attempts per seed per "
            "practice period\n"
            "a cell is STRANDED (outlined) when the period recorded no Pickup and no "
            "Throw attempt at all -- the robot could not reach the pile\n"
            "shared colour scale across panels; the trailing no-practice window is not a "
            "cycle and is excluded",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150, facecolor=_FIGURE_FACECOLOR)
        plt.close(fig)


class ArmDirectories(BaseModel):
    """`--arm NAME=DIR` parsing, kept apart from the report so a malformed flag fails
    before any sweep directory is read."""

    @staticmethod
    def parse(*, entries: list[str], flag: str) -> dict[str, Path]:
        """`["one-way-never=DIR", ...]` -> `{"one-way-never": Path("DIR")}`."""
        parsed = {}
        for entry in entries:
            name, separator, path = entry.partition("=")
            if not separator:
                raise ValueError(f"{flag} must look like one-way-never=PATH, got {entry!r}")
            parsed[name] = Path(path)
        return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help='Repeatable, "one-way-never=/path/to/sweep/root". Aggregation mode.',
    )
    parser.add_argument("--aggregate-output", type=Path, default=None)
    parser.add_argument("--arms-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="The 2x2 outcome figure.")
    parser.add_argument(
        "--penalty-output", type=Path, default=None, help="The reset-free-penalty figure."
    )
    parser.add_argument(
        "--stranding-output", type=Path, default=None, help="The per-cycle stranding heatmap."
    )
    return parser.parse_args()


def _print_manipulation_checks(*, arms: dict) -> None:
    """Everything that decides whether the numbers below are worth reading, printed first
    and on purpose -- and RAISING rather than warning when it fails.

    A reset count that did not move means the flag did nothing and the whole comparison
    is a null by construction. A wrong family denominator means the goal classification
    mis-bucketed EMPTY into a throw family, which produces plausible-looking rates rather
    than an error. A practice record of the wrong shape means every cycle index in the
    stranding timeline is off by one.
    """
    seeds = TwoWayLedgeReport.seeds(arms=arms)
    print(f"Shared seeds ({len(seeds)}): {', '.join(seeds)}")
    print(
        f"Arms ({len(TwoWayLedgeReport.arms())}, a complete 2x2): "
        f"{', '.join(TwoWayLedgeReport.arms())}\n"
    )

    print("MANIPULATION CHECK -- free resets that actually happened (Metrics.num_practice_resets):")
    print(f"{'arm':>18} {'expected':>9} {'observed (min..max)':>21} {'matched':>9}")
    for arm in TwoWayLedgeReport.arms():
        expected = _EXPECTED_RESETS[arm.rsplit("-", maxsplit=1)[1]]
        counts = TwoWayLedgeReport.reset_counts(arms=arms, arm=arm)
        matched = sum(1 for count in counts if count == expected)
        print(
            f"{arm:>18} {expected:>9} {f'{min(counts)}..{max(counts)}':>21} "
            f"{f'{matched}/{len(counts)}':>9}"
        )
    reset_problems = TwoWayLedgeReport.reset_violations(arms=arms)
    if reset_problems:
        raise ValueError(
            "the reset manipulation did not happen as designed: " + "; ".join(reset_problems[:10])
        )

    print(f"\nEXPERIENCE CHECK -- online transitions taken (design: {_EXPECTED_TRANSITIONS} each):")
    for arm in TwoWayLedgeReport.arms():
        achieved = TwoWayLedgeReport.achieved_transitions(arms=arms, arm=arm)
        matched = sum(1 for value in achieved if value == _EXPECTED_TRANSITIONS)
        print(
            f"  {arm:>18}: min {min(achieved)}, max {max(achieved)}, exactly "
            f"{_EXPECTED_TRANSITIONS} in {matched}/{len(achieved)} runs"
        )
    transition_problems = TwoWayLedgeReport.transition_violations(arms=arms)
    if transition_problems:
        raise ValueError(
            "the four arms did not buy the same experience, so their outcomes are not "
            "comparable: " + "; ".join(transition_problems[:10])
        )

    print("\nCOMPOSITION CHECK -- test tasks per family per seed (designed 14/14/2):")
    for family, expected in expected_denominators().items():
        observed = sorted({
            count
            for arm in TwoWayLedgeReport.arms()
            for count in TwoWayLedgeReport.family_denominators(arms=arms, arm=arm, family=family)
        })
        print(f"  {family:>10}: {observed} (expected {expected})")
    violations = TwoWayLedgeReport.composition_violations(arms=arms)
    total_cells = len(TwoWayLedgeReport.arms()) * len(seeds) * len(_FAMILIES)
    print(f"  violations across all four arms and every seed: {len(violations)}/{total_cells}")
    if violations:
        raise ValueError(
            "the realised test-set composition is not the designed one, so every rate "
            "below would be computed on a wrong denominator: " + "; ".join(violations[:10])
        )

    window_problems = TwoWayLedgeReport.practice_window_violations(arms=arms)
    print(
        f"\nPRACTICE-RECORD CHECK -- {TwoWayLedgeReport.num_sweeps(arms=arms)} evaluation "
        f"sweeps and {TwoWayLedgeReport.num_cycles(arms=arms)} practice periods per run; "
        f"malformed records: {len(window_problems)}/{len(TwoWayLedgeReport.arms()) * len(seeds)}"
    )
    if window_problems:
        raise ValueError(
            "a practice record is not shaped as the cycle indexing assumes, so the "
            "stranding timeline would be shifted: " + "; ".join(window_problems[:10])
        )


def _print_final_success(*, arms: dict) -> None:
    seeds = TwoWayLedgeReport.seeds(arms=arms)
    denominators = expected_denominators()

    print(
        f"\nFINAL SWEEP, per-seed tasks solved out of {_NUM_TEST_TASKS} "
        "(the paired unit every test below runs on):"
    )
    print(f"{'arm':>18} " + " ".join(f"{('s' + seed):>4}" for seed in seeds) + f" {'mean':>7}")
    for arm in TwoWayLedgeReport.arms():
        solved = TwoWayLedgeReport.final_solved(arms=arms, arm=arm)
        print(
            f"{arm:>18} "
            + " ".join(f"{value:>4}" for value in solved)
            + f" {statistics.mean(solved):>7.1f}"
        )

    print(f"\nFINAL SWEEP, pooled over the {len(seeds)} shared seeds (solved / attempted):")
    header = " ".join(f"{arm:>17}" for arm in TwoWayLedgeReport.arms())
    print(f"{'family':>10} {'per seed':>9} {header}")
    for family in (None, *_FAMILIES):
        label = "OVERALL" if family is None else family
        per_seed = _NUM_TEST_TASKS if family is None else denominators[family]
        cells = []
        for arm in TwoWayLedgeReport.arms():
            counts = TwoWayLedgeReport.pooled_counts(arms=arms, arm=arm, family=family)
            cells.append(f"{counts[0]}/{counts[1]} ({TwoWayLedgeReport.rate(counts=counts):.0f}%)")
        print(f"{label:>10} {per_seed:>9} " + " ".join(f"{cell:>17}" for cell in cells))
    print(
        "  The two worlds are NOT equally difficult -- --two-way-ledge also drops the "
        "EMPTY solve\n  from 10 steps to 9 (horizon 12 -> 11) and stops RECYCLING being "
        "one-attempt-per-period.\n  So a one-way count is never subtracted from a two-way "
        "one anywhere below; only the\n  within-world gaps are compared across the two."
    )


def _print_penalty(*, arms: dict) -> None:
    """The reset-free penalty per world: the per-seed vector first, the pooled number
    second, and the sensitivity the pooled number has to be read against."""
    seeds = TwoWayLedgeReport.seeds(arms=arms)
    print(
        f"\nRESET-FREE PENALTY -- scheduled minus never, final tasks solved out of "
        f"{_NUM_TEST_TASKS}\n  (positive = removing the free reset COST the agent tasks):"
    )
    print(f"{'world':>10} " + " ".join(f"{('s' + seed):>4}" for seed in seeds) + f" {'mean':>7}")
    for world in _WORLDS:
        values = TwoWayLedgeReport.penalties(arms=arms, world=world)
        print(
            f"{world:>10} "
            + " ".join(f"{int(value):>4}" for value in values)
            + f" {statistics.mean(values):>7.1f}"
        )
    differences = TwoWayLedgeReport.interaction_differences(arms=arms)
    print(
        f"{'INTERACT':>10} "
        + " ".join(f"{int(value):>4}" for value in differences)
        + f" {statistics.mean(differences):>7.1f}   (one-way penalty minus two-way penalty)"
    )

    print("\nPOOLED per world, and the sensitivity each pooled number has to be read against:")
    print(
        f"{'world':>10} {'scheduled':>12} {'never':>12} {'penalty':>10} {'MDE':>9} {'verdict':>30}"
    )
    for world in _WORLDS:
        scheduled_arm, never_arm = TwoWayLedgeReport.world_arms(world=world)
        scheduled = TwoWayLedgeReport.pooled_counts(arms=arms, arm=scheduled_arm)
        never = TwoWayLedgeReport.pooled_counts(arms=arms, arm=never_arm)
        penalty = TwoWayLedgeReport.pooled_penalty(arms=arms, world=world)
        mde = TwoProportionSensitivity.minimum_detectable_effect(counts_a=scheduled, counts_b=never)
        if mde is None:
            # Never "0.0pp": a zero here is the normal approximation collapsing, not a
            # design that can detect anything. Say so in words instead.
            print(
                f"{world:>10} {f'{scheduled[0]}/{scheduled[1]}':>12} "
                f"{f'{never[0]}/{never[1]}':>12} {penalty:>+9.1f}pp {'n/a':>9} "
                f"{'degenerate; supports no inference':>30}"
            )
            continue
        verdict = "above MDE" if abs(penalty) >= mde else "BELOW MDE -- not resolvable"
        print(
            f"{world:>10} {f'{scheduled[0]}/{scheduled[1]}':>12} "
            f"{f'{never[0]}/{never[1]}':>12} {penalty:>+9.1f}pp {mde:>8.1f}pp {verdict:>30}"
        )
    print(
        f"  constant (z_0.025 + z_0.20) = {_MDE_CONSTANT:.6f}; the MDE is evaluated at the "
        "observed rates,\n  from each comparison's OWN two denominators, and is `None` "
        "rather than 0.0pp whenever both\n  arms sit at an extreme -- there the variance "
        "terms are exactly zero, which is the\n  approximation breaking down and not a "
        "design that resolves arbitrarily small effects."
    )

    print(f"\nPER-FAMILY pooled penalty (percentage points; counts are {len(seeds)} seeds pooled):")
    print(f"{'family':>10} " + " ".join(f"{world:>28}" for world in _WORLDS))
    for family in _FAMILIES:
        cells = []
        for world in _WORLDS:
            scheduled_arm, never_arm = TwoWayLedgeReport.world_arms(world=world)
            scheduled = TwoWayLedgeReport.pooled_counts(arms=arms, arm=scheduled_arm, family=family)
            never = TwoWayLedgeReport.pooled_counts(arms=arms, arm=never_arm, family=family)
            penalty = TwoWayLedgeReport.pooled_penalty(arms=arms, world=world, family=family)
            mde = TwoProportionSensitivity.minimum_detectable_effect(
                counts_a=scheduled, counts_b=never
            )
            note = "degenerate" if mde is None else f"MDE {mde:.0f}pp"
            cells.append(
                f"{scheduled[0]}/{scheduled[1]} vs {never[0]}/{never[1]} {penalty:+.0f}pp ({note})"
            )
        print(f"{family:>10} " + " ".join(f"{cell:>28}" for cell in cells))


def _print_paired_test(*, label: str, differences: list[float], seeds: list[str]) -> None:
    """One exact paired test, printed with its p as a fraction of the enumerated null.

    Both tests are run on every comparison: `sign_flip` is the paired t-test's
    assumption-free twin (it tests the mean difference) and `wilcoxon_signed_rank` is its
    distribution-free companion (it tests the ranks). Agreeing is not guaranteed, and
    where they disagree that is itself the finding.
    """
    worse, tied, better = TwoWayLedgeReport.direction_counts(penalties=differences)
    flip = PairedTests.sign_flip(differences=differences)
    wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
    nonzero = [value for value in differences if value != 0.0]
    # Dropping the ties changes the fraction's denominator (2**8 rather than 2**10) but
    # not the p-value: a tied seed contributes the same factor to the numerator and the
    # denominator. Reported in the compact form, and cross-checked against the full
    # enumeration so the claim "they agree" is measured rather than asserted.
    compact = PairedTests.sign_flip(differences=nonzero)
    if abs(compact.p_value - flip.p_value) > 1e-12:
        raise ValueError(
            f"dropping tied seeds changed the exact p-value ({compact.p_value} vs "
            f"{flip.p_value}); one of the two enumerations is wrong"
        )
    print(f"\n  {label}")
    print(f"    per-seed differences (tasks): {[int(value) for value in differences]}")
    print(
        f"    positive in {worse}/{len(seeds)} seeds, tied in {tied}/{len(seeds)}, "
        f"negative in {better}/{len(seeds)}"
    )
    spread = f" (sd {statistics.stdev(differences):.1f})" if len(differences) > 1 else ""
    print(f"    mean {statistics.mean(differences):+.1f} tasks{spread} out of {_NUM_TEST_TASKS}")
    print(
        f"    exact two-sided sign-flip p = {round(compact.p_value * 2 ** len(nonzero))}/"
        f"{2 ** len(nonzero)} = {compact.p_value:.4f}   "
        f"(all {len(differences)} seeds including ties: "
        f"{round(flip.p_value * 2 ** len(differences))}/{2 ** len(differences)}, the same "
        "value)"
    )
    print(
        f"    exact Wilcoxon signed-rank p = {wilcoxon.p_value:.4f} "
        f"({wilcoxon.num_zero_differences}/{len(differences)} tied seeds dropped)"
    )


def _print_paired_tests(*, arms: dict) -> None:
    """The three exact paired tests, in the order the argument runs.

    All four arms ran the same fixed seeds, so every one of these is paired data; an
    unpaired test would throw that structure away and understate the significance.
    """
    seeds = TwoWayLedgeReport.seeds(arms=arms)
    print(f"\nEXACT PAIRED TESTS over the {len(seeds)} shared seeds")
    print("  Sign convention throughout: POSITIVE means the reset-free arm solved fewer tasks.")
    _print_paired_test(
        label=(
            "TWO-WAY WORLD, scheduled vs never -- does the penalty survive when the "
            "world has\n  no irreversible action left?"
        ),
        differences=TwoWayLedgeReport.penalties(arms=arms, world="two-way"),
        seeds=seeds,
    )
    _print_paired_test(
        label=(
            "ONE-WAY WORLD, scheduled vs never -- the reproduction check against PR "
            "#115, on\n  fresh runs of the same configuration"
        ),
        differences=TwoWayLedgeReport.penalties(arms=arms, world="one-way"),
        seeds=seeds,
    )
    _print_paired_test(
        label=(
            "THE INTERACTION, penalty(one-way) minus penalty(two-way) -- the headline: "
            "is the\n  reset-free penalty CAUSED by the irreversible action?"
        ),
        differences=TwoWayLedgeReport.interaction_differences(arms=arms),
        seeds=seeds,
    )


def _print_stranding(*, arms: dict) -> None:
    """The mechanism, measured from the per-cycle practice tallies rather than inferred
    from the outcome numbers."""
    seeds = TwoWayLedgeReport.seeds(arms=arms)
    cycles = TwoWayLedgeReport.num_cycles(arms=arms)
    total = cycles * len(seeds)
    print(
        f"\nSTRANDING -- a practice period with NO Pickup* and NO Throw* attempt at all, "
        f"i.e. the\nrobot could not reach the pile. Measured from "
        f"Metrics.practice_outcomes_per_cycle over\n{len(seeds)} seeds x {cycles} periods:"
    )
    print(f"{'arm':>18} {'stranded cycles':>17} {'seeds that strand':>19} {'item attempts':>15}")
    for arm in TwoWayLedgeReport.arms():
        stranded = TwoWayLedgeReport.num_stranded_cycles(arms=arms, arm=arm)
        stranding_seeds = TwoWayLedgeReport.seeds_that_strand(arms=arms, arm=arm)
        attempts = sum(sum(row) for row in TwoWayLedgeReport.item_attempts(arms=arms, arm=arm))
        print(
            f"{arm:>18} {f'{stranded}/{total}':>17} "
            f"{f'{len(stranding_seeds)}/{len(seeds)}':>19} {attempts:>15}"
        )

    print(f"\nFIRST STRANDED CYCLE per seed ('-' = never strands, out of {cycles} periods):")
    print(f"{'arm':>18} " + " ".join(f"{('s' + seed):>4}" for seed in seeds))
    for arm in TwoWayLedgeReport.arms():
        firsts = TwoWayLedgeReport.first_stranded_cycle(arms=arms, arm=arm)
        print(
            f"{arm:>18} "
            + " ".join(f"{('-' if first is None else f'c{first}'):>4}" for first in firsts)
        )

    print(f"\nPickup*/Throw* ATTEMPTS per cycle, pooled over the {len(seeds)} seeds:")
    print(f"{'arm':>18} " + " ".join(f"{('c' + str(cycle)):>5}" for cycle in range(cycles)))
    for arm in TwoWayLedgeReport.arms():
        rows = TwoWayLedgeReport.item_attempts(arms=arms, arm=arm)
        per_cycle = [sum(row[cycle] for row in rows) for cycle in range(cycles)]
        print(f"{arm:>18} " + " ".join(f"{value:>5}" for value in per_cycle))
    print(
        "  A pooled row cannot distinguish 'every seed still practises a little' from "
        "'one seed\n  practises and the rest are stranded'; the per-seed heatmap "
        "(--stranding-output) and the\n  first-stranded-cycle table above are what "
        "separate them."
    )


def _print_report(*, arms: dict) -> None:
    _print_manipulation_checks(arms=arms)
    _print_final_success(arms=arms)
    _print_penalty(arms=arms)
    _print_paired_tests(arms=arms)
    _print_stranding(arms=arms)


def main() -> None:
    args = _parse_args()
    if args.arm:
        if args.aggregate_output is None:
            raise ValueError("--arm requires --aggregate-output")
        aggregate = TwoWayLedgeReport.aggregate(
            arm_dirs=ArmDirectories.parse(entries=args.arm, flag="--arm")
        )
        # Compact, one line: this is recorded data, not source, and an indented rendering
        # is many times the bytes for no added readability.
        args.aggregate_output.write_text(json.dumps(aggregate, sort_keys=True))
        print(f"wrote {args.aggregate_output}")
        return

    if args.arms_json is None:
        raise ValueError("pass one of: --arm ... --aggregate-output, or --arms-json")
    arms = TwoWayLedgeReport.load_arms(json_path=args.arms_json)
    for output, render in (
        (args.output, TwoWayLedgeReport.render_outcome_figure),
        (args.penalty_output, TwoWayLedgeReport.render_penalty_figure),
        (args.stranding_output, TwoWayLedgeReport.render_stranding_figure),
    ):
        if output is not None:
            render(arms=arms, output=output)
            print(f"wrote {output}")
    _print_report(arms=arms)


if __name__ == "__main__":
    main()
