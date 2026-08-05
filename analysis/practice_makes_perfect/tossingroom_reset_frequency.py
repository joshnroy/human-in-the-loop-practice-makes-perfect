"""Post-run analysis for the Tossing Room reset-frequency experiment: does the
penalty a domain charges for an *irreversible* action depend on how often the
harness hands the robot a free reset?

`PracticeLoop.run` calls `problem.reset_to_task` once per practice cycle, so the
robot gets a free environment reset every `--max-steps-per-interaction` steps.
Tossing Room has exactly one genuinely terminal failure -- a missed `Throw` on the
RECYCLING family strands the robot behind the one-way ledge with the item gone, so
Fast Downward correctly reports no plan for the rest of the period. A missed TRASH
throw is merely expensive (a round trip buys a fresh item), and the EMPTY family
has no `Throw` at all. The hypothesis was that longer practice periods (fewer free
resets) waste more experience per stranding, so RECYCLING should fall further
behind TRASH as the period lengthens.

The four arms hold total online transitions fixed at 2500 and trade cycles against
steps (50x50, 25x100, 10x250, 5x500). The designed metric was the *within-arm*
(TRASH - RECYCLING) final gap, paired by seed, on the reasoning that
`--num-cycles` also sets how many times the sampler refits -- so absolute rates
are not comparable across arms -- while both families inside one arm see the same
refits, so a within-arm difference cancels it.

**That reasoning is only half right, and the measured data is why.** The gap
cancels the refit confound's effect on the *level*, but the gap is itself a
hump-shaped function of training progress: near zero when both families sit at the
floor, largest mid-training when TRASH pulls ahead, near zero again once both
saturate. `--num-cycles` sets progress, so the arms can sit at different points on
that hump and their final gaps can differ for that reason alone.

They do. `family_differences` measures the precondition directly, and it fails:
between the extreme arms (500 steps vs 50) final competence differs by **-40.7pp
on TRASH (p = 0.0195) and -82.9pp on RECYCLING (p = 0.0020)**, exact paired tests.
The arms are not equally trained, so a cross-arm gap difference is not attributable
to reset frequency. `progress_diagnostics` reports the supporting quantities (final
TRASH level, seeds at the ceiling, seeds that never improved, exact ties), and both
are printed *before* any p-value, because they decide whether the p-values below
mean anything. `progress_matched_gaps` is the un-confounded companion: the gap read
at a common TRASH competence rather than at a common transition count.

**That companion is now load-bearing, and `progress_matched_differences` is why it
is a test and not a table.** On the fixed 14/14/2 evaluation set (PR #41) the final
gap *does* trend with period length -- monotone, armD - armA = +42.14pp at exact
Wilcoxon p = 0.0039 -- where on the superseded sampled composition it did not trend
at all. Hold training progress fixed and the same contrast is -5.00pp at p = 0.6875,
a null result. So the trend is what the hump predicts from the competence difference
above, and this design still cannot tell that apart from a reset effect. Anything
here that quotes a number quotes the fixed-composition run; the 2026-08-03 numbers
it replaced are recorded in the experiment log's *previously* columns.

EMPTY is the control: no `Throw`, no stochastic skill, solved by a deterministic
`MoveRoom`-then-`Press` plan. If its rate moves across arms, something other than
irreversibility is in play, so the summary table prints it alongside. Under the
fixed composition it is only **2 tasks per seed**, so its 100% is 2/2 -- a
plan-and-execute smoke test rather than a control with the power to detect a
moderate regression, and it is read that way.

**The manipulation is checked, not assumed.** `composition_violations` asserts the
realised per-family denominators against `TossingRoomTasks.test_goal_type_counts()`
and `transition_violations` asserts the 2500-transition budget, both per arm, seed
and evaluation sweep. This experiment's own finding is a design that failed to
isolate what it claimed; taking the evaluation set on trust would repeat that one
level down, and is exactly what went stale when PR #41 landed.

Reads only already-produced outputs (CLAUDE.md's analysis/ convention -- never
runs a simulation). Two modes, both post-run:

* `--arm NAME=DIR ... --aggregate-output JSON` reads each sweep's
  `DIR/<method>/<seed>/stats.json` and condenses the per-task outcomes into a
  committed per-family aggregate. Raw sweep directories live outside the repo and
  do not travel between machines (see
  `docs/experiment-logs/2026-08-03-cross-machine-reproducibility.md`), so the
  aggregate is the record that survives.
* `--arms-json JSON --output PNG --curves-output PNG` regenerates every figure,
  table and p-value in the experiment log from that aggregate alone.

**Statistics.** Every arm ran the same fixed seeds 0..9, so every comparison here
is *paired*. scipy is not a dependency of this project, so rather than quoting
p-values computed elsewhere, `PairedTests` computes them **exactly**, by
enumeration: at n = 10 there are only 2**10 = 1024 sign assignments, so both the
signed-rank null and the mean-difference null are enumerated in full rather than
approximated. No normal approximation, no special functions, no tie correction
needed -- ties are handled by averaging ranks and enumerating against *those*
ranks, which is exact by construction. Ties matter here: with 14 tasks per throw
family per seed a gap can only land on multiples of 100/14 = 7.1pp, so exact-zero
differences are common and are reported rather than silently dropped.
"""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import EvaluationBreakdown
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Goal.describe() rendering -> task family. Explicit rather than pattern-matched:
# an unrecognised goal is a bug (a domain change, or the wrong sweep directory),
# and silently bucketing it as "other" would quietly shrink a denominator.
_FAMILY_BY_GOAL = {
    "ItemInBin(recycling, recycling_bin)": "RECYCLING",
    "ItemInBin(trash, trash_bin)": "TRASH",
    "BinEmpty(recycling_bin) & BinEmpty(trash_bin)": "EMPTY",
}

# Arm -> --max-steps-per-interaction, i.e. the practice-period length in
# environment steps, which is also the inverse of the free-reset frequency. The
# ordering of this dict is the ordering of every table and figure below.
_ARM_PERIOD = {"armA": 50, "armB": 100, "armC": 250, "armD": 500}

# Arm -> --num-cycles. Chosen so period * cycles = 2500 online transitions in
# every arm, and carried here only to label the confound: this is also the number
# of sampler refits and the number of evaluation sweeps.
_ARM_CYCLES = {"armA": 50, "armB": 25, "armC": 10, "armD": 5}

# armB is the shipped Tossing Room configuration; the figures mark it.
_SHIPPED_ARM = "armB"

# Okabe-Ito, verified colourblind-safe for this exact trio (worst adjacent pair
# deltaE 11.0 deutan / 25.8 normal vision, all three inside the lightness band).
# Each family also gets its own marker, so identity is never colour-alone.
_FAMILY_STYLE = {
    "RECYCLING": ("#0072B2", "o"),
    "TRASH": ("#D55E00", "s"),
    "EMPTY": ("#009E73", "^"),
}

# z_{0.975} and z_{0.80}, for the "how many seeds would 80% power need?" line that
# every non-significant result on this project is required to carry.
_Z_ALPHA = 1.959964
_Z_POWER = 0.841621


class ResetFrequencyReport:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def aggregate(*, arm_dirs: dict[str, Path], method: str = "ees") -> dict:
        """Condenses raw sweep directories into the committed aggregate,
        `{arm: {seed: {family: [[transitions, num_solved, num_total], ...]}}}` --
        the same `[transitions, solved, total]` triple `2026-08-03-ballring-arms.json`
        already uses, one level deeper so each task family keeps its own curve.

        Reads each `stats.json` back through `Metrics.model_validate_json` rather
        than parsing the JSON by hand, per analysis/README.md -- the per-task
        detail this needs is `Metrics.breakdowns`, a real field on that model.
        """
        aggregate: dict = {}
        for arm, root in sorted(arm_dirs.items(), key=lambda item: _ARM_PERIOD[item[0]]):
            seeds: dict = {}
            for stats_path in sorted((root / method).glob("*/stats.json")):
                metrics = Metrics.model_validate_json(stats_path.read_text())
                if not metrics.breakdowns:
                    raise ValueError(
                        f"{stats_path} has no per-task breakdowns -- it predates "
                        f"Metrics.breakdowns, so per-family numbers cannot be recovered"
                    )
                curves: dict[str, list[list[int]]] = {family: [] for family in _FAMILY_STYLE}
                for breakdown in metrics.breakdowns:
                    for family, (solved, total) in ResetFrequencyReport._counts(
                        breakdown=breakdown
                    ).items():
                        curves[family].append([breakdown.num_online_transitions, solved, total])
                seeds[stats_path.parent.name] = curves
            if not seeds:
                raise ValueError(f"no stats.json under {root / method}")
            aggregate[arm] = seeds
        return aggregate

    @staticmethod
    def _counts(*, breakdown: EvaluationBreakdown) -> dict[str, tuple[int, int]]:
        """family -> (num_solved, num_total) for one evaluation sweep."""
        counts: dict[str, list[int]] = {family: [0, 0] for family in _FAMILY_STYLE}
        for outcome in breakdown.outcomes:
            family = _FAMILY_BY_GOAL.get(outcome.goal)
            if family is None:
                raise ValueError(f"unrecognised goal description: {outcome.goal!r}")
            counts[family][0] += int(outcome.solved)
            counts[family][1] += 1
        return {family: (solved, total) for family, (solved, total) in counts.items()}

    @staticmethod
    def load_arms(*, json_path: Path) -> dict:
        arms = json.loads(json_path.read_text())
        missing = sorted(set(_ARM_PERIOD) - set(arms))
        if missing:
            raise ValueError(f"aggregate JSON is missing arms: {missing}")
        return arms

    @staticmethod
    def final_rates(*, arms: dict, arm: str, family: str) -> list[float]:
        """Per-seed % solved for one family at the arm's LAST evaluation sweep,
        ordered by seed so the lists across arms are index-aligned for pairing."""
        rates = []
        for seed in ResetFrequencyReport.seeds(arms=arms):
            _transitions, solved, total = max(arms[arm][seed][family], key=lambda triple: triple[0])
            rates.append(100.0 * solved / total if total else 0.0)
        return rates

    @staticmethod
    def seeds(*, arms: dict) -> list[str]:
        """The seeds every arm shares, sorted numerically. Pairing is only valid
        over these, so the intersection is taken rather than assumed."""
        shared: set[str] | None = None
        for seeds in arms.values():
            shared = set(seeds) if shared is None else shared & set(seeds)
        return sorted(shared or set(), key=int)

    @staticmethod
    def gaps(*, arms: dict, arm: str) -> list[float]:
        """Per-seed (TRASH - RECYCLING) final-sweep gap, in percentage points --
        the metric this experiment was designed around.

        It was chosen because it is *within*-arm and so cancels the sampler-refit
        confound that makes absolute rates incomparable across arms. That
        reasoning is only half right, and the half that fails is why this
        experiment does not answer its question -- see `family_differences`,
        `progress_matched_gaps` and the experiment log. The gap cancels the
        confound's effect on the *level*, but the gap is itself a hump-shaped
        function of training progress: near zero when both families sit at the
        floor, largest mid-training when TRASH pulls ahead, near zero again once
        both saturate. Since `--num-cycles` sets progress, arms can sit at
        different points on that hump -- and measurably do here, differing by
        40.7pp (TRASH) and 82.9pp (RECYCLING) in final competence, which is larger
        than any gap difference this metric reports.

        On the fixed 14/14/2 evaluation set the measured values are **monotone** in
        period length (-5.7, +4.3, +32.1, +36.4 pp for 50/100/250/500), and the
        trend is significant (armD - armA = +42.14pp, exact Wilcoxon p = 0.0039).
        That is *not* support for the hypothesis: the ordering also tracks training
        progress exactly, since the least-trained arm sits mid-hump while the
        most-trained one is pinned at the ceiling with both families at 100%. The
        discriminating test is `progress_matched_differences`, which holds progress
        fixed and returns a null result (-5.00pp, p = 0.6875).

        On the superseded *sampled* composition these values were non-monotone
        (-1.2, +5.4, +35.8, +3.2) and no trend test approached significance. That
        difference is the evaluation set, not the harness -- same machine, same
        seeds, same flags.
        """
        trash = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family="TRASH")
        recycling = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family="RECYCLING")
        return [t - r for t, r in zip(trash, recycling, strict=True)]

    @staticmethod
    def progress_matched_gaps(*, arms: dict, arm: str, level: float) -> list[float | None]:
        """Per-seed (TRASH - RECYCLING) gap at the first checkpoint where that
        seed's TRASH rate reaches `level` -- the gap with training progress held
        fixed, which is what the final-checkpoint gap cannot do.

        None for a seed whose TRASH rate never reaches `level` in this arm; the
        caller decides whether enough seeds survive to be worth reporting.
        `matchable_level` picks the highest level every seed of every arm reaches,
        so the comparison is made where all four arms actually have data.
        """
        results: list[float | None] = []
        for seed in ResetFrequencyReport.seeds(arms=arms):
            trash = arms[arm][seed]["TRASH"]
            recycling = arms[arm][seed]["RECYCLING"]
            gap: float | None = None
            for (_t, trash_solved, trash_total), (_r, rec_solved, rec_total) in zip(
                trash, recycling, strict=True
            ):
                if trash_total and 100.0 * trash_solved / trash_total >= level:
                    gap = 100.0 * trash_solved / trash_total - (
                        100.0 * rec_solved / rec_total if rec_total else 0.0
                    )
                    break
            results.append(gap)
        return results

    @staticmethod
    def progress_matched_differences(
        *, arms: dict, from_arm: str, to_arm: str, level: float
    ) -> list[float]:
        """Per-seed (to_arm - from_arm) difference in the *progress-matched* gap --
        the cross-arm contrast with training progress held fixed, over the seeds
        where both arms reach `level`.

        This is the un-confounded twin of the headline `armD - armA` final-gap
        test, and on the fixed-composition data it is the one that decides the
        experiment's reading: the final gap trends with period length at p < 0.01,
        but the arms also end ~40-80 competence points apart, and the gap is
        hump-shaped in progress. If the trend were a reset effect rather than a
        position-on-the-hump effect, it should still be here once progress is
        matched.
        """
        from_gaps = ResetFrequencyReport.progress_matched_gaps(arms=arms, arm=from_arm, level=level)
        to_gaps = ResetFrequencyReport.progress_matched_gaps(arms=arms, arm=to_arm, level=level)
        return [
            to_gap - from_gap
            for from_gap, to_gap in zip(from_gaps, to_gaps, strict=True)
            if from_gap is not None and to_gap is not None
        ]

    @staticmethod
    def matchable_level(*, arms: dict) -> float:
        """The highest TRASH success level every seed of every arm reaches at some
        checkpoint -- the only level at which a progress-matched comparison has
        data everywhere. Low by construction: it is set by the worst seed of the
        least-trained arm."""
        best = 100.0
        for arm in _ARM_PERIOD:
            for seed in ResetFrequencyReport.seeds(arms=arms):
                peak = max(
                    100.0 * solved / total if total else 0.0
                    for _transitions, solved, total in arms[arm][seed]["TRASH"]
                )
                best = min(best, peak)
        return best

    @staticmethod
    def progress_diagnostics(*, arms: dict, arm: str) -> dict[str, int]:
        """The counts that decide whether a cross-arm gap comparison means
        anything: how many seeds saturated both families (a gap of zero forced by
        the ceiling), how many never improved on their pre-practice rate, and how
        many produced an exact-zero gap (which is the *effective* n of every
        paired test below, since ties are dropped)."""
        seeds = ResetFrequencyReport.seeds(arms=arms)
        trash = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family="TRASH")
        recycling = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family="RECYCLING")
        untrained = [
            100.0 * arms[arm][seed]["TRASH"][0][1] / arms[arm][seed]["TRASH"][0][2]
            for seed in seeds
        ]
        return {
            "at_ceiling_both": sum(
                1 for t, r in zip(trash, recycling, strict=True) if t == 100.0 and r == 100.0
            ),
            "trash_no_better_than_untrained": sum(
                1 for t, u in zip(trash, untrained, strict=True) if t <= u
            ),
            "zero_gap": sum(1 for t, r in zip(trash, recycling, strict=True) if t == r),
        }

    @staticmethod
    def family_differences(*, arms: dict, family: str, from_arm: str, to_arm: str) -> list[float]:
        """Per-seed (to_arm - from_arm) difference in one family's final success
        rate -- paired, since every arm ran the same fixed seeds.

        This is the experiment's **discriminating check**, and it has to be read
        before the gap comparison. The gap is hump-shaped in training progress, so
        two arms only have comparable gaps if they end at comparable competence.
        Run it on TRASH (are the arms equally trained?) and on EMPTY (the control:
        no Throw, no stochastic skill, so any movement here is not irreversibility).
        """
        return [
            t - f
            for f, t in zip(
                ResetFrequencyReport.final_rates(arms=arms, arm=from_arm, family=family),
                ResetFrequencyReport.final_rates(arms=arms, arm=to_arm, family=family),
                strict=True,
            )
        ]

    @staticmethod
    def pooled_rate(*, arms: dict, arm: str, family: str) -> float:
        """One arm's final-sweep success rate for a family, pooled over all seeds
        (total solved / total tasks) rather than averaged over per-seed rates.

        Descriptive only -- every paired test stays on per-seed values, since
        pooling destroys the pairing. It is reported because the per-seed mean is
        an average of ~10 proportions whose denominators are as small as 5, so it
        carries far more sampling noise than the ~100-task pooled figure while
        estimating the same quantity.
        """
        solved, total = ResetFrequencyReport.pooled_counts(arms=arms, arm=arm, family=family)
        return 100.0 * solved / total if total else 0.0

    @staticmethod
    def pooled_counts(*, arms: dict, arm: str, family: str) -> tuple[int, int]:
        """(evaluation episodes solved, evaluation episodes run) at the final sweep,
        summed over seeds -- the primary record behind every pooled rate quoted for
        this experiment.

        Reported as the count rather than only as a percentage because a rate has
        no denominator attached and this project has already published one number
        whose denominator silently changed underneath it. Both integers come from
        `Metrics.breakdowns`; neither is reconstructed by multiplying a percentage.
        """
        solved = total = 0
        for seed in ResetFrequencyReport.seeds(arms=arms):
            _transitions, seed_solved, seed_total = max(
                arms[arm][seed][family], key=lambda triple: triple[0]
            )
            solved += seed_solved
            total += seed_total
        return solved, total

    @staticmethod
    def family_denominators(*, arms: dict, arm: str, family: str) -> list[int]:
        """How many test tasks of one family each seed actually drew. Reported
        because it is the experiment's real limiting quantity: it is the
        denominator every per-family rate is a fraction of, and it sets the noise
        floor `predicted_gap_noise` computes.

        Under the **fixed** composition (PR #41) this is a constant 14/14/2 at 30
        test tasks, which `composition_violations` asserts rather than assumes.
        The original 2026-08-03 run predates that change: `goal_weights` was
        *sampled* then, so 30 test tasks split unevenly and a seed could hold as
        few as five RECYCLING tasks -- which made "40% -> 0%" mean 2/5 -> 0/5.
        """
        return [arms[arm][seed][family][0][2] for seed in ResetFrequencyReport.seeds(arms=arms)]

    @staticmethod
    def expected_composition(*, num_test_tasks: int = 30) -> dict[str, int]:
        """The per-family test-set composition the *code* produces, read from the
        domain itself rather than restated here.

        `TossingRoomTasks.test_goal_type_counts` is public precisely so an analysis
        script can assert the composition of a run it is reading back, and it
        depends on neither the layout nor the seed -- so a default-constructed
        instance is enough. At 30 test tasks: 14 TRASH / 14 RECYCLING / 2 EMPTY.
        """
        counts = TossingRoomTasks(
            env=TossingRoomEnvironment(), num_test_tasks=num_test_tasks
        ).test_goal_type_counts()
        return {goal_type.name: count for goal_type, count in counts.items()}

    @staticmethod
    def composition_violations(*, arms: dict, num_test_tasks: int = 30) -> list[str]:
        """Every (arm, seed, sweep) whose realised per-family denominators disagree
        with `expected_composition`. Empty means the manipulation held everywhere.

        This is the check this experiment's own headline demands. PR #39's lesson
        was a design that did not isolate what it claimed; assuming the evaluation
        composition instead of measuring it would be the same mistake one level
        down, and it is exactly the assumption that went stale when PR #41 replaced
        the sampled composition with a fixed one.
        """
        expected = ResetFrequencyReport.expected_composition(num_test_tasks=num_test_tasks)
        violations = []
        for arm in sorted(arms):
            for seed in ResetFrequencyReport.seeds(arms=arms):
                num_sweeps = min(len(arms[arm][seed][family]) for family in expected)
                for index in range(num_sweeps):
                    realised = {family: arms[arm][seed][family][index][2] for family in expected}
                    if realised != expected:
                        violations.append(f"{arm}/seed {seed}/sweep {index}: {realised}")
        return violations

    @staticmethod
    def transition_violations(*, arms: dict, expected: int = 2500) -> list[str]:
        """Every (arm, seed) that did not reach exactly `expected` online
        transitions. The arms are only comparable at all because the experience
        budget is identical by construction, so a shortfall is a residual confound
        and is measured rather than argued from the loop's arithmetic."""
        return [
            f"{arm}/seed {seed}: {achieved:.0f}"
            for arm in sorted(arms)
            for seed, achieved in zip(
                ResetFrequencyReport.seeds(arms=arms),
                ResetFrequencyReport.achieved_transitions(arms=arms, arm=arm),
                strict=True,
            )
            if achieved != expected
        ]

    @staticmethod
    def predicted_gap_noise(*, arms: dict, arm: str) -> float:
        """The sd the (TRASH - RECYCLING) gap would have from *task sampling alone*,
        if every seed's policy were identical.

        The gap is a difference of two independent binomial proportions, so at the
        worst case p = 0.5 its per-seed sd is sqrt(0.25/n_trash + 0.25/n_recycling),
        and across ten seeds the sd of that difference is the root-mean-square of
        the per-seed values. This is the experiment's **noise floor**. If the
        observed gap sd is close to it, the spread is not seed-to-seed variation in
        what the agent learned -- it is how few tasks of each family each seed drew,
        and no number of extra seeds fixes it. More tasks per family does.
        """
        trash = ResetFrequencyReport.family_denominators(arms=arms, arm=arm, family="TRASH")
        recycling = ResetFrequencyReport.family_denominators(arms=arms, arm=arm, family="RECYCLING")
        variances = [
            100.0**2 * (0.25 / t + 0.25 / r) for t, r in zip(trash, recycling, strict=True)
        ]
        return math.sqrt(statistics.mean(variances))

    @staticmethod
    def gap_curve(*, arms: dict, arm: str) -> list[tuple[float, float, float]]:
        """(mean transitions, mean gap, standard error of the gap) per checkpoint --
        the gap traced against training progress *within* one arm, which is the
        view that shows the gap is not a fixed property of a domain."""
        trash = ResetFrequencyReport.family_curve(arms=arms, arm=arm, family="TRASH")
        seeds = ResetFrequencyReport.seeds(arms=arms)
        curve = []
        for index in range(len(trash)):
            gaps = []
            for seed in seeds:
                _t, trash_solved, trash_total = arms[arm][seed]["TRASH"][index]
                _r, rec_solved, rec_total = arms[arm][seed]["RECYCLING"][index]
                gaps.append(
                    (100.0 * trash_solved / trash_total if trash_total else 0.0)
                    - (100.0 * rec_solved / rec_total if rec_total else 0.0)
                )
            stderr = statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else 0.0
            curve.append((trash[index][0], statistics.mean(gaps), stderr))
        return curve

    @staticmethod
    def family_curve(*, arms: dict, arm: str, family: str) -> list[tuple[float, float, float]]:
        """(mean transitions, mean %, standard error %) per checkpoint index.

        Indexed by checkpoint rather than by transition count because
        `num_online_transitions` is data-driven -- a period ending early on
        InteractionComplete is not charged the steps it did not take -- so seeds
        within an arm need not land on identical x-values. The x is therefore the
        mean achieved transitions at that checkpoint, not an assumed grid.
        """
        seeds = ResetFrequencyReport.seeds(arms=arms)
        num_sweeps = min(len(arms[arm][seed][family]) for seed in seeds)
        curve = []
        for index in range(num_sweeps):
            xs, percents = [], []
            for seed in seeds:
                transitions, solved, total = arms[arm][seed][family][index]
                xs.append(float(transitions))
                percents.append(100.0 * solved / total if total else 0.0)
            stderr = statistics.stdev(percents) / len(percents) ** 0.5 if len(percents) > 1 else 0.0
            curve.append((statistics.mean(xs), statistics.mean(percents), stderr))
        return curve

    @staticmethod
    def achieved_transitions(*, arms: dict, arm: str) -> list[float]:
        """Per-seed total online transitions actually taken. The design intends
        2500 everywhere; a shortfall is a residual confound, so it is measured
        rather than assumed."""
        return [
            float(max(triple[0] for triple in arms[arm][seed]["RECYCLING"]))
            for seed in ResetFrequencyReport.seeds(arms=arms)
        ]

    @staticmethod
    def trend_slopes(*, arms: dict) -> list[float]:
        """One OLS slope per seed: gap (pp) regressed on log2(period length),
        across all four arms.

        A rank correlation over four *arm means* cannot reach p < 0.05 -- Spearman
        at n = 4 bottoms out at p = 0.083 even for perfect monotonicity -- so the
        trend is tested the way the design is actually paired: fit the trend
        inside each seed, then test the ten per-seed slopes against zero. That is
        n = 10 and it answers the stated hypothesis directly.
        """
        xs = ResetFrequencyReport._log_periods()
        mean_x = statistics.mean(xs)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slopes = []
        for ys in ResetFrequencyReport._gaps_per_seed(arms=arms):
            mean_y = statistics.mean(ys)
            slopes.append(
                sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
            )
        return slopes

    @staticmethod
    def trend_rank_correlations(*, arms: dict) -> list[float]:
        """One Spearman rho per seed: the rank correlation between practice-period
        length and that seed's gap, over the four arms.

        The distribution-free companion to trend_slopes, for the same reason the
        tests below are exact rather than parametric -- and, like those slopes,
        tested at n = 10 across seeds rather than at n = 4 across arm means. Note
        the ceiling this carries: a per-seed rho is computed from four points, so
        it can only take a handful of values, and perfect monotonicity in a seed
        gives rho = 1 regardless of how large the movement was. It answers "does
        the gap order with period length", not "by how much".
        """
        xs = ResetFrequencyReport._log_periods()
        x_ranks = PairedTests._average_ranks(values=xs)
        correlations = []
        for ys in ResetFrequencyReport._gaps_per_seed(arms=arms):
            y_ranks = PairedTests._average_ranks(values=ys)
            correlations.append(ResetFrequencyReport._pearson(xs=x_ranks, ys=y_ranks))
        return correlations

    @staticmethod
    def _log_periods() -> list[float]:
        return [
            math.log2(_ARM_PERIOD[arm])
            for arm in sorted(_ARM_PERIOD, key=lambda arm: _ARM_PERIOD[arm])
        ]

    @staticmethod
    def _gaps_per_seed(*, arms: dict) -> list[list[float]]:
        """One list of four arm-ordered gaps per seed -- the paired unit both trend
        statistics are computed inside."""
        per_arm = [
            ResetFrequencyReport.gaps(arms=arms, arm=arm)
            for arm in sorted(_ARM_PERIOD, key=lambda arm: _ARM_PERIOD[arm])
        ]
        return [[gaps[index] for gaps in per_arm] for index in range(len(per_arm[0]))]

    @staticmethod
    def _pearson(*, xs: list[float], ys: list[float]) -> float:
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        spread = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
        # A seed whose gap is identical in all four arms has no variation to
        # correlate; rho is undefined there, and 0.0 is the honest reading -- that
        # seed provides no evidence either way.
        return covariance / spread if spread else 0.0

    @staticmethod
    def render_gap_figure(*, arms: dict, output: Path) -> None:
        """Headline, two panels that have to be read together.

        Left is the designed comparison: the (TRASH - RECYCLING) final gap against
        practice-period length, every seed drawn, with a bootstrap 95% CI on the
        mean. Per-seed points are drawn rather than an error bar alone because on
        this project a large sd has repeatedly turned out to be one collapsed seed
        rather than a broad spread, and no mean/CI rendering can tell those apart.

        Right is why the left panel cannot be read as a reset effect: the same gap
        plotted against training progress *within* each arm. The gap is
        hump-shaped -- small at the floor, large mid-training, small again at
        saturation -- so where an arm ends on that hump is set by how much it
        trained, which `--num-cycles` controls jointly with the reset frequency the
        left panel is indexed by. Publishing the left panel alone would be the
        overclaim this project has already retracted twice.
        """
        arms_ordered = sorted(_ARM_PERIOD, key=lambda arm: _ARM_PERIOD[arm])
        fig, (ax, progress_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
        periods = [_ARM_PERIOD[arm] for arm in arms_ordered]
        means, lows, highs = [], [], []
        for arm in arms_ordered:
            gaps = ResetFrequencyReport.gaps(arms=arms, arm=arm)
            low, high = PairedTests.bootstrap_ci(values=gaps)
            means.append(statistics.mean(gaps))
            lows.append(low)
            highs.append(high)
            # Fanned rather than stacked on one x: the gap can only land on
            # multiples of ~8pp, so several seeds routinely share a value exactly
            # and a single column would render ten seeds as three visible dots.
            offsets = [1.10 * 1.035**index for index in range(len(gaps))]
            ax.scatter(
                [_ARM_PERIOD[arm] * offset for offset in offsets],
                sorted(gaps),
                s=20,
                color="#0072B2",
                alpha=0.45,
                linewidths=0,
                zorder=2,
                label="individual seeds" if arm == arms_ordered[0] else None,
            )
        ax.axhline(0.0, color="dimgrey", linestyle=":", linewidth=1.2, zorder=1)
        ax.errorbar(
            periods,
            means,
            yerr=[
                [m - low for m, low in zip(means, lows, strict=True)],
                [high - m for m, high in zip(means, highs, strict=True)],
            ],
            marker="o",
            markersize=9,
            capsize=5,
            linewidth=2.2,
            color="#0072B2",
            zorder=3,
            label="mean, 95% bootstrap CI",
        )
        # Left of the marker, clear of both the CI whiskers above/below and the
        # per-seed fan to the right.
        for period, mean in zip(periods, means, strict=True):
            ax.annotate(
                f"{mean:+.1f}",
                xy=(period, mean),
                xytext=(-11, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="#333333",
            )
        shipped = _ARM_PERIOD[_SHIPPED_ARM]
        ax.axvline(shipped, color="dimgrey", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.annotate(
            "shipped config",
            xy=(shipped, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="dimgrey",
        )
        ax.set_xscale("log")
        ax.set_xticks(periods)
        ax.set_xticklabels([
            f"{period}\n({_ARM_CYCLES[arm]} resets)"
            for arm, period in zip(arms_ordered, periods, strict=True)
        ])
        ax.set_xlabel("Steps between free resets (log scale)")
        ax.set_ylabel("TRASH - RECYCLING final success gap (pp)")
        ax.set_title("The designed comparison:\nfinal gap vs practice-period length")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        # Sequential greys->black by period length: these are four levels of ONE
        # ordered variable, not four unrelated categories, so a categorical hue
        # per arm would be the wrong encoding. Each arm's endpoint is ringed, which
        # is the value the left panel plots.
        for index, arm in enumerate(arms_ordered):
            shade = str(0.68 - 0.21 * index)
            curve = ResetFrequencyReport.gap_curve(arms=arms, arm=arm)
            xs = [point[0] for point in curve]
            gaps = [point[1] for point in curve]
            errs = [point[2] for point in curve]
            progress_ax.plot(
                xs,
                gaps,
                color=shade,
                linewidth=2.0,
                label=f"{_ARM_CYCLES[arm]} x {_ARM_PERIOD[arm]}",
            )
            # Shaded, because without it four wiggly lines read as four confident
            # humps -- and the per-seed gap sd here is dominated by how few tasks
            # of each family a seed drew, not by what the agent learned.
            progress_ax.fill_between(
                xs,
                [g - e for g, e in zip(gaps, errs, strict=True)],
                [g + e for g, e in zip(gaps, errs, strict=True)],
                color=shade,
                alpha=0.15,
                linewidth=0,
            )
            progress_ax.scatter(
                [xs[-1]],
                [gaps[-1]],
                s=70,
                facecolor=shade,
                edgecolor="#D55E00",
                linewidth=2,
                zorder=4,
            )
        progress_ax.axhline(0.0, color="dimgrey", linestyle=":", linewidth=1.2)
        progress_ax.set_xlabel("Online transitions")
        progress_ax.set_ylabel("TRASH - RECYCLING gap (pp), mean over seeds")
        progress_ax.set_title(
            "Why that cannot be read as a reset effect:\n"
            "the gap moves with training progress (band = 1 s.e.)"
        )
        progress_ax.grid(True, alpha=0.3)
        progress_ax.legend(loc="best", fontsize=9, title="cycles x steps")
        progress_ax.annotate(
            "ringed = the endpoint\nthe left panel plots",
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color="#D55E00",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

        fig.suptitle(
            "Tossing Room: does a rarer free reset punish the irreversible family harder? "
            "10 paired seeds, 2500 online transitions per arm"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)

    @staticmethod
    def render_curve_figure(*, arms: dict, output: Path) -> None:
        """Per-family learning curves, one panel per arm, mean +- standard error.

        One panel per arm rather than one shared panel: the arms differ in how many
        sweeps they record (51 down to 6) and in how much sampler training each
        transition buys, so overlaying them invites exactly the cross-arm level
        comparison this experiment cannot support.
        """
        arms_ordered = sorted(_ARM_PERIOD, key=lambda arm: _ARM_PERIOD[arm])
        fig, axes = plt.subplots(1, len(arms_ordered), figsize=(16, 4.4), sharey=True)
        for ax, arm in zip(axes, arms_ordered, strict=True):
            for family, (color, marker) in _FAMILY_STYLE.items():
                curve = ResetFrequencyReport.family_curve(arms=arms, arm=arm, family=family)
                xs = [point[0] for point in curve]
                means = [point[1] for point in curve]
                errs = [point[2] for point in curve]
                ax.plot(
                    xs,
                    means,
                    color=color,
                    linewidth=2.0,
                    marker=marker,
                    markersize=4 if len(xs) > 12 else 7,
                    markevery=max(1, len(xs) // 8),
                    label=family,
                )
                ax.fill_between(
                    xs,
                    [m - e for m, e in zip(means, errs, strict=True)],
                    [m + e for m, e in zip(means, errs, strict=True)],
                    color=color,
                    alpha=0.15,
                )
            ax.set_title(
                f"{_ARM_CYCLES[arm]} cycles x {_ARM_PERIOD[arm]} steps"
                + ("  (shipped)" if arm == _SHIPPED_ARM else "")
            )
            ax.set_xlabel("Online transitions")
            ax.set_ylim(-3, 103)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("% evaluation tasks solved")
        axes[0].legend(loc="lower right", fontsize=9)
        fig.suptitle(
            "Tossing Room per-family learning curves, mean +- standard error over 10 seeds "
            "(levels are NOT comparable across panels: cycles = sampler refits)"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


class PairedTests(BaseModel):
    """Exact paired tests over the shared seeds, plus the power question every
    non-significant result on this project has to answer.

    Exact by enumeration rather than approximate: at n = 10 the sign-flip null has
    2**10 = 1024 members, so both tests below enumerate it in full. That sidesteps
    the normal approximation (badly behaved at n = 10), the continuity correction,
    and the tie correction all at once -- and it means this file needs no special
    functions, hence no scipy, which is not a dependency of this project.
    """

    model_config = ConfigDict(frozen=True)

    statistic: float
    p_value: float
    num_zero_differences: int

    @staticmethod
    def wilcoxon_signed_rank(*, differences: list[float]) -> "PairedTests":
        """Two-sided exact Wilcoxon signed-rank test of differences against zero.

        Zero differences are dropped (Wilcoxon's original handling) and counted, so
        a result driven by "most seeds did not move at all" is visible rather than
        buried. Tied |differences| get average ranks, and the null is enumerated
        over sign assignments to *those* ranks, which stays exact under ties.
        """
        nonzero = [d for d in differences if d != 0.0]
        num_zero = len(differences) - len(nonzero)
        if not nonzero:
            return PairedTests(statistic=0.0, p_value=1.0, num_zero_differences=num_zero)
        ranks = PairedTests._average_ranks(values=[abs(d) for d in nonzero])
        observed = sum(rank for rank, d in zip(ranks, nonzero, strict=True) if d > 0)
        total = sum(ranks)
        # The statistic's null distribution is symmetric about total/2, so the
        # two-sided p is the mass at least as far from the centre as observed.
        distance = abs(observed - total / 2)
        extreme = sum(
            1
            for signs in itertools.product((0, 1), repeat=len(ranks))
            if abs(sum(rank * sign for rank, sign in zip(ranks, signs, strict=True)) - total / 2)
            >= distance - 1e-9
        )
        return PairedTests(
            statistic=observed,
            p_value=extreme / 2 ** len(ranks),
            num_zero_differences=num_zero,
        )

    @staticmethod
    def sign_flip(*, differences: list[float]) -> "PairedTests":
        """Two-sided exact sign-flip permutation test on the *mean* difference --
        the paired t-test's assumption-free twin, and the reason no t distribution
        (and so no incomplete beta function) is needed here.
        """
        num_zero = sum(1 for d in differences if d == 0.0)
        if all(d == 0.0 for d in differences):
            return PairedTests(statistic=0.0, p_value=1.0, num_zero_differences=num_zero)
        observed = abs(sum(differences))
        extreme = sum(
            1
            for signs in itertools.product((-1, 1), repeat=len(differences))
            if abs(sum(d * sign for d, sign in zip(differences, signs, strict=True)))
            >= observed - 1e-9
        )
        return PairedTests(
            statistic=statistics.mean(differences),
            p_value=extreme / 2 ** len(differences),
            num_zero_differences=num_zero,
        )

    @staticmethod
    def seeds_for_80_percent_power(*, differences: list[float]) -> float:
        """Paired-sample size that would give 80% power at alpha = 0.05 two-sided
        for an effect the size of the one observed. Reported whenever p > 0.05, so
        "not established" comes with the cost of establishing it.

        Returns infinity when the observed mean difference is exactly zero: no
        sample size detects a zero effect.
        """
        mean = statistics.mean(differences)
        if mean == 0.0 or len(differences) < 2:
            return math.inf
        sd = statistics.stdev(differences)
        return (_Z_ALPHA + _Z_POWER) ** 2 * (sd / mean) ** 2

    @staticmethod
    def bootstrap_ci(*, values: list[float], num_resamples: int = 20000) -> tuple[float, float]:
        """Percentile bootstrap 95% CI of the mean. Seeded, so the figure is
        reproducible; the CI is drawn rather than an sd bar because the per-seed
        scatter beside it already shows the spread."""
        rng = np.random.default_rng(0)
        sample = np.asarray(values, dtype=float)
        draws = rng.choice(sample, size=(num_resamples, sample.size), replace=True).mean(axis=1)
        return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

    @staticmethod
    def _average_ranks(*, values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2 + 1
            for index in order[position : end + 1]:
                ranks[index] = average
            position = end + 1
        return ranks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help='Repeatable, "armA=/path/to/sweep/root". Aggregation mode.',
    )
    parser.add_argument("--aggregate-output", type=Path, default=None)
    parser.add_argument("--arms-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Headline gap figure.")
    parser.add_argument("--curves-output", type=Path, default=None, help="Per-family curves.")
    return parser.parse_args()


def _format_power(*, needed: float) -> str:
    """Render `seeds_for_80_percent_power` for a table cell.

    `<1` rather than `0`: an effect large enough that fewer than one pair would
    suffice is not "zero seeds needed", and a bare 0 reads like a bug or like the
    test being impossible. It happens here on the RECYCLING arm-extreme difference,
    where the effect is ~5.6 sds wide.
    """
    if not math.isfinite(needed):
        return "infinite (zero observed effect)"
    return "<1" if needed < 1.0 else f"{needed:.0f}"


def _print_report(*, arms: dict) -> None:
    arms_ordered = sorted(_ARM_PERIOD, key=lambda arm: _ARM_PERIOD[arm])
    seeds = ResetFrequencyReport.seeds(arms=arms)
    print(f"Shared seeds: {', '.join(seeds)}\n")

    print(f"{'arm':>5} {'period':>7} {'cycles':>7} {'transitions':>18}", end="")
    for family in _FAMILY_STYLE:
        print(f" {family:>18}", end="")
    print(f" {'gap (T-R)':>16}")
    for arm in arms_ordered:
        achieved = ResetFrequencyReport.achieved_transitions(arms=arms, arm=arm)
        print(
            f"{arm:>5} {_ARM_PERIOD[arm]:>7} {_ARM_CYCLES[arm]:>7} "
            f"{statistics.mean(achieved):>10.0f} +-{statistics.stdev(achieved):<5.0f}",
            end="",
        )
        for family in _FAMILY_STYLE:
            rates = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family=family)
            print(f" {statistics.mean(rates):>10.1f} +-{statistics.stdev(rates):<5.1f}", end="")
        gaps = ResetFrequencyReport.gaps(arms=arms, arm=arm)
        print(f" {statistics.mean(gaps):>8.1f} +-{statistics.stdev(gaps):<6.1f}")

    print("\nPer-seed (TRASH - RECYCLING) gap, pp:")
    print(f"{'arm':>5} " + " ".join(f"{('s' + seed):>6}" for seed in seeds))
    for arm in arms_ordered:
        gaps = ResetFrequencyReport.gaps(arms=arms, arm=arm)
        print(f"{arm:>5} " + " ".join(f"{gap:>6.1f}" for gap in gaps))

    # The manipulation check, printed before anything derived from it. Both halves
    # of this experiment's design are assertions about the runs, not about the
    # code: that every seed evaluated on the same fixed composition, and that every
    # arm spent the same experience budget. PR #39's finding was a design that did
    # not isolate what it claimed, so neither is taken on trust.
    print("\nManipulation check:")
    expected = ResetFrequencyReport.expected_composition()
    composition = ResetFrequencyReport.composition_violations(arms=arms)
    transitions = ResetFrequencyReport.transition_violations(arms=arms)
    print(
        f"  test-set composition expected {expected}: "
        + ("OK, every arm/seed/sweep" if not composition else f"VIOLATED {composition[:5]}")
    )
    print(
        "  online transitions expected 2500: "
        + ("OK, every arm/seed" if not transitions else f"VIOLATED {transitions[:5]}")
    )
    for family in _FAMILY_STYLE:
        denominators = ResetFrequencyReport.family_denominators(
            arms=arms, arm=arms_ordered[0], family=family
        )
        print(f"  {family:>10}: {denominators}  (identical in every arm -- same test RNG stream)")

    print("\nPooled final count, solved/run over all 10 seeds (descriptive only):")
    print(f"{'arm':>5} " + " ".join(f"{family:>20}" for family in _FAMILY_STYLE))
    for arm in arms_ordered:
        cells = []
        for family in _FAMILY_STYLE:
            solved, total = ResetFrequencyReport.pooled_counts(arms=arms, arm=arm, family=family)
            cells.append(f"{f'{solved}/{total}':>12} ({100.0 * solved / total:5.1f}%)")
        print(f"{arm:>5} " + " ".join(cells))

    print("\nPer-seed final solved/run, by family:")
    for family in _FAMILY_STYLE:
        print(f"  {family}:")
        for arm in arms_ordered:
            cells = []
            for seed in seeds:
                _t, solved, total = max(arms[arm][seed][family], key=lambda triple: triple[0])
                cells.append(f"{solved}/{total}")
            print(f"    {arm:>5} " + " ".join(f"{cell:>6}" for cell in cells))

    print("\nIs the gap's spread real, or just task-sampling noise?")
    for arm in arms_ordered:
        observed = statistics.stdev(ResetFrequencyReport.gaps(arms=arms, arm=arm))
        predicted = ResetFrequencyReport.predicted_gap_noise(arms=arms, arm=arm)
        print(
            f"  {arm}: observed gap sd {observed:5.1f}pp vs {predicted:5.1f}pp predicted from "
            f"binomial task sampling alone ({100 * observed / predicted:.0f}% of the noise floor)"
        )

    # Printed BEFORE the tests, because it decides whether they mean anything: if
    # the arms sit at different training progress, their gaps differ for that
    # reason alone and no p-value below is attributable to reset frequency.
    print("\nAre the arms even comparable? (seeds out of 10)")
    print(
        f"{'arm':>5} {'final TRASH':>13} {'both at 100%':>13} {'TRASH no better':>17} "
        f"{'gap exactly 0':>14}"
    )
    for arm in arms_ordered:
        diagnostics = ResetFrequencyReport.progress_diagnostics(arms=arms, arm=arm)
        trash = ResetFrequencyReport.final_rates(arms=arms, arm=arm, family="TRASH")
        print(
            f"{arm:>5} {statistics.mean(trash):>12.1f}% "
            f"{diagnostics['at_ceiling_both']:>13} "
            f"{diagnostics['trash_no_better_than_untrained']:>17} "
            f"{diagnostics['zero_gap']:>14}"
        )

    # The discriminating check. The gap is hump-shaped in training progress, so a
    # cross-arm gap difference is only attributable to reset frequency if the arms
    # end at comparable competence. TRASH answers "equally trained?"; EMPTY is the
    # control, where no Throw and no irreversible action exist at all.
    print("\nAre the arms equally trained? (paired, armD 500 - armA 50, exact tests)")
    for family in _FAMILY_STYLE:
        differences = ResetFrequencyReport.family_differences(
            arms=arms, family=family, from_arm="armA", to_arm="armD"
        )
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
        flip = PairedTests.sign_flip(differences=differences)
        needed = PairedTests.seeds_for_80_percent_power(differences=differences)
        power = _format_power(needed=needed)
        print(
            f"  {family:>10}: mean {statistics.mean(differences):+6.1f}pp, "
            f"sd {statistics.stdev(differences):5.1f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f}, sign-flip p={flip.p_value:.4f}, "
            f"n for 80% power: {power}"
        )

    level = ResetFrequencyReport.matchable_level(arms=arms)
    print(f"\nProgress-matched gap, at the first checkpoint where TRASH reaches {level:.1f}%:")
    for arm in arms_ordered:
        matched = ResetFrequencyReport.progress_matched_gaps(arms=arms, arm=arm, level=level)
        present = [gap for gap in matched if gap is not None]
        # How many seeds match at sweep 0 -- i.e. before any practice. If most do,
        # this "progress-matched" gap is just the untrained gap, and it cannot
        # discriminate the arms at all (they share an untrained policy by
        # construction). Reported so that is visible rather than inferred.
        at_start = sum(
            1
            for seed in seeds
            if arms[arm][seed]["TRASH"][0][2]
            and 100.0 * arms[arm][seed]["TRASH"][0][1] / arms[arm][seed]["TRASH"][0][2] >= level
        )
        summary = (
            f"mean {statistics.mean(present):+.1f}pp, sd {statistics.stdev(present):.1f}"
            if len(present) > 1
            else "not enough seeds reach this level"
        )
        print(
            f"  {arm}: {len(present)}/10 seeds, {summary}, {at_start} matched before any practice"
        )

    # Tested, not eyeballed. On the fixed-composition re-run the *final* gap does
    # trend with period length at p < 0.01, so "is that the treatment or the hump?"
    # stops being rhetorical and becomes the question the whole log turns on. The
    # progress-matched gap is the same contrast with training progress held fixed,
    # so if the trend were a reset effect it should survive here.
    matched_extremes = ResetFrequencyReport.progress_matched_differences(
        arms=arms, from_arm="armA", to_arm="armD", level=level
    )
    if matched_extremes:
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=matched_extremes)
        flip = PairedTests.sign_flip(differences=matched_extremes)
        needed = PairedTests.seeds_for_80_percent_power(differences=matched_extremes)
        power = _format_power(needed=needed)
        print(
            f"  progress-matched gap, armD - armA (n={len(matched_extremes)}): "
            f"mean {statistics.mean(matched_extremes):+.2f}pp, "
            f"sd {statistics.stdev(matched_extremes):.2f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f}, sign-flip p={flip.p_value:.4f}, "
            f"n for 80% power: {power}"
        )

    print("\nPaired tests (exact, 10 shared seeds):")
    extremes = [
        d - a
        for a, d in zip(
            ResetFrequencyReport.gaps(arms=arms, arm="armA"),
            ResetFrequencyReport.gaps(arms=arms, arm="armD"),
            strict=True,
        )
    ]
    for label, differences in (
        ("gap: armD (500) - armA (50)", extremes),
        (
            "per-seed trend slope (pp per doubling of period)",
            ResetFrequencyReport.trend_slopes(arms=arms),
        ),
        (
            "per-seed Spearman rho of gap vs period",
            ResetFrequencyReport.trend_rank_correlations(arms=arms),
        ),
    ):
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
        flip = PairedTests.sign_flip(differences=differences)
        needed = PairedTests.seeds_for_80_percent_power(differences=differences)
        power = _format_power(needed=needed)
        print(
            f"  {label}: mean {statistics.mean(differences):+.2f}, "
            f"sd {statistics.stdev(differences):.2f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f} ({wilcoxon.num_zero_differences} zero diffs), "
            f"sign-flip p={flip.p_value:.4f}, n for 80% power: {power}"
        )

    print("\nWithin-arm gap vs zero (is RECYCLING behind TRASH at all?):")
    for arm in arms_ordered:
        gaps = ResetFrequencyReport.gaps(arms=arms, arm=arm)
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=gaps)
        flip = PairedTests.sign_flip(differences=gaps)
        needed = PairedTests.seeds_for_80_percent_power(differences=gaps)
        power = _format_power(needed=needed)
        print(
            f"  {arm}: mean {statistics.mean(gaps):+.1f}pp, Wilcoxon p={wilcoxon.p_value:.4f}, "
            f"sign-flip p={flip.p_value:.4f}, {wilcoxon.num_zero_differences} zero diffs, "
            f"n for 80% power: {power}"
        )


def main() -> None:
    args = _parse_args()
    if args.arm:
        arm_dirs = {}
        for entry in args.arm:
            name, separator, path = entry.partition("=")
            if not separator:
                raise ValueError(f"--arm must look like armA=DIR, got {entry!r}")
            arm_dirs[name] = Path(path)
        aggregate = ResetFrequencyReport.aggregate(arm_dirs=arm_dirs)
        if args.aggregate_output is None:
            raise ValueError("--arm requires --aggregate-output")
        # Compact, one line, matching 2026-08-03-ballring-arms.json: this is
        # recorded data, not source, and an indented rendering of ~950 sweeps is
        # 40x the bytes for no added readability.
        args.aggregate_output.write_text(json.dumps(aggregate, sort_keys=True))
        print(f"wrote {args.aggregate_output}")
        return

    if args.arms_json is None:
        raise ValueError("pass either --arm ... --aggregate-output, or --arms-json")
    arms = ResetFrequencyReport.load_arms(json_path=args.arms_json)
    if args.output is not None:
        ResetFrequencyReport.render_gap_figure(arms=arms, output=args.output)
        print(f"wrote {args.output}")
    if args.curves_output is not None:
        ResetFrequencyReport.render_curve_figure(arms=arms, output=args.curves_output)
        print(f"wrote {args.curves_output}")
    _print_report(arms=arms)


if __name__ == "__main__":
    main()
