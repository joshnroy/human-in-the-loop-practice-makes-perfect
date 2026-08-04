"""Post-run analysis for the Tossing Room reset-*interval* experiment: with
training held fixed, does rescuing the robot more often reduce the penalty its one
irreversible action carries?

Tossing Room has exactly one genuinely terminal failure. On the `RECYCLING` family
a missed `Throw` strands the robot: the pile sits in room 3, the recycling bin in
room 1, and `blocked_right_from = 2` makes room 3 unreachable once the item is
gone, so Fast Downward correctly reports no plan for the rest of the period. A
missed `TRASH` throw is merely expensive (a round trip buys a fresh item), and the
`EMPTY` family has no `Throw` at all -- it is the control.

`PracticeLoop` used to hand out a free reset only at the top of each practice
cycle, which caps what stranding can cost at "the rest of this period". The
prediction: reset more often and a stranded robot is rescued sooner, so RECYCLING
suffers less and the (TRASH - RECYCLING) gap shrinks.

**What is different here from PR #39's reset-*frequency* experiment, and why that
one could not answer this.** It varied the period length with
`--num-cycles` inverted to hold transitions fixed -- but `--num-cycles` sets the
number of free resets *and* the number of sampler refits with one number, so its
arms ended ~40 competence points apart on identical experience and its gap peaked
mid-curve rather than at the fewest-resets end. It measured the training
difference, not irreversibility. These four arms instead hold `--num-cycles 25`
and `--max-steps-per-interaction 100` in **every** arm -- 25 refits and 2500
transitions everywhere -- and vary only `--practice-reset-interval` (10/25/50/100
steps, a 10x range in reset frequency). Arm D's interval equals the period length,
so it is the old behaviour stated explicitly.

The design's own checks, all computed here and all printed *before* any p-value:

* `reset_counts` -- the manipulation check. `Metrics.num_practice_resets` counts
  resets as they actually happened, so "the arms differed in reset frequency" is a
  measurement rather than a restatement of the flag. Expected 250/100/50/25.
* `family_denominators` -- the realised per-family test-set composition, asserted
  rather than assumed to be 14 TRASH / 14 RECYCLING / 2 EMPTY.
* `family_differences` -- the progress-match check PR #39
  established as mandatory. The gap is a hump-shaped function of training progress,
  so two arms only have comparable gaps if they end at comparable competence. If
  they do not, the same confound has reappeared through another door.
* `achieved_transitions` -- a period that ends early on `InteractionComplete` is
  not charged the steps it did not take, and a reset can revive a robot that had
  nothing applicable left, so equal experience is measured too.

Reads only already-produced outputs (CLAUDE.md's `analysis/` convention -- never
runs a simulation). Two modes, both post-run, matching PR #39's own analysis
script, whose shape this reuses:

* `--arm NAME=DIR ... --aggregate-output JSON` condenses each sweep's
  `DIR/<method>/<seed>/stats.json` into a committed per-family aggregate. Raw sweep
  directories live outside the repo and do not travel between machines (see
  `docs/experiment-logs/2026-08-03-cross-machine-reproducibility.md`), so the
  aggregate is the record that survives.
* `--arms-json JSON --output PNG --curves-output PNG` regenerates every figure,
  table and p-value in the experiment log from that aggregate alone.

**Statistics.** Every arm ran the same fixed seeds 0..19, so every comparison is
*paired*, and scipy is not a dependency of this project, so `PairedTests` computes
each p-value **exactly** by enumerating the sign-flip null in full. At n = 20 that
null has 2**20 = 1,048,576 members, which a naive `itertools.product` loop cannot
do in reasonable time -- `_subset_sums` enumerates it meet-in-the-middle instead,
splitting the terms in half and adding the two halves' subset sums by numpy
broadcasting. Same exact answer, no normal approximation, no continuity or tie
correction, no special functions.
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

# Arm -> --practice-reset-interval, in environment steps. This is the ONLY thing
# that differs between arms; --num-cycles 25 and --max-steps-per-interaction 100
# are held fixed everywhere. The ordering of this dict is the ordering of every
# table and figure below.
_ARM_INTERVAL = {"armA": 10, "armB": 25, "armC": 50, "armD": 100}

# The period length and cycle count every arm shares -- carried as constants
# precisely because the previous experiment's confound was that these moved.
_PERIOD_STEPS = 100
_NUM_CYCLES = 25

# armD's interval equals the period length, so it reproduces the behaviour that
# shipped before --practice-reset-interval existed. The figures mark it.
_BASELINE_ARM = "armD"

# --num-test-tasks every arm ran with, and therefore the composition the domain
# allocates for it. Named rather than inlined because the composition is a function
# of it alone.
_NUM_TEST_TASKS = 30


def expected_denominators(*, num_test_tasks: int = _NUM_TEST_TASKS) -> dict[str, int]:
    """The deterministic Tossing Room test-set composition this experiment relies
    on -- 14 TRASH / 14 RECYCLING / 2 EMPTY at 30 test tasks.

    Asked of the domain rather than hardcoded here, because a hardcoded copy is a
    second source of truth that silently goes stale: the allocation rule lives in
    `TossingRoomTasks.test_goal_type_counts`, which is public precisely so an
    analysis can assert the composition of a run it is reading back. Nothing about
    it depends on the seed or the layout, so a throwaway instance answers it.

    The realised counts are then checked against this per seed
    (`composition_violations`) rather than assumed. A prerequisite landing does not
    relieve the experiment of its manipulation check -- that discipline is what
    caught the previous reset experiment's confound.
    """
    counts = TossingRoomTasks(
        env=TossingRoomEnvironment(), num_test_tasks=num_test_tasks
    ).test_goal_type_counts()
    return {goal_type.name: count for goal_type, count in counts.items()}


# Okabe-Ito, verified colourblind-safe for this exact trio (worst adjacent pair
# deltaE 11.0 deutan / 25.8 normal vision, all three inside the lightness band).
# Each family also gets its own marker, so identity is never colour-alone.
_FAMILY_STYLE = {
    "RECYCLING": ("#0072B2", "o"),
    "TRASH": ("#D55E00", "s"),
    "EMPTY": ("#009E73", "^"),
}

# z_{0.975} and z_{0.80}, for the "how many seeds would 80% power need?" line that
# every non-significant result on this project is required to carry, and for the
# minimum detectable effect this design actually had.
_Z_ALPHA = 1.959964
_Z_POWER = 0.841621


class ResetIntervalReport:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def aggregate(*, arm_dirs: dict[str, Path], method: str = "ees") -> dict:
        """Condenses raw sweep directories into the committed aggregate,
        `{arm: {seed: {"resets": int, "families": {family: [[transitions, solved,
        total], ...]}}}}`.

        One level deeper than `2026-08-03-ballring-arms.json`'s `[transitions,
        solved, total]` triples so each task family keeps its own curve, and with
        the realised reset count carried alongside -- without it the committed
        record could not answer "did the manipulation happen?", which is the first
        question this experiment has to survive.

        Reads each `stats.json` back through `Metrics.model_validate_json` rather
        than parsing the JSON by hand, per analysis/README.md.
        """
        aggregate: dict = {}
        for arm, root in sorted(arm_dirs.items(), key=lambda item: _ARM_INTERVAL[item[0]]):
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
                    for family, (solved, total) in ResetIntervalReport._counts(
                        breakdown=breakdown
                    ).items():
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
        missing = sorted(set(_ARM_INTERVAL) - set(arms))
        if missing:
            raise ValueError(f"aggregate JSON is missing arms: {missing}")
        return arms

    @staticmethod
    def seeds(*, arms: dict) -> list[str]:
        """The seeds every arm shares, sorted numerically. Pairing is only valid
        over these, so the intersection is taken rather than assumed."""
        shared: set[str] | None = None
        for seeds in arms.values():
            shared = set(seeds) if shared is None else shared & set(seeds)
        return sorted(shared or set(), key=int)

    @staticmethod
    def reset_counts(*, arms: dict, arm: str) -> list[int]:
        """Per-seed count of the free resets that actually happened -- the
        manipulation check.

        This is the first thing the report prints, because a design that silently
        did not vary what it claimed is exactly how the previous reset experiment
        went wrong. `expected_resets` gives the number the design intends; a
        mismatch invalidates everything below it.
        """
        return [arms[arm][seed]["resets"] for seed in ResetIntervalReport.seeds(arms=arms)]

    @staticmethod
    def expected_resets(*, arm: str) -> int:
        """Resets the design intends for one arm: one at the top of each period
        plus one every `interval` steps inside it, suppressed on the period's last
        step (where the next period's own reset follows immediately)."""
        return _NUM_CYCLES * (_PERIOD_STEPS // _ARM_INTERVAL[arm])

    @staticmethod
    def final_rates(*, arms: dict, arm: str, family: str) -> list[float]:
        """Per-seed % solved for one family at the arm's LAST evaluation sweep,
        ordered by seed so the lists across arms are index-aligned for pairing."""
        rates = []
        for seed in ResetIntervalReport.seeds(arms=arms):
            _transitions, solved, total = max(
                arms[arm][seed]["families"][family], key=lambda triple: triple[0]
            )
            rates.append(100.0 * solved / total if total else 0.0)
        return rates

    @staticmethod
    def gaps(*, arms: dict, arm: str) -> list[float]:
        """Per-seed (TRASH - RECYCLING) final-sweep gap, in percentage points --
        the primary metric.

        Within-arm by construction, so it cancels anything that shifts both
        families' level together. Unlike the previous experiment, that is not the
        only defence here: every arm gets the same 25 refits over the same 2500
        transitions, so the arms should sit at the same point on the
        gap-versus-progress hump to begin with -- which `family_differences`
        checks rather than assumes.
        """
        trash = ResetIntervalReport.final_rates(arms=arms, arm=arm, family="TRASH")
        recycling = ResetIntervalReport.final_rates(arms=arms, arm=arm, family="RECYCLING")
        return [t - r for t, r in zip(trash, recycling, strict=True)]

    @staticmethod
    def family_differences(*, arms: dict, family: str, from_arm: str, to_arm: str) -> list[float]:
        """Per-seed (to_arm - from_arm) difference in one family's final success
        rate -- paired, since every arm ran the same fixed seeds.

        The progress-match check. Run on TRASH and RECYCLING it asks "are the arms
        equally trained?", which decides whether a cross-arm gap difference is
        attributable to reset frequency at all. Run on EMPTY it is the control: no
        `Throw`, no stochastic skill, so movement there is not irreversibility.
        """
        return [
            t - f
            for f, t in zip(
                ResetIntervalReport.final_rates(arms=arms, arm=from_arm, family=family),
                ResetIntervalReport.final_rates(arms=arms, arm=to_arm, family=family),
                strict=True,
            )
        ]

    @staticmethod
    def mean_rate_over_training(*, arms: dict, arm: str, family: str) -> list[float]:
        """Per-seed mean success rate across *all* checkpoints for one family -- the
        normalised area under that seed's learning curve, i.e. how fast it learned
        rather than where it ended up.

        **Post-hoc.** The pre-specified metric was the final-checkpoint gap, and it
        returned a saturated null: by 2500 transitions every arm sits at 95-99% on
        both throw families, so the endpoint has almost no room to differ. The
        learning curves plainly do differ, and this is the statistic that reads
        them without choosing a checkpoint after the fact.

        Chosen over "transitions to reach X%" precisely because that one is
        censored -- a seed that never reaches the threshold has no value, and at
        least one here ends at 21% -- which would silently drop the worst seeds
        from the comparison and flatter the arm they fall in. Averaging the curve
        is defined for every seed.
        """
        seeds = ResetIntervalReport.seeds(arms=arms)
        means = []
        for seed in seeds:
            triples = arms[arm][seed]["families"][family]
            means.append(
                statistics.mean(
                    100.0 * solved / total if total else 0.0 for _t, solved, total in triples
                )
            )
        return means

    @staticmethod
    def mean_gap_over_training(*, arms: dict, arm: str) -> list[float]:
        """Per-seed mean (TRASH - RECYCLING) gap across all checkpoints.

        **Post-hoc**, for the same reason as `mean_rate_over_training`: the final
        gap is measured where both families have saturated. This asks whether
        RECYCLING trails TRASH *along the way*, which is where a penalty for
        irreversibility would show up if the run is long enough to wash it out by
        the end.
        """
        trash = ResetIntervalReport.mean_rate_over_training(arms=arms, arm=arm, family="TRASH")
        recycling = ResetIntervalReport.mean_rate_over_training(
            arms=arms, arm=arm, family="RECYCLING"
        )
        return [t - r for t, r in zip(trash, recycling, strict=True)]

    @staticmethod
    def pooled_rate(*, arms: dict, arm: str, family: str) -> float:
        """One arm's final-sweep success rate for a family, pooled over all seeds
        (total solved / total tasks) rather than averaged over per-seed rates.

        Descriptive only -- every paired test stays on per-seed values, since
        pooling destroys the pairing. Reported because the per-seed mean averages
        20 proportions with denominators of 14, so it carries more sampling noise
        than the 280-task pooled figure while estimating the same quantity.
        """
        solved = total = 0
        for seed in ResetIntervalReport.seeds(arms=arms):
            _transitions, seed_solved, seed_total = max(
                arms[arm][seed]["families"][family], key=lambda triple: triple[0]
            )
            solved += seed_solved
            total += seed_total
        return 100.0 * solved / total if total else 0.0

    @staticmethod
    def family_denominators(*, arms: dict, arm: str, family: str) -> list[int]:
        """How many test tasks of one family each seed actually held.

        Asserted against `expected_denominators()` rather than merely reported: the
        deterministic 14/14/2 composition is what sets this experiment's noise
        floor, and the previous experiment's binding limitation was a *sampled*
        composition whose per-seed counts moved.
        """
        return [
            arms[arm][seed]["families"][family][0][2]
            for seed in ResetIntervalReport.seeds(arms=arms)
        ]

    @staticmethod
    def composition_violations(*, arms: dict) -> list[str]:
        """Every (arm, seed, family) whose realised test-task count is not the
        designed one. Empty is the only acceptable result."""
        violations = []
        for arm in _ARM_INTERVAL:
            for family, expected in expected_denominators().items():
                counts = ResetIntervalReport.family_denominators(arms=arms, arm=arm, family=family)
                for seed, count in zip(ResetIntervalReport.seeds(arms=arms), counts, strict=True):
                    if count != expected:
                        violations.append(f"{arm} seed {seed} {family}: {count} != {expected}")
        return violations

    @staticmethod
    def predicted_gap_noise(*, arms: dict, arm: str) -> float:
        """The sd the (TRASH - RECYCLING) gap would have from *task sampling alone*,
        if every seed's policy were identical -- this experiment's noise floor.

        The gap is a difference of two independent binomial proportions, so at the
        worst case p = 0.5 its per-seed sd is
        sqrt(0.25/n_trash + 0.25/n_recycling), and across seeds the sd of that
        difference is the root-mean-square of the per-seed values. At the designed
        14/14 that is 18.9pp. An observed sd near the floor means the spread is how
        few tasks each seed held, which more seeds cannot fix; an observed sd far
        above it is genuine seed-to-seed heterogeneity, which only more seeds can.
        """
        trash = ResetIntervalReport.family_denominators(arms=arms, arm=arm, family="TRASH")
        recycling = ResetIntervalReport.family_denominators(arms=arms, arm=arm, family="RECYCLING")
        variances = [
            100.0**2 * (0.25 / t + 0.25 / r) for t, r in zip(trash, recycling, strict=True)
        ]
        return math.sqrt(statistics.mean(variances))

    @staticmethod
    def gap_curve(*, arms: dict, arm: str) -> list[tuple[float, float, float]]:
        """(mean transitions, mean gap, standard error of the gap) per checkpoint --
        the gap traced against training progress *within* one arm, which is the
        view that shows the gap is not a fixed property of a domain."""
        trash = ResetIntervalReport.family_curve(arms=arms, arm=arm, family="TRASH")
        seeds = ResetIntervalReport.seeds(arms=arms)
        curve = []
        for index in range(len(trash)):
            gaps = []
            for seed in seeds:
                _t, trash_solved, trash_total = arms[arm][seed]["families"]["TRASH"][index]
                _r, rec_solved, rec_total = arms[arm][seed]["families"]["RECYCLING"][index]
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
        seeds = ResetIntervalReport.seeds(arms=arms)
        num_sweeps = min(len(arms[arm][seed]["families"][family]) for seed in seeds)
        curve = []
        for index in range(num_sweeps):
            xs, percents = [], []
            for seed in seeds:
                transitions, solved, total = arms[arm][seed]["families"][family][index]
                xs.append(float(transitions))
                percents.append(100.0 * solved / total if total else 0.0)
            stderr = statistics.stdev(percents) / len(percents) ** 0.5 if len(percents) > 1 else 0.0
            curve.append((statistics.mean(xs), statistics.mean(percents), stderr))
        return curve

    @staticmethod
    def achieved_transitions(*, arms: dict, arm: str) -> list[float]:
        """Per-seed total online transitions actually taken. The design intends
        2500 everywhere, and unlike the previous experiment that is not automatic:
        a mid-period reset can revive a robot whose practice planner had nothing
        applicable left and would otherwise have raised InteractionComplete, so
        frequent resets could buy an arm extra experience. Measured, not assumed.
        """
        return [
            float(max(triple[0] for triple in arms[arm][seed]["families"]["RECYCLING"]))
            for seed in ResetIntervalReport.seeds(arms=arms)
        ]

    @staticmethod
    def trend_slopes(*, arms: dict) -> list[float]:
        """One OLS slope per seed: gap (pp) regressed on log2(reset interval),
        across all four arms.

        A rank correlation over four *arm means* cannot reach p < 0.05 -- Spearman
        at n = 4 bottoms out at p = 0.083 even for perfect monotonicity -- so the
        trend is tested the way the design is actually paired: fit the trend inside
        each seed, then test the per-seed slopes against zero. The hypothesis
        predicts a *positive* slope (a longer interval means rarer rescues, so a
        wider gap).
        """
        xs = ResetIntervalReport._log_intervals()
        mean_x = statistics.mean(xs)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slopes = []
        for ys in ResetIntervalReport._gaps_per_seed(arms=arms):
            mean_y = statistics.mean(ys)
            slopes.append(
                sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
            )
        return slopes

    @staticmethod
    def trend_rank_correlations(*, arms: dict) -> list[float]:
        """One Spearman rho per seed: the rank correlation between reset interval
        and that seed's gap, over the four arms.

        The distribution-free companion to trend_slopes, tested across seeds rather
        than across arm means for the same reason. Note its ceiling: a per-seed rho
        is computed from four points, so perfect monotonicity in a seed gives
        rho = 1 regardless of how large the movement was. It answers "does the gap
        order with the interval", not "by how much".
        """
        xs = ResetIntervalReport._log_intervals()
        x_ranks = PairedTests._average_ranks(values=xs)
        correlations = []
        for ys in ResetIntervalReport._gaps_per_seed(arms=arms):
            y_ranks = PairedTests._average_ranks(values=ys)
            correlations.append(ResetIntervalReport._pearson(xs=x_ranks, ys=y_ranks))
        return correlations

    @staticmethod
    def _log_intervals() -> list[float]:
        return [
            math.log2(_ARM_INTERVAL[arm])
            for arm in sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
        ]

    @staticmethod
    def _gaps_per_seed(*, arms: dict) -> list[list[float]]:
        """One list of four arm-ordered gaps per seed -- the paired unit both trend
        statistics are computed inside."""
        per_arm = [
            ResetIntervalReport.gaps(arms=arms, arm=arm)
            for arm in sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
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
        the reset interval, every seed drawn, with a bootstrap 95% CI on the mean,
        and the binomial noise floor drawn as a band. Per-seed points are drawn
        rather than an error bar alone because on this project a large sd has
        repeatedly turned out to be one collapsed seed rather than a broad spread,
        and no mean/CI rendering can tell those apart.

        Right is the check that decides whether the left panel means anything:
        final competence per family per arm. The gap is hump-shaped in training
        progress, so the arms must end at the same competence for a cross-arm gap
        difference to be attributable to reset frequency. Unlike the previous
        experiment this is *designed* to hold (same cycles, same transitions
        everywhere) -- which is exactly why it has to be shown rather than claimed.
        """
        arms_ordered = sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
        fig, (ax, progress_ax) = plt.subplots(1, 2, figsize=(14, 5.5))
        intervals = [_ARM_INTERVAL[arm] for arm in arms_ordered]
        means, lows, highs = [], [], []
        for arm in arms_ordered:
            gaps = ResetIntervalReport.gaps(arms=arms, arm=arm)
            low, high = PairedTests.bootstrap_ci(values=gaps)
            means.append(statistics.mean(gaps))
            lows.append(low)
            highs.append(high)
            # Fanned rather than stacked on one x: the gap can only land on
            # multiples of ~7pp, so several seeds routinely share a value exactly
            # and a single column would render 20 seeds as a handful of dots.
            offsets = [1.10 * 1.020**index for index in range(len(gaps))]
            ax.scatter(
                [_ARM_INTERVAL[arm] * offset for offset in offsets],
                sorted(gaps),
                s=18,
                color="#0072B2",
                alpha=0.45,
                linewidths=0,
                zorder=2,
                label="individual seeds" if arm == arms_ordered[0] else None,
            )
        floor = ResetIntervalReport.predicted_gap_noise(arms=arms, arm=arms_ordered[0])
        # The band a *null* would still wander inside. Without it, four point
        # estimates a few points apart read as a trend.
        ax.axhspan(
            -floor / len(ResetIntervalReport.seeds(arms=arms)) ** 0.5,
            floor / len(ResetIntervalReport.seeds(arms=arms)) ** 0.5,
            color="dimgrey",
            alpha=0.10,
            zorder=0,
            label="+-1 s.e. of the binomial noise floor",
        )
        ax.axhline(0.0, color="dimgrey", linestyle=":", linewidth=1.2, zorder=1)
        ax.errorbar(
            intervals,
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
        # Below the marker, clear of both the CI whiskers and the per-seed fan to
        # the right -- and never to the *left*, which clipped the leftmost arm's
        # value against the y-axis.
        for interval, mean in zip(intervals, means, strict=True):
            ax.annotate(
                f"{mean:+.1f}",
                xy=(interval, mean),
                xytext=(-6, -16),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#333333",
            )
        baseline = _ARM_INTERVAL[_BASELINE_ARM]
        ax.axvline(baseline, color="dimgrey", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.annotate(
            "old behaviour",
            xy=(baseline, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="dimgrey",
        )
        ax.set_xscale("log")
        ax.set_xticks(intervals)
        ax.set_xticklabels([
            f"{interval}\n({ResetIntervalReport.expected_resets(arm=arm)} resets)"
            for arm, interval in zip(arms_ordered, intervals, strict=True)
        ])
        ax.set_xlabel("Steps between free resets (log scale) -- rarer rescues to the right")
        ax.set_ylabel("TRASH - RECYCLING final success gap (pp)")
        ax.set_title(
            "The designed comparison: final gap vs reset interval\n"
            "(25 refits and 2500 transitions in every arm)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # The progress-match check, drawn beside the headline rather than buried in
        # the log: if these bars are not level across arms, the gap comparison is
        # confounded exactly as the previous experiment's was.
        width = 0.26
        positions = np.arange(len(arms_ordered))
        for index, (family, (color, _marker)) in enumerate(_FAMILY_STYLE.items()):
            values = [
                statistics.mean(ResetIntervalReport.final_rates(arms=arms, arm=arm, family=family))
                for arm in arms_ordered
            ]
            errors = [
                statistics.stdev(ResetIntervalReport.final_rates(arms=arms, arm=arm, family=family))
                / len(ResetIntervalReport.seeds(arms=arms)) ** 0.5
                for arm in arms_ordered
            ]
            progress_ax.bar(
                positions + (index - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=color,
                label=family,
            )
        progress_ax.set_xticks(positions)
        progress_ax.set_xticklabels([
            f"{_ARM_INTERVAL[arm]}\n{arm}" + ("\n(old behaviour)" if arm == _BASELINE_ARM else "")
            for arm in arms_ordered
        ])
        progress_ax.set_ylim(0, 105)
        progress_ax.set_xlabel("Steps between free resets")
        progress_ax.set_ylabel("% final evaluation tasks solved")
        progress_ax.set_title(
            "The precondition: are the arms equally trained?\n(bars = mean over seeds, 1 s.e.)"
        )
        progress_ax.grid(True, axis="y", alpha=0.3)
        progress_ax.legend(loc="lower right", fontsize=9)

        fig.suptitle(
            "Tossing Room: does rescuing the robot more often reduce the price of an "
            f"irreversible action? {len(ResetIntervalReport.seeds(arms=arms))} paired seeds"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)

    @staticmethod
    def render_curve_figure(*, arms: dict, output: Path) -> None:
        """Per-family learning curves, one panel per arm, mean +- standard error.

        One panel per arm rather than one shared panel, matching the previous
        experiment's figure -- but here the panels ARE comparable, since every arm
        records the same 26 sweeps over the same 2500 transitions with the same 25
        refits. That comparability is the point: the panels should differ only in
        how often the robot was rescued.
        """
        arms_ordered = sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
        fig, axes = plt.subplots(1, len(arms_ordered), figsize=(16, 4.4), sharey=True)
        for ax, arm in zip(axes, arms_ordered, strict=True):
            for family, (color, marker) in _FAMILY_STYLE.items():
                curve = ResetIntervalReport.family_curve(arms=arms, arm=arm, family=family)
                xs = [point[0] for point in curve]
                means = [point[1] for point in curve]
                errs = [point[2] for point in curve]
                ax.plot(
                    xs,
                    means,
                    color=color,
                    linewidth=2.0,
                    marker=marker,
                    markersize=4,
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
                f"reset every {_ARM_INTERVAL[arm]} steps "
                f"({ResetIntervalReport.expected_resets(arm=arm)} total)"
                + ("\n(old behaviour)" if arm == _BASELINE_ARM else "")
            )
            ax.set_xlabel("Online transitions")
            ax.set_ylim(-3, 103)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("% evaluation tasks solved")
        axes[0].legend(loc="lower right", fontsize=9)
        fig.suptitle(
            "Tossing Room per-family learning curves, mean +- standard error over "
            f"{len(ResetIntervalReport.seeds(arms=arms))} seeds "
            "(panels ARE comparable: same cycles, transitions and refits in every arm)"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


class PairedTests(BaseModel):
    """Exact paired tests over the shared seeds, plus the power question every
    non-significant result on this project has to answer.

    Exact by enumeration rather than approximate, which sidesteps the normal
    approximation (badly behaved at these n), the continuity correction and the tie
    correction all at once -- and needs no special functions, hence no scipy, which
    is not a dependency of this project.

    At n = 20 the sign-flip null has 2**20 = 1,048,576 members, so the enumeration
    is done meet-in-the-middle (`_subset_sums`) rather than by iterating
    `itertools.product`: the terms are split in half, each half's 2**10 subset sums
    are enumerated, and the two are added by numpy broadcasting. Identical answer,
    seconds instead of minutes.
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
        sums = PairedTests._subset_sums(weights=ranks)
        extreme = int(np.count_nonzero(np.abs(sums - total / 2) >= distance - 1e-9))
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

        Flipping the sign of a subset S is the same as subtracting twice that
        subset's sum from the total, so this reuses `_subset_sums` rather than
        enumerating +-1 vectors.
        """
        num_zero = sum(1 for d in differences if d == 0.0)
        if all(d == 0.0 for d in differences):
            return PairedTests(statistic=0.0, p_value=1.0, num_zero_differences=num_zero)
        observed = abs(sum(differences))
        total = sum(differences)
        sums = PairedTests._subset_sums(weights=differences)
        extreme = int(np.count_nonzero(np.abs(total - 2.0 * sums) >= observed - 1e-9))
        return PairedTests(
            statistic=statistics.mean(differences),
            p_value=extreme / 2 ** len(differences),
            num_zero_differences=num_zero,
        )

    @staticmethod
    def _subset_sums(*, weights: list[float]) -> np.ndarray:
        """Every one of the 2**len(weights) subset sums, meet-in-the-middle.

        Enumerating directly costs 2**n * n Python operations, which at n = 20 is
        ~20 million and takes minutes per test. Splitting the weights in half and
        broadcasting the two halves' subset sums against each other costs
        2**(n/2) work plus one 2**n-element numpy add -- the same exhaustive
        enumeration, done in seconds. Exactness is preserved: every subset appears
        exactly once, as (a subset of the left half) + (a subset of the right half).
        """
        half = len(weights) // 2
        left = np.array([
            sum(combo) for combo in itertools.product(*[(0.0, w) for w in weights[:half]])
        ])
        right = np.array([
            sum(combo) for combo in itertools.product(*[(0.0, w) for w in weights[half:]])
        ])
        return (left[:, None] + right[None, :]).ravel()

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
    def minimum_detectable_effect(*, differences: list[float]) -> float:
        """The smallest true effect this design had an 80% chance of detecting, at
        the spread actually observed: (z_alpha + z_power) * sd / sqrt(n).

        The companion to seeds_for_80_percent_power, and the more useful number
        when a result is null -- it says what the experiment could have found, so a
        reader can tell "no effect" from "no power".
        """
        if len(differences) < 2:
            return math.inf
        return (_Z_ALPHA + _Z_POWER) * statistics.stdev(differences) / len(differences) ** 0.5

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


def _print_manipulation_checks(*, arms: dict) -> None:
    """Everything that decides whether the numbers below are worth reading, printed
    first and on purpose. The previous reset experiment's lesson was that a design
    which silently did not vary what it claimed reports a clean-looking null."""
    arms_ordered = sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
    seeds = ResetIntervalReport.seeds(arms=arms)
    print(f"Shared seeds ({len(seeds)}): {', '.join(seeds)}\n")

    print("MANIPULATION CHECK -- free resets that actually happened:")
    print(f"{'arm':>5} {'interval':>9} {'expected':>9} {'observed (min..max)':>22} {'status':>8}")
    for arm in arms_ordered:
        counts = ResetIntervalReport.reset_counts(arms=arms, arm=arm)
        expected = ResetIntervalReport.expected_resets(arm=arm)
        ok = all(count == expected for count in counts)
        print(
            f"{arm:>5} {_ARM_INTERVAL[arm]:>9} {expected:>9} "
            f"{f'{min(counts)}..{max(counts)}':>22} {'OK' if ok else 'MISMATCH':>8}"
        )

    print("\nCOMPOSITION CHECK -- test tasks per family per seed (designed 14/14/2):")
    violations = ResetIntervalReport.composition_violations(arms=arms)
    for family, expected in expected_denominators().items():
        counts = set(
            ResetIntervalReport.family_denominators(arms=arms, arm=arms_ordered[0], family=family)
        )
        print(f"  {family:>10}: {sorted(counts)} (expected {expected})")
    print(f"  violations across all arms and seeds: {len(violations)}")
    for violation in violations[:10]:
        print(f"    {violation}")

    print("\nEXPERIENCE CHECK -- online transitions actually taken (design: 2500 everywhere):")
    for arm in arms_ordered:
        achieved = ResetIntervalReport.achieved_transitions(arms=arms, arm=arm)
        print(
            f"  {arm}: mean {statistics.mean(achieved):.0f}, sd {statistics.stdev(achieved):.1f}, "
            f"min {min(achieved):.0f}, max {max(achieved):.0f}"
        )


def _print_report(*, arms: dict) -> None:
    arms_ordered = sorted(_ARM_INTERVAL, key=lambda arm: _ARM_INTERVAL[arm])
    seeds = ResetIntervalReport.seeds(arms=arms)
    _print_manipulation_checks(arms=arms)

    print(f"\n{'arm':>5} {'interval':>9} {'resets':>7}", end="")
    for family in _FAMILY_STYLE:
        print(f" {family:>18}", end="")
    print(f" {'gap (T-R)':>16}")
    for arm in arms_ordered:
        print(
            f"{arm:>5} {_ARM_INTERVAL[arm]:>9} {ResetIntervalReport.expected_resets(arm=arm):>7}",
            end="",
        )
        for family in _FAMILY_STYLE:
            rates = ResetIntervalReport.final_rates(arms=arms, arm=arm, family=family)
            print(f" {statistics.mean(rates):>10.1f} +-{statistics.stdev(rates):<5.1f}", end="")
        gaps = ResetIntervalReport.gaps(arms=arms, arm=arm)
        print(f" {statistics.mean(gaps):>8.1f} +-{statistics.stdev(gaps):<6.1f}")

    print("\nPer-seed (TRASH - RECYCLING) gap, pp:")
    print(f"{'arm':>5} " + " ".join(f"{('s' + seed):>6}" for seed in seeds))
    for arm in arms_ordered:
        gaps = ResetIntervalReport.gaps(arms=arms, arm=arm)
        print(f"{arm:>5} " + " ".join(f"{gap:>6.1f}" for gap in gaps))

    print("\nPer-seed final RECYCLING %, because an sd here is often one collapsed seed:")
    for arm in arms_ordered:
        rates = ResetIntervalReport.final_rates(arms=arms, arm=arm, family="RECYCLING")
        print(f"{arm:>5} " + " ".join(f"{rate:>6.0f}" for rate in rates))

    print(
        f"\nPooled final rate (total solved / total tasks over all {len(seeds)} seeds, "
        "descriptive only):"
    )
    print(f"{'arm':>5} " + " ".join(f"{family:>11}" for family in _FAMILY_STYLE))
    for arm in arms_ordered:
        print(
            f"{arm:>5} "
            + " ".join(
                f"{ResetIntervalReport.pooled_rate(arms=arms, arm=arm, family=family):>10.1f}%"
                for family in _FAMILY_STYLE
            )
        )

    print("\nIs the gap's spread real, or just task-sampling noise?")
    for arm in arms_ordered:
        observed = statistics.stdev(ResetIntervalReport.gaps(arms=arms, arm=arm))
        predicted = ResetIntervalReport.predicted_gap_noise(arms=arms, arm=arm)
        print(
            f"  {arm}: observed gap sd {observed:5.1f}pp vs {predicted:5.1f}pp predicted from "
            f"binomial task sampling alone ({100 * observed / predicted:.0f}% of the noise floor)"
        )

    # The progress-match check, printed BEFORE the trend tests because it decides
    # whether they mean anything. Unlike the previous experiment this is designed to
    # pass -- every arm gets 25 refits over 2500 transitions -- so a failure here
    # means the confound reappeared through another door and the gap is
    # uninterpretable regardless of its p-value.
    print(
        "\nPROGRESS MATCH -- are the arms equally trained? "
        f"(paired, {_BASELINE_ARM} minus armA, exact tests)"
    )
    for family in _FAMILY_STYLE:
        differences = ResetIntervalReport.family_differences(
            arms=arms, family=family, from_arm="armA", to_arm=_BASELINE_ARM
        )
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
        flip = PairedTests.sign_flip(differences=differences)
        mde = PairedTests.minimum_detectable_effect(differences=differences)
        print(
            f"  {family:>10}: mean {statistics.mean(differences):+6.1f}pp, "
            f"sd {statistics.stdev(differences):5.1f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f}, sign-flip p={flip.p_value:.4f}, "
            f"MDE at n={len(differences)}: {mde:.1f}pp"
        )

    print(f"\nPaired tests (exact, {len(seeds)} shared seeds):")
    extremes = [
        d - a
        for a, d in zip(
            ResetIntervalReport.gaps(arms=arms, arm="armA"),
            ResetIntervalReport.gaps(arms=arms, arm=_BASELINE_ARM),
            strict=True,
        )
    ]
    for label, differences in (
        ("gap: armD (100) - armA (10)", extremes),
        (
            "per-seed trend slope (pp per doubling of interval)",
            ResetIntervalReport.trend_slopes(arms=arms),
        ),
        (
            "per-seed Spearman rho of gap vs interval",
            ResetIntervalReport.trend_rank_correlations(arms=arms),
        ),
    ):
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
        flip = PairedTests.sign_flip(differences=differences)
        needed = PairedTests.seeds_for_80_percent_power(differences=differences)
        power = f"{needed:.0f}" if math.isfinite(needed) else "infinite (zero observed effect)"
        mde = PairedTests.minimum_detectable_effect(differences=differences)
        print(
            f"  {label}: mean {statistics.mean(differences):+.2f}, "
            f"sd {statistics.stdev(differences):.2f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f} ({wilcoxon.num_zero_differences} zero diffs), "
            f"sign-flip p={flip.p_value:.4f}, n for 80% power: {power}, MDE: {mde:.2f}"
        )

    print("\nWithin-arm gap vs zero (is RECYCLING behind TRASH at all?):")
    for arm in arms_ordered:
        gaps = ResetIntervalReport.gaps(arms=arms, arm=arm)
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=gaps)
        flip = PairedTests.sign_flip(differences=gaps)
        print(
            f"  {arm}: mean {statistics.mean(gaps):+.1f}pp, Wilcoxon p={wilcoxon.p_value:.4f}, "
            f"sign-flip p={flip.p_value:.4f}, {wilcoxon.num_zero_differences} zero diffs"
        )

    # Everything below is POST-HOC -- added after the pre-specified final-checkpoint
    # metric returned a null measured at a ceiling. Labelled rather than folded in
    # with the tests above, because a statistic chosen after seeing the curves does
    # not carry the same evidential weight as one chosen before.
    print("\n--- POST-HOC: the trajectory, not the endpoint ---")
    print("\nMean success over all checkpoints (normalised area under the learning curve):")
    print(f"{'arm':>5} " + " ".join(f"{family:>16}" for family in _FAMILY_STYLE))
    for arm in arms_ordered:
        print(f"{arm:>5} ", end="")
        for family in _FAMILY_STYLE:
            values = ResetIntervalReport.mean_rate_over_training(arms=arms, arm=arm, family=family)
            print(f" {statistics.mean(values):>9.1f} +-{statistics.stdev(values):<4.1f}", end="")
        print()

    print(f"\nLearning speed, paired armA (10) minus {_BASELINE_ARM} (100), exact tests:")
    speedups = {}
    for family in _FAMILY_STYLE:
        differences = [
            a - d
            for a, d in zip(
                ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armA", family=family),
                ResetIntervalReport.mean_rate_over_training(
                    arms=arms, arm=_BASELINE_ARM, family=family
                ),
                strict=True,
            )
        ]
        speedups[family] = differences
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differences)
        flip = PairedTests.sign_flip(differences=differences)
        print(
            f"  {family:>10}: mean {statistics.mean(differences):+6.1f}pp, "
            f"sd {statistics.stdev(differences):5.1f}, "
            f"Wilcoxon p={wilcoxon.p_value:.4f}, sign-flip p={flip.p_value:.4f}"
        )

    # The discriminating question. Frequent resets speeding BOTH throw families up
    # equally is a general "less time wasted" effect; irreversibility predicts the
    # terminal family benefits MORE.
    differential = [r - t for r, t in zip(speedups["RECYCLING"], speedups["TRASH"], strict=True)]
    wilcoxon = PairedTests.wilcoxon_signed_rank(differences=differential)
    flip = PairedTests.sign_flip(differences=differential)
    mde = PairedTests.minimum_detectable_effect(differences=differential)
    needed = PairedTests.seeds_for_80_percent_power(differences=differential)
    print(
        "\n  Does RECYCLING gain MORE than TRASH from frequent resets? "
        "(the irreversibility-specific claim; a generic 'less wasted motion' effect "
        "would speed both up equally)\n"
        f"    RECYCLING speedup minus TRASH speedup: mean {statistics.mean(differential):+.1f}pp, "
        f"sd {statistics.stdev(differential):.1f}, Wilcoxon p={wilcoxon.p_value:.4f}, "
        f"sign-flip p={flip.p_value:.4f}, MDE at n={len(differential)}: {mde:.1f}pp, "
        f"n for 80% power: {needed:.0f}"
    )

    print("\nMean (TRASH - RECYCLING) gap over all checkpoints, per arm:")
    for arm in arms_ordered:
        values = ResetIntervalReport.mean_gap_over_training(arms=arms, arm=arm)
        wilcoxon = PairedTests.wilcoxon_signed_rank(differences=values)
        print(
            f"  {arm}: mean {statistics.mean(values):+5.1f}pp, "
            f"sd {statistics.stdev(values):4.1f}, vs zero Wilcoxon p={wilcoxon.p_value:.4f}"
        )
    mid_gap_extremes = [
        d - a
        for a, d in zip(
            ResetIntervalReport.mean_gap_over_training(arms=arms, arm="armA"),
            ResetIntervalReport.mean_gap_over_training(arms=arms, arm=_BASELINE_ARM),
            strict=True,
        )
    ]
    wilcoxon = PairedTests.wilcoxon_signed_rank(differences=mid_gap_extremes)
    flip = PairedTests.sign_flip(differences=mid_gap_extremes)
    mde = PairedTests.minimum_detectable_effect(differences=mid_gap_extremes)
    print(
        f"  {_BASELINE_ARM} minus armA: mean {statistics.mean(mid_gap_extremes):+.1f}pp, "
        f"sd {statistics.stdev(mid_gap_extremes):.1f}, Wilcoxon p={wilcoxon.p_value:.4f}, "
        f"sign-flip p={flip.p_value:.4f}, MDE at n={len(mid_gap_extremes)}: {mde:.1f}pp"
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
        aggregate = ResetIntervalReport.aggregate(arm_dirs=arm_dirs)
        if args.aggregate_output is None:
            raise ValueError("--arm requires --aggregate-output")
        # Compact, one line, matching 2026-08-03-ballring-arms.json: this is
        # recorded data, not source, and an indented rendering is many times the
        # bytes for no added readability.
        args.aggregate_output.write_text(json.dumps(aggregate, sort_keys=True))
        print(f"wrote {args.aggregate_output}")
        return

    if args.arms_json is None:
        raise ValueError("pass either --arm ... --aggregate-output, or --arms-json")
    arms = ResetIntervalReport.load_arms(json_path=args.arms_json)
    if args.output is not None:
        ResetIntervalReport.render_gap_figure(arms=arms, output=args.output)
        print(f"wrote {args.output}")
    if args.curves_output is not None:
        ResetIntervalReport.render_curve_figure(arms=arms, output=args.curves_output)
        print(f"wrote {args.curves_output}")
    _print_report(arms=arms)


if __name__ == "__main__":
    main()
