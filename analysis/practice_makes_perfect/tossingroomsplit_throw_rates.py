"""Post-run analysis for Tossing Room (split throws): do `ThrowTrash` and
`ThrowRecycling` -- two lifted skills with two independent samplers, same architecture
-- learn at rates matching how much practice the layout affords each of them?

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- never runs a
simulation):

* `--results-root <dir>` -- the `<root>/ees/<seed>/` tree written by
  `scripts/run_sweep.py`. Source of the task-level record: `stats.json`'s
  `(transitions, solved, total)` triples and per-task `breakdowns`.
* `--traces <file>` (repeatable) -- shards written by
  `scripts/tossingroomsplit_skill_traces.py`. Source of the per-SKILL record: attempts,
  successes and competence, which never leave `EesMethod`'s internals and so cannot be
  in `stats.json`.

**The two are the same runs, and that is checked, not assumed.** `check_against_sweep`
compares every traced seed's per-sweep `(transitions, solved, total)` against that
seed's real `stats.json` and refuses to report anything if they disagree or if a traced
seed is missing from the sweep entirely. A gate that silently checks zero seeds passes,
so a missing seed is a disagreement here rather than a skip.

**Everything is counts.** Success is `x/y`, attempts are totals, and the attempt ratio
comes with both of its terms -- 12:1 and 1200:100 are different evidence and a bare
ratio hides which one you have. Only *differences* of two rates (gaps, the noise floor,
the minimum detectable effect) stay in percentage points, which is their correct unit.

The Wilcoxon test is imported from `tossingroom_comparison` rather than reimplemented:
it is exact-by-enumeration, already pinned against hand-computed values in
`tests/analysis/practice_makes_perfect/test_tossingroom_comparison.py`, and a second
copy of a hand-rolled significance test is exactly how a sign error gets published.

**The committed 2026-08-05 traces are the CAPACITY-1 run.** An earlier set, taken before
Tossing Room's capacity-1 bins and per-bin buttons were ported into this domain, has been
withdrawn and replaced rather than re-scored: that port changed the DYNAMICS (a bin holds
at most one item, a throw at a full one is refused, EMPTY prefills one item per bin and
needs both buttons pressed in the one order the ledge permits, and the evaluation horizon
went 7 -> 12), so no number computed from the old traces was comparable to one computed
from these. Do not pool a trace taken before that port with one taken after; they are
measurements of two different worlds.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.tossingroom_comparison import (  # noqa: E402
    TossingRoomComparison,
)

# The two skills under comparison, with fixed colours and linestyles so a figure keeps
# its identity when another panel is added. Linestyle repeats identity as a second
# channel, for readers who cannot separate the hues.
_SKILL_STYLES: dict[str, tuple[str, str]] = {
    "ThrowTrash": ("tab:blue", "-"),
    "ThrowRecycling": ("tab:orange", "--"),
}
# `Goal.describe()` output for each throw family, as it appears in `Metrics.breakdowns`.
_FAMILY_GOALS: dict[str, str] = {
    "ThrowTrash": "TrashInBin(trash, trash_bin)",
    "ThrowRecycling": "RecyclingInBin(recycling, recycling_bin)",
}
# (z_{0.025} + z_{0.20}): the standard two-sided 80%-power constant.
_MDE_CONSTANT = 1.959963985 + 0.841621234
# 3x the domain's throw_tolerance of 0.1. A greedy draw further than this from the force
# its own grounding required is a wrong answer, not an inaccurate one -- see
# `badly_missed_force_totals` for why this replaced a fixed "below 0.4" floor.
_BADLY_MISSED_THRESHOLD = 0.30


class TossingRoomSplitThrowRates:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ------------------------------------------------------------------ the gate

    @staticmethod
    def check_against_sweep(*, traces: list[dict], results_root: Path) -> list[str]:
        """Every traced seed's evaluation record against the real `stats.json` that
        `scripts/run_sweep.py` wrote for the same seed. Returns a list of human-readable
        disagreements, empty when the two agree everywhere.

        A traced seed with no `stats.json` is a disagreement, not a skip: the whole
        point of this function is to license reporting the two sources side by side, and
        a check that quietly examined nothing would license everything."""
        problems: list[str] = []
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            seed = run["seed"]
            stats_path = results_root / "ees" / str(seed) / "stats.json"
            if not stats_path.exists():
                problems.append(
                    f"seed {seed}: traced, but no sweep output at {stats_path} -- the "
                    f"per-skill counts have nothing to be cross-checked against"
                )
                continue
            swept = [tuple(triple) for triple in json.loads(stats_path.read_text())["evaluations"]]
            traced = [
                (sweep["transitions"], sweep["solved"], sweep["total"]) for sweep in run["sweeps"]
            ]
            if traced != swept:
                problems.append(
                    f"seed {seed}: traced evaluations {traced} != swept {swept}; the "
                    f"instrumented run is not the run the sweep measured"
                )
        return problems

    # -------------------------------------------------------------- reading the traces

    @staticmethod
    def runs(*, traces: list[dict]) -> list[dict]:
        """Every seed's run, pooled across shards and sorted by seed. The collector is
        serial within a process, so a full set is collected one process per seed; pooling
        here is what makes that sharding invisible to everything downstream."""
        pooled = [run for trace in traces for run in trace["seeds"]]
        return sorted(pooled, key=lambda run: run["seed"])

    @staticmethod
    def attempt_totals(*, traces: list[dict]) -> dict[str, int]:
        """Total practice attempts per lifted skill, summed over every period of every
        seed. An attempt is an execution EES actually observed the outcome of -- see
        `SkillTally`."""
        totals: dict[str, int] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    totals[name] = totals.get(name, 0) + tally["attempts"]
        return totals

    @staticmethod
    def success_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (successes, attempts)}` -- the per-skill success record as the two
        counts it is, never as the quotient."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    successes, attempts = totals.get(name, (0, 0))
                    totals[name] = (
                        successes + tally["successes"],
                        attempts + tally["attempts"],
                    )
        return totals

    @staticmethod
    def greedy_success_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (successes, attempts)}` over attempts whose parameters came from the
        LEARNED sampler, i.e. with the epsilon-greedy random ones subtracted out.

        This is the one that says what a sampler has learned. At the paper's
        epsilon = 0.5 about half of every practice attempt is a coin flip, so the pooled
        rate partly measures how often a coin flip works -- and EES itself excludes
        random attempts from competence for exactly this reason."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    successes, attempts = totals.get(name, (0, 0))
                    totals[name] = (
                        successes + tally["successes"] - tally.get("random_successes", 0),
                        attempts + tally["attempts"] - tally.get("random_attempts", 0),
                    )
        return totals

    @staticmethod
    def random_success_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """The complement of `greedy_success_totals` -- reported beside it, never pooled
        into it, and useful as a control: a random draw's success rate should not move
        as training proceeds."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    successes, attempts = totals.get(name, (0, 0))
                    totals[name] = (
                        successes + tally.get("random_successes", 0),
                        attempts + tally.get("random_attempts", 0),
                    )
        return totals

    @staticmethod
    def _sum_pair(*, traces: list[dict], numerator: str) -> dict[str, tuple[int, int]]:
        """`{skill: (sum of <numerator>, sum of attempts)}` over every period of every
        seed -- the shape every count in this file is reported in."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    top, bottom = totals.get(name, (0, 0))
                    totals[name] = (top + tally.get(numerator, 0), bottom + tally["attempts"])
        return totals

    @staticmethod
    def landing_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (landings, attempts)}` -- what the DYNAMICS did, as opposed to what
        EES scored. See `SkillTally` in the collector for why the two differ."""
        return TossingRoomSplitThrowRates._sum_pair(traces=traces, numerator="landed")

    @staticmethod
    def prefilled_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (attempts made into an already-non-empty bin, attempts)}`.

        On the traces this file was written against, every one of those was scored a
        success before its force was chosen. **That channel is closed on the current
        domain** -- a bin holds at most one item, each throw carries its bin's empty
        precondition, and the dynamics refuse a throw at a full bin -- so on any run made
        after that redesign this should be 0 for both skills, and a nonzero value is a
        regression to report rather than a quantity to interpret."""
        return TossingRoomSplitThrowRates._sum_pair(traces=traces, numerator="prefilled")

    @staticmethod
    def inflated_successes(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (scored successes that did not land, scored successes)}` -- how much
        of a skill's apparent success record is an artifact of the add-effect check."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    spurious, scored = totals.get(name, (0, 0))
                    totals[name] = (
                        spurious + tally["successes"] - tally.get("landed", 0),
                        scored + tally["successes"],
                    )
        return totals

    @staticmethod
    def attempts_per_period_histogram(*, traces: list[dict], skill: str) -> dict[int, int]:
        """`{attempts in a period: how many periods had that many}`, pooled over seeds.

        The distribution, not the mean, is what carries this domain's structural claim: a
        skill averaging 0.8 attempts per period could be 0-or-1 every single time or 0
        four times and 4 once, and only the first says "the layout allows one attempt".
        A period with no record for the skill is a real zero and is counted."""
        histogram: dict[int, int] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                attempts = period["skills"].get(skill, {}).get("attempts", 0)
                histogram[attempts] = histogram.get(attempts, 0) + 1
        return dict(sorted(histogram.items()))

    @staticmethod
    def transitions_to_reach(*, traces: list[dict], skill: str, level: float) -> int | None:
        """Transitions at the first sweep where this family's pooled success rate reaches
        `level`, or None if it never does.

        None rather than the final sweep: "never got there" and "got there at the very
        end" are different findings, and collapsing them flatters the slower family."""
        for transitions, solved, total in TossingRoomSplitThrowRates.family_counts(
            traces=traces, skill=skill
        ):
            if total and solved / total >= level:
                return transitions
        return None

    @staticmethod
    def area_under_curve(*, traces: list[dict], skill: str) -> list[float]:
        """One value per seed: the mean of that seed's own per-sweep success rates for
        this family.

        A mean rather than a trapezoid sum because sweeps are evenly spaced in
        transitions, so the two differ only by a constant -- and the mean stays on the
        0-1 scale a rate belongs on. Per seed, because the two families are measured on
        the *same* seed and the comparison is therefore paired."""
        values: list[float] = []
        goal = _FAMILY_GOALS[skill]
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            rates = []
            for sweep in run["sweeps"]:
                solved, total = sweep["families"].get(goal, (0, 0))
                rates.append(solved / total if total else 0.0)
            values.append(statistics.mean(rates) if rates else 0.0)
        return values

    @staticmethod
    def per_seed_attempts(*, traces: list[dict]) -> dict[int, dict[str, int]]:
        """One seed's totals per skill. A pooled ratio can be carried by a single run;
        this is what shows whether the effect is structural."""
        per_seed: dict[int, dict[str, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            counts: dict[str, int] = {}
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    counts[name] = counts.get(name, 0) + tally["attempts"]
            per_seed[run["seed"]] = counts
        return per_seed

    @staticmethod
    def attempt_ratio(*, traces: list[dict]) -> float | None:
        """`ThrowTrash` attempts per `ThrowRecycling` attempt, or None when the recycling
        throw was never attempted at all.

        None rather than `inf`: a zero denominator is a genuinely possible outcome of
        this layout (the one-way ledge can strand the robot before it ever throws), and
        it is a fact to report rather than a number to print."""
        totals = TossingRoomSplitThrowRates.attempt_totals(traces=traces)
        recycling = totals.get("ThrowRecycling", 0)
        if recycling == 0:
            return None
        return totals.get("ThrowTrash", 0) / recycling

    # --------------------------------------------------- is the curve a curve at all

    @staticmethod
    def seed_checkpoint_extremes(
        *, traces: list[dict], skill: str, high: int, low: int
    ) -> dict[str, int]:
        """How many (seed, checkpoint) pairs sit at an extreme of this family's score, and
        how many sit anywhere in between.

        A pooled mean-over-seeds curve that rises smoothly is consistent with two entirely
        different worlds: seeds that improve gradually, and seeds that sit at 0/14 until
        they snap to 14/14 at a seed-specific moment. The mean cannot separate them, and
        describing the second as "still climbing" would be an artifact of the averaging.
        Counting how rarely a seed is observed *between* the extremes is what separates
        them, so this is a count over seed-checkpoints rather than a property of a line.

        Both thresholds are inclusive and are counts, not rates -- `high=12, low=4` on a
        14-task family. A seed-checkpoint at exactly 12/14 is at the top extreme."""
        counts = {"extreme": 0, "middle": 0, "total": 0}
        goal = _FAMILY_GOALS[skill]
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for sweep in run["sweeps"]:
                solved, total = sweep["families"].get(goal, (0, 0))
                if not total:
                    continue
                counts["total"] += 1
                counts["extreme" if solved >= high or solved <= low else "middle"] += 1
        return counts

    @staticmethod
    def per_seed_family_peaks(*, traces: list[dict], skill: str) -> dict[int, dict[str, int]]:
        """`{seed: {peak, final, total}}` in counts, for one goal family.

        The peak is carried next to the endpoint because a seed can reach a high score and
        fall back out of it, and a table of final scores alone reports that seed as one
        that never learned. `total` travels with both so neither is ever read as a bare
        number."""
        goal = _FAMILY_GOALS[skill]
        peaks: dict[int, dict[str, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            scores = [sweep["families"].get(goal, (0, 0)) for sweep in run["sweeps"]]
            solved = [count for count, _total in scores]
            totals = {total for _count, total in scores if total}
            peaks[run["seed"]] = {
                "peak": max(solved) if solved else 0,
                "final": solved[-1] if solved else 0,
                "total": max(totals) if totals else 0,
            }
        return peaks

    # ------------------------------------------- what the sampler actually answered

    @staticmethod
    def format_p_value(*, p: float) -> str:
        """`p = 0.0000` claims a p-value of zero, which no test returns. Anything below
        the printed resolution is reported as an inequality instead."""
        return "< 0.0001" if p < 0.0001 else f"{p:.4f}"

    @staticmethod
    def random_landing_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (landings, attempts)}` over the epsilon-random draws.

        This is the control the informed rate has to beat to mean anything: a learned
        sampler landing at its own coin-flip rate has learned nothing, and no pooled
        greedy number can say that. It is a separate method rather than an inline sum
        inside the figure because it is the decisive comparison on the page, and every
        other statistic here goes through a tested one."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    landed, attempts = totals.get(name, (0, 0))
                    totals[name] = (
                        landed + tally.get("landed_random", 0),
                        attempts + tally.get("random_attempts", 0),
                    )
        return totals

    @staticmethod
    def informed_landing_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (landings, attempts)}` over the greedy draws a classifier that
        actually *discriminated* among the candidates made.

        This is `greedy_success_totals` with the contamination removed.
        `LearnedSkillSampler.sample` returns a uniform draw whenever its scores fail to
        rank the candidates -- unfitted, either single-class shortcut, or a saturated
        plateau -- and reports `was_random=False` for it, because `was_random` means
        specifically "the epsilon-greedy branch fired". Those draws are therefore greedy
        by the old split while carrying no belief at all, and the skill with fewer
        observations spends longer in that state, which is exactly the asymmetry this
        experiment measures. Reported beside the greedy column, never instead of it."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    landed, attempts = totals.get(name, (0, 0))
                    totals[name] = (
                        landed + tally.get("informed_landed", 0),
                        attempts + tally.get("informed_attempts", 0),
                    )
        return totals

    @staticmethod
    def uninformed_greedy_totals(*, traces: list[dict]) -> dict[str, tuple[int, int]]:
        """`{skill: (uninformed greedy draws, greedy draws)}` -- how much of the old
        "learned-sampler" pool was the uniform fallback rather than a learned choice.

        On a shard collected before the `was_informed` instrumentation this reports
        every greedy draw as uninformed, because absent is indistinguishable from zero
        here. That is why `print_informed_split` gates on `informed_landing_totals`
        first: read this number on its own from a legacy shard and it says 100%
        fallback, which is a statement about the instrumentation, not the sampler."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    greedy = tally["attempts"] - tally.get("random_attempts", 0)
                    uninformed, drawn = totals.get(name, (0, 0))
                    totals[name] = (
                        uninformed + greedy - tally.get("informed_attempts", 0),
                        drawn + greedy,
                    )
        return totals

    @staticmethod
    def per_seed_informed_draws(*, traces: list[dict], skill: str) -> dict[int, tuple[int, int]]:
        """`{seed: (informed draws, greedy draws)}` for one skill.

        The pooled count hides that the recycling sampler's whole budget is single
        digits per run, so the per-seed split is what says whether the pooled number is
        a central value or one seed's."""
        per_seed: dict[int, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            informed = 0
            greedy = 0
            for period in run["periods"]:
                tally = period["skills"].get(skill)
                if tally is None:
                    continue
                greedy += tally["attempts"] - tally.get("random_attempts", 0)
                informed += tally.get("informed_attempts", 0)
            per_seed[run["seed"]] = (informed, greedy)
        return per_seed

    @staticmethod
    def informed_badly_missed_force_totals(
        *, traces: list[dict], miss_threshold: float
    ) -> dict[str, tuple[int, int]]:
        """`badly_missed_force_totals` restricted to the draws a discriminating
        classifier made -- "did what it *learned* point somewhere wrong", rather than
        "did the pool that includes its uniform fallback point somewhere wrong"."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    forces = tally.get("informed_forces")
                    targets = tally.get("informed_targets")
                    if forces is None or targets is None:
                        continue
                    missed, drawn = totals.get(name, (0, 0))
                    totals[name] = (
                        missed
                        + sum(
                            1
                            for force, target in zip(forces, targets, strict=True)
                            if abs(force - target) > miss_threshold
                        ),
                        drawn + len(forces),
                    )
        return totals

    @staticmethod
    def print_informed_split(*, traces: list[dict]) -> None:
        """Results (4) and (6) with the greedy pool separated into the draws a
        discriminating classifier made and the uniform fallback."""
        informed = TossingRoomSplitThrowRates.informed_landing_totals(traces=traces)
        if not any(attempts for _landed, attempts in informed.values()):
            print(
                "\nNo informed/uninformed split recorded -- these traces predate the "
                "was_informed instrumentation."
            )
            return
        uninformed = TossingRoomSplitThrowRates.uninformed_greedy_totals(traces=traces)
        missed = TossingRoomSplitThrowRates.informed_badly_missed_force_totals(
            traces=traces, miss_threshold=_BADLY_MISSED_THRESHOLD
        )
        print("\nGreedy draws split by whether the classifier discriminated")
        header = (
            f"{'skill':>18}{'uninformed/greedy':>20}{'landed/informed':>18}"
            f"{'missed >0.30/informed':>24}"
        )
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            u_count, u_total = uninformed.get(skill, (0, 0))
            i_landed, i_total = informed.get(skill, (0, 0))
            m_count, m_total = missed.get(skill, (0, 0))
            print(
                f"{skill:>18}{f'{u_count}/{u_total}':>20}"
                f"{f'{i_landed}/{i_total}':>18}{f'{m_count}/{m_total}':>24}"
            )
        trash = informed.get("ThrowTrash", (0, 0))
        recyc = informed.get("ThrowRecycling", (0, 0))
        if trash[1] and recyc[1]:
            gap = trash[0] / trash[1] - recyc[0] / recyc[1]
            mde = TossingRoomSplitThrowRates.minimum_detectable_effect(
                n_first=trash[1], n_second=recyc[1]
            )
            p = TossingRoomComparison.fisher_exact_two_sided(
                a=trash[0], b=trash[1] - trash[0], c=recyc[0], d=recyc[1] - recyc[0]
            )
            print(f"  informed gap (TRASH - RECYCLING) {100 * gap:+.2f}pp")
            print(f"  minimum detectable effect (80%)  {100 * mde:.2f}pp")
            shown = TossingRoomSplitThrowRates.format_p_value(p=p)
            print(f"  Fisher exact (pooled 2x2)        p = {shown}")

        # The decisive comparison: each skill's learned draws against its OWN coin flip.
        # A sampler that cannot beat this has learned nothing, however good the
        # cross-skill gap looks.
        random_draws = TossingRoomSplitThrowRates.random_landing_totals(traces=traces)
        print("\n  each skill's informed draws against its own epsilon-random control")
        # Every row carries the MDE for ITS OWN two denominators. Without that column the
        # only MDE on the page belonged to a neighbouring comparison, and PR #90's log
        # duly borrowed it: it quoted 20.19pp (the floor for trash-random vs
        # recycling-random, 310 vs 57) for the recycling-informed vs recycling-random null
        # result, whose real denominators are 56 vs 57 and whose MDE is 26.36pp. A null
        # result is only as strong as its own floor, so the floor travels with the row.
        header = (
            f"{'skill':>18}{'landed/informed':>18}{'landed/random':>16}{'gap':>10}"
            f"{'noise floor':>14}{'MDE (80%)':>12}{'Fisher p':>12}"
        )
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            i_landed, i_total = informed.get(skill, (0, 0))
            r_landed, r_total = random_draws.get(skill, (0, 0))
            if not (i_total and r_total):
                continue
            gap = i_landed / i_total - r_landed / r_total
            floor = TossingRoomSplitThrowRates.noise_floor(n_first=i_total, n_second=r_total)
            row_mde = TossingRoomSplitThrowRates.minimum_detectable_effect(
                n_first=i_total, n_second=r_total
            )
            p = TossingRoomComparison.fisher_exact_two_sided(
                a=i_landed, b=i_total - i_landed, c=r_landed, d=r_total - r_landed
            )
            print(
                f"{skill:>18}{f'{i_landed}/{i_total}':>18}{f'{r_landed}/{r_total}':>16}"
                f"{f'{100 * gap:+.2f}pp':>10}{f'{100 * floor:.2f}pp':>14}"
                f"{f'{100 * row_mde:.2f}pp':>12}"
                f"{TossingRoomSplitThrowRates.format_p_value(p=p):>12}"
            )
        print("\n  per seed, informed/greedy draws")
        for skill in _SKILL_STYLES:
            per_seed = TossingRoomSplitThrowRates.per_seed_informed_draws(
                traces=traces, skill=skill
            )
            shown = ", ".join(f"{per_seed[s][0]}/{per_seed[s][1]}" for s in sorted(per_seed))
            print(f"{skill:>18}  {shown}")

    @staticmethod
    def badly_missed_force_totals(
        *, traces: list[dict], miss_threshold: float
    ) -> dict[str, tuple[int, int]]:
        """`{skill: (greedy draws missing by more than `miss_threshold`, greedy draws)}`.

        The question this answers is "did the sampler converge on a *wrong* answer rather
        than merely an inaccurate one" -- the shape a stuck sampler has, and what
        distinguishes it from one still hunting.

        **This replaced a simpler count that the throw-representation change made
        vacuous.** When a task's required force was drawn `U(0.5, 1.0)` at tolerance 0.1,
        any force below `0.5 - 0.1 = 0.4` missed *whatever* task it was aiming at, so "a
        force no task in this domain could have wanted" was countable from the choice
        alone. The required force is now an unobserved function of the bin's
        `throw_distance` and the item's `weight` and spans `[0.1, 0.9]`, so with tolerance
        0.1 **every** force in the `U(0, 1)` draw range is right for *some* task and that
        count is identically 0. The equivalent statement has to be made per grounding
        instead, against the force that grounding actually required -- which the collector
        already records alongside the force chosen, so this needs no new instrumentation.

        Greedy draws only. An epsilon-random force is a coin flip and carries no belief;
        the collector does not record them for this reason."""
        totals: dict[str, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for period in run["periods"]:
                for name, tally in period["skills"].items():
                    forces = tally.get("greedy_forces")
                    targets = tally.get("greedy_targets")
                    if forces is None or targets is None:
                        continue
                    missed, drawn = totals.get(name, (0, 0))
                    totals[name] = (
                        missed
                        + sum(
                            1
                            for force, target in zip(forces, targets, strict=True)
                            if abs(force - target) > miss_threshold
                        ),
                        drawn + len(forces),
                    )
        return totals

    @staticmethod
    def longest_missing_streaks(*, traces: list[dict], skill: str) -> dict[int, int]:
        """`{seed: longest run of consecutive practice periods in which every greedy
        attempt of this skill missed}`.

        "Stuck" is a claim about consecutive periods, not a pooled rate: the same number
        of misses spread evenly is a sampler still searching, and concentrated is a
        sampler that has stopped moving. A period in which the skill was never practiced
        greedily is evidence of neither, so it neither extends nor breaks a streak -- and
        for a skill the layout affords one attempt per period, those periods are common.

        `landed` is the miss test rather than `successes`, for the same reason
        `greedy_landing_curve` uses it: what the dynamics did, not what EES scored."""
        streaks: dict[int, int] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            longest = 0
            current = 0
            for period in run["periods"]:
                tally = period["skills"].get(skill)
                if tally is None:
                    continue
                greedy = tally["attempts"] - tally.get("random_attempts", 0)
                if greedy == 0:
                    continue
                landed = tally.get("landed", 0) - tally.get("landed_random", 0)
                current = current + 1 if landed == 0 else 0
                longest = max(longest, current)
            streaks[run["seed"]] = longest
        return streaks

    # ------------------------------------------------------------------- the curves

    @staticmethod
    def _series(*, traces: list[dict], per_run) -> list[tuple[int, float, float]]:
        """`(transitions, mean over seeds, standard error)` per evaluation sweep.

        The x-axis is read off each sweep's own recorded `transitions`, never off the
        period index: a period that ends early (`InteractionComplete`) contributes fewer
        than `max_steps_per_interaction` steps, so counting periods would mis-scale the
        axis. Sweep *i* is credited with everything practiced up to and including period
        *i - 1*, because sweep 0 happens before any practice."""
        by_transitions: dict[int, list[float]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for index, sweep in enumerate(run["sweeps"]):
                by_transitions.setdefault(int(sweep["transitions"]), []).append(
                    per_run(run=run, index=index)
                )
        series: list[tuple[int, float, float]] = []
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            series.append((transitions, statistics.mean(values), stderr))
        return series

    @staticmethod
    def cumulative_attempts(*, traces: list[dict], skill: str) -> list[tuple[int, float, float]]:
        """Attempts of one skill accumulated up to each evaluation sweep.

        A period in which the skill never came up contributes a real zero, not a missing
        value -- dropping it would remove exactly the periods that make recycling's
        curve flat, which is the finding."""

        def total(*, run: dict, index: int) -> float:
            return float(
                sum(
                    period["skills"].get(skill, {}).get("attempts", 0)
                    for period in run["periods"][:index]
                )
            )

        # index 0 is the pre-practice sweep, so it sums an empty prefix -- but sweeps and
        # periods are off by one, and crediting period i-1 to sweep i is what lines the
        # two up. Shift by one so sweep 1 carries period 0.
        def shifted(*, run: dict, index: int) -> float:
            if index + 1 <= len(run["periods"]):
                return total(run=run, index=index + 1)
            return total(run=run, index=len(run["periods"]))

        return TossingRoomSplitThrowRates._series(traces=traces, per_run=shifted)

    @staticmethod
    def competence_curve(*, traces: list[dict], skill: str) -> list[tuple[int, float, float]]:
        """One skill's competence at each evaluation sweep. Sweep 0 predates any
        practice, so there is no competence model yet and the value is the Beta(10, 1)
        prior mean EES itself would price an unexecuted skill at."""
        prior = 10.0 / 11.0

        def value(*, run: dict, index: int) -> float:
            if index == 0 or index - 1 >= len(run["competence"]):
                return prior
            record = run["competence"][index - 1].get(skill)
            return prior if record is None else float(record["competence"])

        return TossingRoomSplitThrowRates._series(traces=traces, per_run=value)

    @staticmethod
    def greedy_success_curve(*, traces: list[dict], skill: str) -> list[tuple[int, float, float]]:
        """Cumulative fraction of that skill's LEARNED-sampler practice attempts that
        succeeded, at each evaluation sweep.

        This is the direct "is this sampler getting better" signal, and it is the one
        `competence_curve` cannot give: competence is a windowed estimate under a
        Beta(10, 1) prior, so a skill with few observations stays pinned near the prior's
        0.909 no matter what it can actually do -- which is exactly the situation the
        throw with one attempt per period is in. A sweep before any practice has no
        attempts to average and reports 0.0."""

        def value(*, run: dict, index: int) -> float:
            successes = 0
            attempts = 0
            for period in run["periods"][:index]:
                tally = period["skills"].get(skill)
                if tally is None:
                    continue
                successes += tally["successes"] - tally.get("random_successes", 0)
                attempts += tally["attempts"] - tally.get("random_attempts", 0)
            return successes / attempts if attempts else 0.0

        return TossingRoomSplitThrowRates._series(traces=traces, per_run=value)

    @staticmethod
    def greedy_landing_curve(*, traces: list[dict], skill: str) -> list[tuple[int, float, float]]:
        """The honest per-attempt learning curve: cumulative fraction of LEARNED-sampler
        attempts that actually landed in the bin.

        Tracks `landed` rather than `successes` on purpose. A throw into an already-non-
        empty bin is scored a success at any force (see the collector's `SkillTally`), and
        that happens to the trash throw constantly and to the recycling throw never -- so
        a curve built on scored successes reproduces exactly the artifact the audit
        exists to expose, and does so asymmetrically between the two skills being
        compared."""

        def value(*, run: dict, index: int) -> float:
            landings = 0
            attempts = 0
            for period in run["periods"][:index]:
                tally = period["skills"].get(skill)
                if tally is None:
                    continue
                landings += tally.get("landed", 0) - tally.get("landed_random", 0)
                attempts += tally["attempts"] - tally.get("random_attempts", 0)
            return landings / attempts if attempts else 0.0

        return TossingRoomSplitThrowRates._series(traces=traces, per_run=value)

    @staticmethod
    def family_success_curve(*, traces: list[dict], skill: str) -> list[tuple[int, float, float]]:
        """Fraction of that skill's own goal family solved at each sweep, per seed. The
        counts behind it are printed by `print_table`; this is for the figure only, where
        a rate is what a line can carry."""
        goal = _FAMILY_GOALS[skill]

        def value(*, run: dict, index: int) -> float:
            solved, total = run["sweeps"][index]["families"].get(goal, (0, 0))
            return solved / total if total else 0.0

        return TossingRoomSplitThrowRates._series(traces=traces, per_run=value)

    @staticmethod
    def family_counts(*, traces: list[dict], skill: str) -> list[tuple[int, int, int]]:
        """`(transitions, solved, total)` for one goal family, summed across seeds -- the
        counts the report quotes, so nothing is ever reconstructed by multiplying a
        percentage by n."""
        goal = _FAMILY_GOALS[skill]
        by_transitions: dict[int, tuple[int, int]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            for sweep in run["sweeps"]:
                solved, total = sweep["families"].get(goal, (0, 0))
                seen_solved, seen_total = by_transitions.get(int(sweep["transitions"]), (0, 0))
                by_transitions[int(sweep["transitions"])] = (
                    seen_solved + solved,
                    seen_total + total,
                )
        return [
            (transitions, solved, total)
            for transitions, (solved, total) in sorted(by_transitions.items())
        ]

    # ------------------------------------------------------------------ the statistics

    @staticmethod
    def noise_floor(*, n_first: int, n_second: int) -> float:
        """`sqrt(0.25/n_a + 0.25/n_b)`: the standard error of a difference of two
        proportions at the worst case p = 0.5. The smallest gap this design could
        distinguish from zero even in principle."""
        return math.sqrt(0.25 / n_first + 0.25 / n_second)

    @staticmethod
    def minimum_detectable_effect(*, n_first: int, n_second: int) -> float:
        """The gap this design has 80% power to detect two-sided at alpha = 0.05, i.e.
        2.8 standard errors. Reported next to any gap, because a gap smaller than this is
        not evidence of no difference -- it is evidence of not enough data."""
        return _MDE_CONSTANT * TossingRoomSplitThrowRates.noise_floor(
            n_first=n_first, n_second=n_second
        )

    @staticmethod
    def final_family_rates(*, traces: list[dict], skill: str) -> list[float]:
        """Per-seed final-sweep success rate for one goal family -- the paired vector the
        Wilcoxon test consumes."""
        goal = _FAMILY_GOALS[skill]
        rates: list[float] = []
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            solved, total = run["sweeps"][-1]["families"].get(goal, (0, 0))
            rates.append(solved / total if total else 0.0)
        return rates

    # ---------------------------------------------------------------------- reporting

    @staticmethod
    def print_table(*, traces: list[dict]) -> None:
        runs = TossingRoomSplitThrowRates.runs(traces=traces)
        print(f"\n{len(runs)} seeds: {[run['seed'] for run in runs]}")

        attempts = TossingRoomSplitThrowRates.attempt_totals(traces=traces)
        successes = TossingRoomSplitThrowRates.success_totals(traces=traces)
        print("\nPractice attempts and successes, pooled over every period of every seed")
        header = f"{'skill':>18}{'successes/attempts':>22}{'attempts/seed':>16}"
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            made, tried = successes.get(skill, (0, 0))
            print(f"{skill:>18}{f'{made}/{tried}':>22}{tried / len(runs):>16.1f}")
        # Every lifted skill, not just the two throws: the ratio between the throws is
        # only interpretable next to how much of the practice budget went anywhere else.
        print("\nEvery lifted skill's observed practice attempts")
        header = f"{'skill':>18}{'attempts':>12}{'share of all attempts':>25}"
        print(header)
        print("-" * len(header))
        observed = sum(attempts.values())
        for name, count in sorted(attempts.items(), key=lambda item: -item[1]):
            print(f"{name:>18}{count:>12}{f'{count}/{observed}':>25}")
        print(f"{'TOTAL':>18}{observed:>12}")

        ratio = TossingRoomSplitThrowRates.attempt_ratio(traces=traces)
        trash, recycling = attempts.get("ThrowTrash", 0), attempts.get("ThrowRecycling", 0)
        if ratio is None:
            print("\nattempt ratio: undefined -- ThrowRecycling was never attempted")
        else:
            print(f"\nattempt ratio: {trash}:{recycling} = {ratio:.2f} trash per recycling")

        print("\nPer seed")
        header = f"{'seed':>6}{'ThrowTrash':>14}{'ThrowRecycling':>18}{'ratio':>10}"
        print(header)
        print("-" * len(header))
        for seed, counts in TossingRoomSplitThrowRates.per_seed_attempts(traces=traces).items():
            t, r = counts.get("ThrowTrash", 0), counts.get("ThrowRecycling", 0)
            shown = f"{t / r:.1f}" if r else "n/a"
            print(f"{seed:>6}{t:>14}{r:>18}{shown:>10}")

        print("\nCompetence and task success over training (counts summed across seeds)")
        header = (
            f"{'transitions':>12}{'trash comp':>12}{'recyc comp':>12}"
            f"{'TRASH solved':>16}{'RECYCLING solved':>20}"
        )
        print(header)
        print("-" * len(header))
        trash_comp = TossingRoomSplitThrowRates.competence_curve(traces=traces, skill="ThrowTrash")
        recyc_comp = TossingRoomSplitThrowRates.competence_curve(
            traces=traces, skill="ThrowRecycling"
        )
        trash_counts = TossingRoomSplitThrowRates.family_counts(traces=traces, skill="ThrowTrash")
        recyc_counts = TossingRoomSplitThrowRates.family_counts(
            traces=traces, skill="ThrowRecycling"
        )
        for (transitions, tc, _), (_, rc, _), (_, ts, tt), (_, rs, rt) in zip(
            trash_comp, recyc_comp, trash_counts, recyc_counts, strict=True
        ):
            print(f"{transitions:>12}{tc:>12.3f}{rc:>12.3f}{f'{ts}/{tt}':>16}{f'{rs}/{rt}':>20}")

        TossingRoomSplitThrowRates.print_statistics(traces=traces)

    @staticmethod
    def print_statistics(*, traces: list[dict]) -> None:
        trash_counts = TossingRoomSplitThrowRates.family_counts(traces=traces, skill="ThrowTrash")
        recyc_counts = TossingRoomSplitThrowRates.family_counts(
            traces=traces, skill="ThrowRecycling"
        )
        _, trash_solved, trash_total = trash_counts[-1]
        _, recyc_solved, recyc_total = recyc_counts[-1]
        floor = TossingRoomSplitThrowRates.noise_floor(n_first=trash_total, n_second=recyc_total)
        mde = TossingRoomSplitThrowRates.minimum_detectable_effect(
            n_first=trash_total, n_second=recyc_total
        )
        gap = trash_solved / trash_total - recyc_solved / recyc_total
        print("\nFinal sweep, task level")
        print(f"  TRASH      {trash_solved}/{trash_total}")
        print(f"  RECYCLING  {recyc_solved}/{recyc_total}")
        print(f"  gap (TRASH - RECYCLING)          {100 * gap:+.2f}pp")
        print(f"  binomial noise floor             {100 * floor:.2f}pp")
        print(f"  minimum detectable effect (80%)  {100 * mde:.2f}pp")

        paired = TossingRoomComparison.wilcoxon_signed_rank(
            first=TossingRoomSplitThrowRates.final_family_rates(traces=traces, skill="ThrowTrash"),
            second=TossingRoomSplitThrowRates.final_family_rates(
                traces=traces, skill="ThrowRecycling"
            ),
        )
        print(
            f"  paired Wilcoxon (per seed)       n={paired['n']}, "
            f"W={paired['statistic']}, p={paired['p']:.4f}"
        )
        if paired["n"]:
            print(f"  p floor at this n                {2 / 2 ** paired['n']:.4f}")

        auc = TossingRoomComparison.wilcoxon_signed_rank(
            first=TossingRoomSplitThrowRates.area_under_curve(traces=traces, skill="ThrowTrash"),
            second=TossingRoomSplitThrowRates.area_under_curve(
                traces=traces, skill="ThrowRecycling"
            ),
        )
        trash_auc = statistics.mean(
            TossingRoomSplitThrowRates.area_under_curve(traces=traces, skill="ThrowTrash")
        )
        recyc_auc = statistics.mean(
            TossingRoomSplitThrowRates.area_under_curve(traces=traces, skill="ThrowRecycling")
        )
        print("\nWhole curve, not just its endpoint (area under the per-family curve)")
        print(f"  TRASH mean AUC       {100 * trash_auc:.2f}")
        print(f"  RECYCLING mean AUC   {100 * recyc_auc:.2f}")
        print(f"  difference           {100 * (trash_auc - recyc_auc):+.2f}pp")
        print(f"  paired Wilcoxon      n={auc['n']}, W={auc['statistic']}, p={auc['p']:.4f}")

        print("\nTransitions to first reach a given share of that family's test tasks")
        header = f"{'level':>8}{'TRASH':>12}{'RECYCLING':>14}{'ratio':>10}"
        print(header)
        print("-" * len(header))
        for level in (0.25, 0.5, 0.75, 0.9):
            trash_at = TossingRoomSplitThrowRates.transitions_to_reach(
                traces=traces, skill="ThrowTrash", level=level
            )
            recyc_at = TossingRoomSplitThrowRates.transitions_to_reach(
                traces=traces, skill="ThrowRecycling", level=level
            )
            ratio_text = (
                f"{recyc_at / trash_at:.1f}x" if trash_at and recyc_at and trash_at > 0 else "n/a"
            )
            print(f"{level:>8.2f}{str(trash_at):>12}{str(recyc_at):>14}{ratio_text:>10}")

        print("\nAttempts per practice period (the structural asymmetry, as a distribution)")
        for skill in _SKILL_STYLES:
            histogram = TossingRoomSplitThrowRates.attempts_per_period_histogram(
                traces=traces, skill=skill
            )
            shown = ", ".join(f"{k}: {v}" for k, v in histogram.items())
            print(f"  {skill:>16}  {{{shown}}}")

        greedy = TossingRoomSplitThrowRates.greedy_success_totals(traces=traces)
        random_draws = TossingRoomSplitThrowRates.random_success_totals(traces=traces)
        all_attempts = TossingRoomSplitThrowRates.success_totals(traces=traces)
        print("\nPractice-attempt level (the asymmetric denominator)")
        header = f"{'skill':>18}{'all':>14}{'greedy':>14}{'epsilon-random':>18}"
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            a_made, a_tried = all_attempts.get(skill, (0, 0))
            g_made, g_tried = greedy.get(skill, (0, 0))
            r_made, r_tried = random_draws.get(skill, (0, 0))
            print(
                f"{skill:>18}{f'{a_made}/{a_tried}':>14}"
                f"{f'{g_made}/{g_tried}':>14}{f'{r_made}/{r_tried}':>18}"
            )
        g_trash, g_recyc = greedy.get("ThrowTrash", (0, 0)), greedy.get("ThrowRecycling", (0, 0))
        if g_trash[1] and g_recyc[1]:
            gap = g_trash[0] / g_trash[1] - g_recyc[0] / g_recyc[1]
            mde = TossingRoomSplitThrowRates.minimum_detectable_effect(
                n_first=g_trash[1], n_second=g_recyc[1]
            )
            print(f"  greedy gap (TRASH - RECYCLING)   {100 * gap:+.2f}pp")
            print(f"  minimum detectable effect (80%)  {100 * mde:.2f}pp")

        TossingRoomSplitThrowRates.print_landing_audit(traces=traces)
        TossingRoomSplitThrowRates.print_switch_audit(traces=traces)
        TossingRoomSplitThrowRates.print_sampler_answers(traces=traces)
        TossingRoomSplitThrowRates.print_informed_split(traces=traces)

    @staticmethod
    def print_switch_audit(*, traces: list[dict]) -> None:
        """Whether the per-family curve is a curve at all, or a mean over seeds that are
        each at one end or the other.

        Thresholds are 12/14 and 4/14, the same ones the Tossing Room baseline used, so
        the two domains' splits are read on the same scale."""
        print("\nIs the curve a curve? Every (seed, checkpoint) scored against 12/14 and 4/14")
        header = f"{'skill':>18}{'at an extreme':>16}{'in between':>14}{'seed-checkpoints':>19}"
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            split = TossingRoomSplitThrowRates.seed_checkpoint_extremes(
                traces=traces, skill=skill, high=12, low=4
            )
            extreme = f"{split['extreme']}/{split['total']}"
            middle = f"{split['middle']}/{split['total']}"
            print(f"{skill:>18}{extreme:>16}{middle:>14}{split['total']:>19}")

        print("\nPer seed: the peak that family reached, and where it ended")
        header = (
            f"{'seed':>6}{'TRASH peak':>13}{'TRASH final':>14}"
            f"{'RECYCLING peak':>17}{'RECYCLING final':>18}"
        )
        print(header)
        print("-" * len(header))
        trash_peaks = TossingRoomSplitThrowRates.per_seed_family_peaks(
            traces=traces, skill="ThrowTrash"
        )
        recyc_peaks = TossingRoomSplitThrowRates.per_seed_family_peaks(
            traces=traces, skill="ThrowRecycling"
        )
        for seed in sorted(trash_peaks):
            t, r = trash_peaks[seed], recyc_peaks[seed]
            t_peak, t_final = f"{t['peak']}/{t['total']}", f"{t['final']}/{t['total']}"
            r_peak, r_final = f"{r['peak']}/{r['total']}", f"{r['final']}/{r['total']}"
            print(f"{seed:>6}{t_peak:>13}{t_final:>14}{r_peak:>17}{r_final:>18}")

    @staticmethod
    def print_sampler_answers(*, traces: list[dict]) -> None:
        """What each sampler answered, not just how often it was asked.

        The threshold is 3x `throw_tolerance` = 0.30: a draw that far from the force its
        own grounding required is a wrong answer rather than an inaccurate one. See
        `badly_missed_force_totals` for why this is measured per grounding now."""
        unreachable = TossingRoomSplitThrowRates.badly_missed_force_totals(
            traces=traces, miss_threshold=_BADLY_MISSED_THRESHOLD
        )
        if not any(drawn for _below, drawn in unreachable.values()):
            print("\nNo greedy forces recorded -- these traces predate the force instrumentation.")
            return
        print("\nWhat each sampler answered (greedy draws only; missed its grounding by >0.30)")
        header = f"{'skill':>18}{'missed by >0.30':>18}{'longest all-miss streak, per seed':>36}"
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            below, drawn = unreachable.get(skill, (0, 0))
            streaks = TossingRoomSplitThrowRates.longest_missing_streaks(traces=traces, skill=skill)
            shown = ", ".join(str(streaks[seed]) for seed in sorted(streaks))
            print(f"{skill:>18}{f'{below}/{drawn}':>16}{shown:>36}")

    # ---------------------------------------------------------- the per-seed figure

    @staticmethod
    def per_seed_family_series(
        *, traces: list[dict], skill: str
    ) -> dict[int, list[tuple[int, int]]]:
        """`{seed: [(transitions, solved), ...]}` in counts, for one goal family. The
        per-seed lines a mean would hide."""
        goal = _FAMILY_GOALS[skill]
        series: dict[int, list[tuple[int, int]]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            series[run["seed"]] = [
                (int(sweep["transitions"]), sweep["families"].get(goal, (0, 0))[0])
                for sweep in run["sweeps"]
            ]
        return series

    @staticmethod
    def per_seed_force_series(
        *, traces: list[dict], skill: str
    ) -> dict[int, list[tuple[int, float]]]:
        """`{seed: [(period index, mean greedy force that period), ...]}`. Periods with no
        greedy draw are omitted rather than interpolated -- for the throw the layout
        rations, most periods have none, and drawing a line through them would invent
        exactly the smoothness under examination."""
        series: dict[int, list[tuple[int, float]]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            points: list[tuple[int, float]] = []
            for index, period in enumerate(run["periods"]):
                forces = period["skills"].get(skill, {}).get("greedy_forces") or []
                if forces:
                    points.append((index, statistics.mean(forces)))
            series[run["seed"]] = points
        return series

    @staticmethod
    def render_per_seed(*, traces: list[dict], output: Path) -> None:
        """The spread a mean hides: one line per seed for each family, the distribution of
        every seed-checkpoint score, and what each sampler answered period by period."""
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
        trash_ax, recyc_ax, spread_ax, force_ax = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

        for ax, skill in ((trash_ax, "ThrowTrash"), (recyc_ax, "ThrowRecycling")):
            color, _linestyle = _SKILL_STYLES[skill]
            series = TossingRoomSplitThrowRates.per_seed_family_series(traces=traces, skill=skill)
            for seed, points in series.items():
                ax.plot(
                    [transitions for transitions, _solved in points],
                    [solved for _transitions, solved in points],
                    linewidth=1.1,
                    alpha=0.75,
                    color=color,
                )
                ax.annotate(
                    str(seed),
                    xy=points[-1],
                    fontsize=7,
                    color=color,
                    xytext=(3, -2),
                    textcoords="offset points",
                )
            ax.set_ylim(-0.5, 14.5)
            ax.set_yticks(range(0, 15, 2))
            ax.set_ylabel("tasks solved out of 14")
            ax.set_xlabel("Number of online transitions")
            ax.axhline(12, color="0.4", linewidth=0.8, linestyle="--")
            ax.axhline(4, color="0.4", linewidth=0.8, linestyle="--")
            split = TossingRoomSplitThrowRates.seed_checkpoint_extremes(
                traces=traces, skill=skill, high=12, low=4
            )
            ax.set_title(
                f"{skill}, one line per seed\n{split['extreme']}/{split['total']} "
                f"seed-checkpoints at an extreme ({split['middle']}/{split['total']} in "
                "between the dashed lines)",
                fontsize=9,
            )

        width = 0.4
        for offset, skill in zip((-width / 2, width / 2), _SKILL_STYLES, strict=True):
            color, _linestyle = _SKILL_STYLES[skill]
            goal = _FAMILY_GOALS[skill]
            histogram = {score: 0 for score in range(15)}
            for run in TossingRoomSplitThrowRates.runs(traces=traces):
                for sweep in run["sweeps"]:
                    solved, total = sweep["families"].get(goal, (0, 0))
                    if total:
                        histogram[solved] += 1
            spread_ax.bar(
                [score + offset for score in histogram],
                list(histogram.values()),
                width=width,
                color=color,
                label=skill,
            )
        spread_ax.set_xlabel("tasks solved out of 14, at one (seed, checkpoint)")
        spread_ax.set_ylabel("number of seed-checkpoints")
        spread_ax.set_xticks(range(0, 15, 2))
        spread_ax.set_title(
            "Every seed-checkpoint, as a distribution: the ends are where the mass is",
            fontsize=9,
        )

        for skill in _SKILL_STYLES:
            color, _linestyle = _SKILL_STYLES[skill]
            series = TossingRoomSplitThrowRates.per_seed_force_series(traces=traces, skill=skill)
            for points in series.values():
                force_ax.plot(
                    [index for index, _force in points],
                    [force for _index, force in points],
                    linewidth=0.9,
                    alpha=0.5,
                    color=color,
                )
        # The shaded band is the support of the required force, [0.1, 0.9] on the
        # defaults. Unlike the old U(0.5, 1.0) target band it covers nearly the whole draw
        # range, so sitting inside it is no longer evidence of anything -- what the title
        # reports is the per-grounding miss instead.
        force_ax.axhspan(0.1, 0.9, color="0.6", alpha=0.15, linewidth=0)
        force_ax.set_ylim(0, 1.02)
        force_ax.set_xlabel("practice period (0-24)")
        force_ax.set_ylabel("mean greedy force chosen that period")
        missed = TossingRoomSplitThrowRates.badly_missed_force_totals(
            traces=traces, miss_threshold=_BADLY_MISSED_THRESHOLD
        )
        trash_below, trash_drawn = missed.get("ThrowTrash", (0, 0))
        recyc_below, recyc_drawn = missed.get("ThrowRecycling", (0, 0))
        force_ax.set_title(
            "What each sampler answered — shaded band is the required-force support\n"
            f"missed its own grounding by more than 0.30: TRASH {trash_below}/{trash_drawn}"
            f" draws, RECYCLING {recyc_below}/{recyc_drawn}",
            fontsize=9,
        )

        for ax in (trash_ax, recyc_ax, spread_ax, force_ax):
            ax.grid(True, alpha=0.25, linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        spread_ax.legend(fontsize=8, framealpha=0.9, loc="best")
        # Built by hand: every seed is drawn as its own line, so asking matplotlib for the
        # labels would produce ten identical entries per skill.
        force_ax.legend(
            handles=[
                plt.Line2D([], [], color=_SKILL_STYLES[skill][0], label=skill)
                for skill in _SKILL_STYLES
            ],
            fontsize=8,
            framealpha=0.9,
            loc="best",
        )
        fig.suptitle(
            "Tossing Room (split throws), capacity-1: the per-seed spread a mean hides "
            "(10 seeds, 14 tasks per throw family)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def print_landing_audit(*, traces: list[dict]) -> None:
        """What EES scored against what the environment actually did.

        A throw's add effects are `{<Kind>InBin(item, bin), HandEmpty(robot)}`, and on the
        pre-capacity-1 domain a throw always released the item, so a throw into an ALREADY
        non-empty bin was scored a success at any force. The trash bin could be in that
        state and the recycling bin could not, which made the two skills' success signals
        incomparable and inverted the per-attempt comparison that was published from them.

        On the CURRENT domain this is a regression check: capacity-1 bins, the bin-empty
        precondition on each throw and the dynamics' refusal of a throw at a full bin
        together mean the `prefilled bin` and `scored, missed` columns must read 0 for both
        skills -- as they do on the committed traces. They are printed rather than dropped
        precisely so a reader can see that, and so a future change that reopens the channel
        is loud rather than silent."""
        landings = TossingRoomSplitThrowRates.landing_totals(traces=traces)
        prefilled = TossingRoomSplitThrowRates.prefilled_totals(traces=traces)
        inflated = TossingRoomSplitThrowRates.inflated_successes(traces=traces)
        print("\nScored success vs. what actually landed (the add-effect audit)")
        header = (
            f"{'skill':>18}{'landed/attempts':>19}{'scored/attempts':>19}"
            f"{'prefilled bin':>16}{'scored, missed':>17}"
        )
        print(header)
        print("-" * len(header))
        for skill in _SKILL_STYLES:
            landed, attempts = landings.get(skill, (0, 0))
            filled, _ = prefilled.get(skill, (0, 0))
            spurious, scored = inflated.get(skill, (0, 0))
            print(
                f"{skill:>18}{f'{landed}/{attempts}':>19}{f'{scored}/{attempts}':>19}"
                f"{f'{filled}/{attempts}':>16}{f'{spurious}/{scored}':>17}"
            )
        trash_landed, trash_attempts = landings.get("ThrowTrash", (0, 0))
        recyc_landed, recyc_attempts = landings.get("ThrowRecycling", (0, 0))
        if trash_attempts and recyc_attempts:
            gap = trash_landed / trash_attempts - recyc_landed / recyc_attempts
            mde = TossingRoomSplitThrowRates.minimum_detectable_effect(
                n_first=trash_attempts, n_second=recyc_attempts
            )
            print(f"  landing gap (TRASH - RECYCLING)  {100 * gap:+.2f}pp")
            print(f"  minimum detectable effect (80%)  {100 * mde:.2f}pp")

    # ------------------------------------------------------------------------ figure

    @staticmethod
    def _plot(*, ax, series: list[tuple[int, float, float]], label: str, skill: str) -> None:
        color, linestyle = _SKILL_STYLES[skill]
        xs = [transitions for transitions, _mean, _stderr in series]
        means = [mean for _transitions, mean, _stderr in series]
        errs = [stderr for _transitions, _mean, stderr in series]
        ax.plot(xs, means, label=label, linewidth=2, color=color, linestyle=linestyle)
        ax.fill_between(
            xs,
            [m - e for m, e in zip(means, errs, strict=True)],
            [m + e for m, e in zip(means, errs, strict=True)],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    @staticmethod
    def render(*, traces: list[dict], output: Path) -> None:
        """Four panels: what the layout affords each throw (top row) and what each throw
        got out of it (bottom row).

        The top-right panel is a histogram rather than a line because the claim it
        carries -- "recycling buys at most one attempt per period, ever" -- is about a
        distribution, and a mean of 0.8 attempts per period would be equally consistent
        with a story this domain is not telling."""
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
        attempts_ax, histogram_ax, learning_ax, task_ax = (
            axes[0][0],
            axes[0][1],
            axes[1][0],
            axes[1][1],
        )

        for skill in _SKILL_STYLES:
            color, _linestyle = _SKILL_STYLES[skill]
            TossingRoomSplitThrowRates._plot(
                ax=attempts_ax,
                series=TossingRoomSplitThrowRates.cumulative_attempts(traces=traces, skill=skill),
                label=skill,
                skill=skill,
            )
            TossingRoomSplitThrowRates._plot(
                ax=learning_ax,
                series=TossingRoomSplitThrowRates.greedy_landing_curve(traces=traces, skill=skill),
                label=f"{skill} — actually landed",
                skill=skill,
            )
            # Competence on the same axes as the measured landing rate, deliberately: all
            # three lines answer "how good is this skill", and showing them together is
            # what makes visible that EES's own estimate sits far above what either
            # sampler achieves (the Beta(10, 1) prior dominates a skill with few
            # observations) and that its *scored* successes are, for the trash throw,
            # mostly throws that missed.
            competence = TossingRoomSplitThrowRates.competence_curve(traces=traces, skill=skill)
            learning_ax.plot(
                [transitions for transitions, _mean, _stderr in competence],
                [mean for _transitions, mean, _stderr in competence],
                label=f"{skill} — EES competence",
                linewidth=1.2,
                linestyle=":",
                color=color,
                alpha=0.85,
            )
            scored = TossingRoomSplitThrowRates.greedy_success_curve(traces=traces, skill=skill)
            learning_ax.plot(
                [transitions for transitions, _mean, _stderr in scored],
                [mean for _transitions, mean, _stderr in scored],
                label=f"{skill} — EES scored a success",
                linewidth=1.0,
                linestyle=(0, (1, 3)),
                color=color,
                alpha=0.55,
            )
            TossingRoomSplitThrowRates._plot(
                ax=task_ax,
                series=TossingRoomSplitThrowRates.family_success_curve(traces=traces, skill=skill),
                label=skill,
                skill=skill,
            )

        width = 0.4
        for offset, skill in zip((-width / 2, width / 2), _SKILL_STYLES, strict=True):
            histogram = TossingRoomSplitThrowRates.attempts_per_period_histogram(
                traces=traces, skill=skill
            )
            color, _linestyle = _SKILL_STYLES[skill]
            histogram_ax.bar(
                [key + offset for key in histogram],
                list(histogram.values()),
                width=width,
                color=color,
                label=skill,
            )

        attempts_ax.set_ylabel("cumulative practice attempts")
        attempts_ax.set_title("How much practice each throw got", fontsize=10)
        attempts_ax.set_ylim(bottom=0)

        histogram_ax.set_xlabel("attempts within one 100-step practice period")
        histogram_ax.set_ylabel("number of periods")
        histogram_ax.set_title(
            "Recycling is 0 or 1, never more: the one-way ledge, in one picture",
            fontsize=10,
        )

        learning_ax.set_ylabel("fraction of greedy practice attempts")
        learning_ax.set_title(
            "What landed, what EES scored, and what EES believes — three different things",
            fontsize=10,
        )
        learning_ax.set_ylim(0, 1.02)

        task_ax.set_ylabel("fraction of that family's test tasks solved")
        task_ax.set_title("Task success, by goal family", fontsize=10)
        task_ax.set_ylim(0, 1.02)

        for ax in (attempts_ax, learning_ax, task_ax):
            ax.set_xlabel("Number of online transitions")
        for ax in (attempts_ax, histogram_ax, learning_ax, task_ax):
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.legend(fontsize=8, framealpha=0.9, loc="best")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.suptitle(
            "Tossing Room (split throws), capacity-1: two lifted skills, two samplers, "
            "very different practice budgets (mean ± stderr over 10 seeds)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def render_informed_split(*, traces: list[dict], output: Path) -> None:
        """Results (4) and (6) drawn, per seed rather than pooled.

        Left: how much of each seed's "learned-sampler" pool was actually a
        discriminating classifier, against how much was `sample`'s uniform fallback.
        Right: the landing rate over the informed draws alone, one point per seed, with
        the pooled rate as a line. Ten seeds and single-digit recycling denominators
        mean a bar chart of two means would hide everything that matters here."""
        informed_pooled = TossingRoomSplitThrowRates.informed_landing_totals(traces=traces)
        random_pooled = TossingRoomSplitThrowRates.random_landing_totals(traces=traces)
        fig, (split_ax, rate_ax) = plt.subplots(1, 2, figsize=(12.5, 4.6))

        seeds = sorted(run["seed"] for run in TossingRoomSplitThrowRates.runs(traces=traces))
        width = 0.38
        for offset, skill in zip((-width / 2, width / 2), _SKILL_STYLES, strict=True):
            per_seed = TossingRoomSplitThrowRates.per_seed_informed_draws(
                traces=traces, skill=skill
            )
            colour, _linestyle = _SKILL_STYLES[skill]
            positions = [index + offset for index in range(len(seeds))]
            informed = [per_seed.get(seed, (0, 0))[0] for seed in seeds]
            greedy = [per_seed.get(seed, (0, 0))[1] for seed in seeds]
            split_ax.bar(
                positions,
                greedy,
                width=width,
                color=colour,
                alpha=0.28,
                label=f"{skill}: all greedy draws",
            )
            split_ax.bar(
                positions,
                informed,
                width=width,
                color=colour,
                label=f"{skill}: classifier discriminated",
            )
        split_ax.set_xticks(range(len(seeds)))
        split_ax.set_xticklabels([str(seed) for seed in seeds])
        split_ax.set_xlabel("seed")
        split_ax.set_ylabel("draws (count, not a rate)")
        split_ax.set_title("How many 'learned-sampler' draws were actually learned")
        split_ax.legend(fontsize=8)

        for skill in _SKILL_STYLES:
            colour, _linestyle = _SKILL_STYLES[skill]
            xs = []
            ys = []
            for run in TossingRoomSplitThrowRates.runs(traces=traces):
                landed = 0
                attempts = 0
                for period in run["periods"]:
                    tally = period["skills"].get(skill)
                    if tally is None:
                        continue
                    landed += tally.get("informed_landed", 0)
                    attempts += tally.get("informed_attempts", 0)
                if attempts:
                    xs.append(seeds.index(run["seed"]))
                    ys.append(landed / attempts)
            rate_ax.scatter(xs, ys, color=colour, label=f"{skill} (per seed)", zorder=3)
            pooled_landed, pooled_attempts = informed_pooled.get(skill, (0, 0))
            if pooled_attempts:
                rate_ax.axhline(
                    pooled_landed / pooled_attempts,
                    color=colour,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{skill} informed {pooled_landed}/{pooled_attempts}",
                )
            # The control the informed rate has to beat to mean anything: the same
            # skill's own epsilon-random draws. A learned sampler sitting on its own
            # coin-flip line has learned nothing, which no pooled greedy number can say.
            random_landed, random_attempts = random_pooled.get(skill, (0, 0))
            if random_attempts:
                rate_ax.axhline(
                    random_landed / random_attempts,
                    color=colour,
                    linestyle=":",
                    linewidth=1.6,
                    label=f"{skill} epsilon-random {random_landed}/{random_attempts}",
                )
        rate_ax.set_xticks(range(len(seeds)))
        rate_ax.set_xticklabels([str(seed) for seed in seeds])
        rate_ax.set_xlabel("seed (a seed with no informed draw has no point)")
        rate_ax.set_ylabel("landed / informed draws")
        rate_ax.set_ylim(-0.05, 1.05)
        rate_ax.set_title("Landing rate over informed draws, against the coin-flip control")
        rate_ax.legend(fontsize=7)

        fig.suptitle(
            "Separating a trained classifier's greedy draws from sample()'s uniform "
            "fallback, per seed",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        action="append",
        required=True,
        help="Repeatable: one shard JSON from scripts/tossingroomsplit_skill_traces.py.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="The <root>/ees/<seed>/ tree from scripts/run_sweep.py, for the "
        "traced-vs-swept consistency gate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-seed-output",
        type=Path,
        default=None,
        help="Second figure: one line per seed rather than a mean, the distribution of "
        "every seed-checkpoint score, and the force each sampler actually chose.",
    )
    parser.add_argument(
        "--informed-output",
        type=Path,
        default=None,
        help="Third figure: the greedy pool split per seed into the draws a "
        "discriminating classifier made and sample()'s uniform fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    traces = [json.loads(path.read_text()) for path in args.traces]
    problems = TossingRoomSplitThrowRates.check_against_sweep(
        traces=traces, results_root=args.results_root
    )
    if problems:
        print("TRACED RUNS DISAGREE WITH THE SWEEP -- refusing to report:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"consistency gate: all {len(TossingRoomSplitThrowRates.runs(traces=traces))} traced "
        f"seeds reproduce their swept stats.json exactly"
    )
    TossingRoomSplitThrowRates.print_table(traces=traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    TossingRoomSplitThrowRates.render(traces=traces, output=args.output)
    print(f"\nwrote {args.output}")
    if args.per_seed_output is not None:
        args.per_seed_output.parent.mkdir(parents=True, exist_ok=True)
        TossingRoomSplitThrowRates.render_per_seed(traces=traces, output=args.per_seed_output)
        print(f"wrote {args.per_seed_output}")
    if args.informed_output is not None:
        args.informed_output.parent.mkdir(parents=True, exist_ok=True)
        TossingRoomSplitThrowRates.render_informed_split(traces=traces, output=args.informed_output)
        print(f"wrote {args.informed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
