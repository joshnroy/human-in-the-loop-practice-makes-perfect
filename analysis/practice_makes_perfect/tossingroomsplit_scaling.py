"""Post-run analysis for Tossing Room (split throws) at a 10x interaction budget: is
`ThrowRecycling`'s sampler limited by how many attempts it gets, or by something more
attempts cannot fix?

At the standard 2,500-transition budget recycling's informed draws land 11/56 against its
own epsilon-random control's 11/57 -- a null result at an MDE of 26.36pp
(`docs/experiment-logs/2026-08-05-tossingroomsplit-throw-rates.md`, result 4). That page
could not say whether the sampler is *slow* or *broken*, because a null result at one
sample size is compatible with both. This module measures the same comparison as a
function of budget, so the two can be told apart: a slow sampler separates from its
control somewhere on the curve, a broken one never does.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- never runs a
simulation), the same two sources and the same consistency gate as
`tossingroomsplit_throw_rates`, which it imports rather than reimplements:

* `--results-root <dir>` -- the `<root>/ees/<seed>/` tree from `scripts/run_sweep.py`.
* `--traces <file>` (repeatable) -- shards from `scripts/tossingroomsplit_skill_traces.py`.

**The headline is a curve, not an endpoint.** A single 25,000-transition number would
answer "is it better at the end", which is not the question. Where the two rates separate
-- and whether that place is a transition count or a number of accumulated successes -- is
what transfers to another domain, since transitions are a property of this layout's
budget while accumulated successes are a property of the classifier.

**Two binnings, deliberately.** Draws are pooled by elapsed transitions AND, separately,
by how many landings that skill had already accumulated in that seed when the draw was
made. The second is the mechanism variable: `MlpBinaryClassifier` only ever sees its
positives, one landing pins where the good force region sits for one target, and only two
landings at well-separated targets reveal the SLOPE of the force/target relation. If the
gap opens at a fixed accumulated-landing count regardless of when in the run that count
was reached, that is a statement about the classifier; if it opens at a fixed transition
count regardless of how many landings had accrued, it is a statement about this layout.
`factorial_split` crosses the two so they cannot be confounded.

**Every MDE is derived from its own two denominators**, via the imported `noise_floor` /
`minimum_detectable_effect`. The 20.19pp figure quoted in PR #90 is the MDE of that page's
310-vs-57 epsilon-random comparison (whose noise floor is 7.21pp), and it is NOT the MDE
of its 56-vs-57 null, which is 26.36pp against a floor of 9.41pp. Do not carry either
number between comparisons.

**Bin width is chosen to reproduce the standard run's sample size.** A 2,500-transition
bin is exactly one standard run, so the first bin of this sweep should reproduce the
published null and each later bin is an independent replication at the same power. That
makes the bins comparable to each other and to the prior result, at the cost of testing
ten hypotheses -- so a Holm-Bonferroni-adjusted threshold is reported beside the raw p.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Literal

import matplotlib
from pydantic import BaseModel

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.tossingroom_comparison import (  # noqa: E402
    TossingRoomComparison,
)
from analysis.practice_makes_perfect.tossingroomsplit_throw_rates import (  # noqa: E402
    _SKILL_STYLES,
    TossingRoomSplitThrowRates,
)

# `_SKILL_STYLES` is imported rather than copied: the two pages' figures are meant to be
# readable side by side, and two copies of a colour map drift the moment one is edited.
#
# Accumulated-landing bands, as lower bounds with the last open-ended. Unequal on purpose:
# the structure the diagnosis predicts is all at 0, 1 and 2 positives -- one landing pins
# where the good force region sits, two are the minimum that can pin its slope -- and the
# tail only needs enough resolution to show whether the gap keeps widening. Defined once
# because the printed table and the figure must not be able to disagree about it.
_LANDING_EDGES: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 20)
# The two draw kinds this page compares. `fallback` -- `LearnedSkillSampler.sample`'s
# uniform draw on a degenerate score vector -- is neither, and is reported separately
# rather than pooled into either: pooling it into `informed` is the exact error PR #90
# corrected, and pooling it into `random` would flatter the control.
_INFORMED: Literal["informed"] = "informed"
_RANDOM: Literal["random"] = "random"
# `TossingRoomSplitEnvironment.throw_tolerance`. A throw lands when it is within this of
# the force its grounding required, so two landings separated by LESS than this are
# consistent with a single flat force band and cannot reveal a slope.
_THROW_TOLERANCE = 0.1
# One standard run (25 cycles x 100 steps), so bin 1 replicates the published null.
_DEFAULT_BIN_WIDTH = 2500


class Draw(BaseModel):
    """One throw attempt, with the state of that skill's evidence when it was made.

    `transitions` is the count of online transitions completed BEFORE this draw's
    practice period began, not after it. The sampler that produced the draw was last
    refitted at the preceding cycle boundary, so that is the amount of data it actually
    had -- dating the draw by the end of its own period would credit the sampler with
    evidence it had not yet seen.

    `prior_landings` and `prior_separation` are likewise strictly-before quantities, and
    both count EVERY landing rather than only the informed ones: `observe_outcome` feeds
    the classifier every attempt it makes, epsilon-random draws included, so the random
    draws are part of its training set even though they are the control it is scored
    against. Restricting them to informed landings would understate the evidence
    available at exactly the early draws where the question is decided."""

    seed: int
    transitions: int
    target: float
    landed: bool
    kind: Literal["random", "informed", "fallback"]
    prior_landings: int
    prior_separation: float


class TossingRoomSplitScaling:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ------------------------------------------------------------ reading the traces

    @staticmethod
    def steps_per_period(*, traces: list[dict]) -> int:
        """The shards' common `max_steps_per_interaction`, after checking that they came
        from ONE experiment.

        Both the protocol length and the period length are checked, because the failure
        this guards against is pooling the standard-budget shards with the 10x ones: they
        share `max_steps_per_interaction = 100` and differ only in `num_cycles`, so a
        period-length check alone would wave them through, the consistency gate would pass
        (each shard really does match its own sweep), and the first window -- the one that
        replicates the published null -- would silently carry twice the sample from two
        different experiments. That is the single comparison this page exists to make."""
        for field in ("max_steps_per_interaction", "num_cycles"):
            values = {shard[field] for shard in traces}
            if len(values) != 1:
                raise ValueError(
                    f"shards disagree on {field}: {sorted(values)} -- these are different "
                    "experiments and must not be pooled"
                )
        return traces[0]["max_steps_per_interaction"]

    @staticmethod
    def require_per_draw_record(*, traces: list[dict]) -> None:
        """Refuse traces collected before the per-draw throw record existed.

        Called at the top of every entry point rather than only from `draws`, because the
        mechanism series walk the raw periods themselves; a guard on one path only would
        still let the other fail as a bare `KeyError` deep in a zip, which is exactly the
        confusion it exists to prevent. Every count on this page is per draw, so there is
        no degraded mode -- those traces have to be recollected."""
        for shard in traces:
            for run in shard["seeds"]:
                for index, period in enumerate(run["periods"]):
                    for name, tally in period["skills"].items():
                        if name.startswith("Throw") and "throw_kinds" not in tally:
                            raise KeyError(
                                f"seed {run['seed']}, period {index}, {name}: this trace "
                                "has no per-draw throw record (`throw_targets`/"
                                "`throw_landed_flags`/`throw_kinds`), so it predates the "
                                "collector this analysis needs. Recollect with "
                                "scripts/tossingroomsplit_skill_traces.py; the per-period "
                                "counts it does carry cannot substitute, because every "
                                "comparison here is per draw."
                            )

    @staticmethod
    def transitions_before(*, run: dict, index: int) -> int:
        """Online transitions completed before practice period `index` began.

        Read off the run's own evaluation record rather than computed as
        `index * max_steps_per_interaction`. A period does NOT have to consume its full
        step budget -- `EesMethod` raises `InteractionComplete` and `PracticeLoop` does
        not charge the unspent steps -- so the arithmetic is an assumption about the
        method, while `sweeps[index]["transitions"]` is the harness's own record of the
        same quantity. The two agree on every run measured so far; at 250 periods there
        are ten times as many chances for them not to."""
        return run["sweeps"][index]["transitions"]

    @staticmethod
    def draws(*, traces: list[dict], skill: str) -> list[Draw]:
        """Every throw attempt of one skill, across every seed, in execution order.

        Flat rather than grouped by seed because every comparison below pools over seeds
        and filters on `Draw.seed` when it needs one; a dict-of-lists would be unpacked
        again at every call site."""
        TossingRoomSplitScaling.require_per_draw_record(traces=traces)
        TossingRoomSplitScaling.steps_per_period(traces=traces)
        collected: list[Draw] = []
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            landed_targets: list[float] = []
            for index, period in enumerate(run["periods"]):
                tally = period["skills"].get(skill)
                if tally is None:
                    continue
                for target, landed, kind in zip(
                    tally["throw_targets"],
                    tally["throw_landed_flags"],
                    tally["throw_kinds"],
                    strict=True,
                ):
                    collected.append(
                        Draw(
                            seed=run["seed"],
                            transitions=TossingRoomSplitScaling.transitions_before(
                                run=run, index=index
                            ),
                            target=target,
                            landed=landed,
                            kind=kind,
                            prior_landings=len(landed_targets),
                            prior_separation=(
                                max(landed_targets) - min(landed_targets)
                                if len(landed_targets) > 1
                                else 0.0
                            ),
                        )
                    )
                    if landed:
                        landed_targets.append(target)
        return collected

    # ------------------------------------------------------------------- comparisons

    @staticmethod
    def rate(*, draws: list[Draw], kind: str) -> tuple[int, int]:
        """`(landed, attempts)` for one kind of draw. Always returned as a pair so no
        caller can accidentally report the rate without its denominator."""
        subset = [draw for draw in draws if draw.kind == kind]
        return sum(draw.landed for draw in subset), len(subset)

    @staticmethod
    def compare(*, draws: list[Draw], label: str) -> dict:
        """Informed draws against their own epsilon-random control, on one subset.

        The comparison is each skill against ITS OWN coin flip within the same runs, not
        against the other skill: the two skills face different task distributions and
        different amounts of practice, so a cross-skill difference confounds the sampler
        with the layout. Within one skill the epsilon branch fires on the same tasks in
        the same states, so it is a genuine control."""
        informed_landed, informed_total = TossingRoomSplitScaling.rate(draws=draws, kind=_INFORMED)
        random_landed, random_total = TossingRoomSplitScaling.rate(draws=draws, kind=_RANDOM)
        record = {
            "label": label,
            "informed": (informed_landed, informed_total),
            "random": (random_landed, random_total),
            "gap": None,
            "p": None,
            "noise_floor": None,
            "mde": None,
        }
        if informed_total == 0 or random_total == 0:
            return record
        record["gap"] = 100.0 * (informed_landed / informed_total - random_landed / random_total)
        record["p"] = TossingRoomComparison.fisher_exact_two_sided(
            a=informed_landed,
            b=informed_total - informed_landed,
            c=random_landed,
            d=random_total - random_landed,
        )
        record["noise_floor"] = 100.0 * TossingRoomSplitThrowRates.noise_floor(
            n_first=informed_total, n_second=random_total
        )
        record["mde"] = 100.0 * TossingRoomSplitThrowRates.minimum_detectable_effect(
            n_first=informed_total, n_second=random_total
        )
        return record

    @staticmethod
    def binned_by_transitions(*, draws: list[Draw], bin_width: int) -> list[dict]:
        """One comparison per `bin_width`-transition window, non-overlapping.

        Windowed rather than cumulative because a cumulative rate is dominated by its
        early data and would smear a genuine crossover across the rest of the run: a
        sampler that switched on at 10,000 transitions would still show a depressed
        cumulative rate at 25,000. The cumulative view is kept in `cumulative_by_seed`
        for the per-seed traces, where the sparsity of a single seed's draws makes a
        windowed rate unreadable.

        A window with no draws is KEPT, as a record whose arms are both 0/0 and whose gap
        and p are `None`. Dropping it would leave a hole that the printed table hides and
        that the plotted line silently interpolates across, drawing a straight segment
        through a budget range where nothing was measured."""
        if not draws:
            return []
        span = max(draw.transitions for draw in draws) + bin_width
        records = []
        for start in range(0, span, bin_width):
            window = [draw for draw in draws if start <= draw.transitions < start + bin_width]
            records.append(
                TossingRoomSplitScaling.compare(draws=window, label=f"{start}-{start + bin_width}")
                | {"start": start, "end": start + bin_width}
            )
        return records

    @staticmethod
    def binned_by_prior_landings(
        *, draws: list[Draw], edges: tuple[int, ...] = _LANDING_EDGES
    ) -> list[dict]:
        """One comparison per accumulated-landings band -- the mechanism variable.

        `edges` are lower bounds and the last band is open-ended, so they must start at 0
        and increase strictly: an edge list starting above 0 would silently drop every
        draw below it, and an unsorted or duplicated one would silently put the same draw
        in two bands. Both produce a table whose denominators do not sum to the sample,
        which is invisible on the page, so they are refused here."""
        if list(edges) != sorted(set(edges)) or edges[0] != 0:
            raise ValueError(
                f"landing-band edges must start at 0 and strictly increase, got {edges}: "
                "otherwise draws are silently dropped or double-counted"
            )
        records = []
        for index, lower in enumerate(edges):
            upper = edges[index + 1] if index + 1 < len(edges) else None
            band = [
                draw
                for draw in draws
                if draw.prior_landings >= lower and (upper is None or draw.prior_landings < upper)
            ]
            if not band:
                continue
            label = (
                f"{lower}+"
                if upper is None
                else (f"{lower}" if upper == lower + 1 else f"{lower}-{upper - 1}")
            )
            records.append(
                TossingRoomSplitScaling.compare(draws=band, label=label) | {"lower": lower}
            )
        return records

    @staticmethod
    def factorial_split(*, draws: list[Draw], landing_threshold: int) -> list[dict]:
        """The 2x2 that stops transitions and accumulated landings from being confounded.

        They are correlated by construction -- landings accrue as the run goes on -- so a
        gap that appears "late" and a gap that appears "after k landings" look identical
        until the two are crossed. The informative cells are the off-diagonal ones: LATE
        draws that still have few landings behind them, and EARLY draws that already have
        many. Whichever factor carries the gap in those two cells is the one that
        matters.

        The median that splits early from late is taken over only the draws that are
        actually compared -- informed and epsilon-random. Including the uniform fallback
        would bias it: a fallback draw is by definition one whose classifier could not
        rank candidates, so they are concentrated early, and pooling them drags the
        midpoint earlier than the median of the compared draws and leaves the "early"
        cells systematically under-powered against the "late" ones.

        Each cell carries its own denominators and its own MDE, so a cell too small to
        support a conclusion shows up as an MDE larger than any plausible effect. The
        caller reports that; this function does not adjudicate."""
        compared = [draw for draw in draws if draw.kind in (_INFORMED, _RANDOM)]
        if not compared:
            return []
        midpoint = statistics.median(draw.transitions for draw in compared)
        cells = []
        for time_label, time_test in (
            ("early", lambda draw: draw.transitions < midpoint),
            ("late", lambda draw: draw.transitions >= midpoint),
        ):
            for evidence_label, evidence_test in (
                (f"<{landing_threshold} landings", lambda d: d.prior_landings < landing_threshold),
                (
                    f">={landing_threshold} landings",
                    lambda d: d.prior_landings >= landing_threshold,
                ),
            ):
                cell = [draw for draw in compared if time_test(draw) and evidence_test(draw)]
                cells.append(
                    TossingRoomSplitScaling.compare(
                        draws=cell, label=f"{time_label}, {evidence_label}"
                    )
                    | {"midpoint": midpoint}
                )
        return cells

    @staticmethod
    def holm_threshold(*, p_values: list[float], alpha: float = 0.05) -> float | None:
        """The largest p in `p_values` that survives Holm-Bonferroni at `alpha`, or None
        if none does.

        Reported rather than applied silently: ten bins are ten hypotheses, and quoting
        the smallest of ten raw p-values as though one test had been run is how a bin
        that is noise gets published as a crossover."""
        ordered = sorted(p_values)
        surviving = None
        for index, value in enumerate(ordered):
            if value <= alpha / (len(ordered) - index):
                surviving = value
            else:
                break
        return surviving

    @staticmethod
    def first_separating_bin(*, records: list[dict], alpha: float = 0.05) -> dict | None:
        """The earliest window whose informed rate beats its own control significantly AND
        whose advantage then persists -- every later window that was measured at all must
        also have a positive gap.

        The persistence clause is what makes this a crossover rather than a lucky window:
        one significant window in ten at alpha = 0.05 is roughly what chance produces,
        while a crossover is a change of regime. Three details decide what it means:

        * **The last window can never qualify.** With nothing after it there is no
          persistence evidence, and a vacuous `all([])` would return exactly the lone
          significant window the clause exists to exclude.
        * **Windows that measured nothing are skipped, not counted against.** An empty or
          one-armed window has `gap is None`, and treating that as a contradiction would
          let a gap in the data veto a real crossover.
        * **alpha is deliberately uncorrected here**, because persistence is doing the
          multiplicity work. The Holm threshold over the same windows is reported
          alongside so a reader can apply the stricter rule; the two are printed together
          and never presented as one number."""
        for index, record in enumerate(records):
            later = [entry for entry in records[index + 1 :] if entry["gap"] is not None]
            if not later:
                continue
            if record["p"] is None or record["p"] >= alpha or record["gap"] <= 0:
                continue
            if all(entry["gap"] > 0 for entry in later):
                return record
        return None

    @staticmethod
    def first_separating_band(*, records: list[dict], alpha: float = 0.05) -> dict | None:
        """The same rule, applied to the accumulated-landings bands instead of the
        transition windows.

        It exists so the page's central claim -- that the crossover is better described
        by accumulated successes than by elapsed transitions -- is adjudicated by the same
        statistic on both axes rather than by eye. Without it, one axis would have a
        decision rule and the other only a bar chart."""
        return TossingRoomSplitScaling.first_separating_bin(records=records, alpha=alpha)

    @staticmethod
    def stratified_by_transitions(
        *, draws: list[Draw], bin_width: int, landing_floor: int
    ) -> list[dict]:
        """Transition windows computed on draws that already have `landing_floor`
        landings behind them -- the cut `factorial_split` is too coarse to make.

        The 2x2 splits time at the median of all compared draws, which on a 25,000-
        transition run is around 13,000 -- more than five standard runs, and far coarser
        than the window where a crossover is actually found. A gap that is confined to
        the first window or two is invisible at that resolution. Holding the landing
        count at or above a floor and then windowing time finely is what tests whether
        elapsed transitions still matter once evidence is held fixed, which is the one
        question the factorial cannot answer on this domain: recycling takes almost
        exactly one attempt per period, so its landings are near-linear in its
        transitions and the two are confounded by construction."""
        return TossingRoomSplitScaling.binned_by_transitions(
            draws=[draw for draw in draws if draw.prior_landings >= landing_floor],
            bin_width=bin_width,
        )

    @staticmethod
    def split_by_separation(
        *, draws: list[Draw], tolerance: float = _THROW_TOLERANCE
    ) -> list[dict]:
        """Draws whose skill's own past landings already spanned more than `tolerance`,
        against those where they did not -- the diagnosis's mechanism claim, tested.

        This is the comparison `Draw.prior_separation` exists for. Binning by landing
        COUNT asks how much evidence the classifier had; binning by separation asks
        whether that evidence contained the thing it needs. One landing pins where the
        good force region sits for one target and any number of landings within a
        tolerance of each other is consistent with a single flat band, so the slope of
        the force/target relation is only present in the data at all once the span
        exceeds the tolerance."""
        return [
            TossingRoomSplitScaling.compare(
                draws=[draw for draw in draws if test(draw)], label=label
            )
            for label, test in (
                (f"landings span <= {tolerance}", lambda d: d.prior_separation <= tolerance),
                (f"landings span > {tolerance}", lambda d: d.prior_separation > tolerance),
            )
        ]

    # ------------------------------------------------------------------- the mechanism

    @staticmethod
    def separation_series(*, traces: list[dict], skill: str) -> dict[int, list[tuple[int, float]]]:
        """Per seed, `(transitions, widest gap between two landed targets so far)`.

        The maximum PAIRWISE gap over a set of reals on a line is just its range, so this
        is max - min over the landed targets, which is why it is computed directly rather
        than over pairs. Below `throw_tolerance` the landings are consistent with one
        flat band and the classifier has no slope to fit; above it, the tilt is present
        in the data whether or not the fit finds it. That makes this the mechanism check:
        it distinguishes "the evidence was never there" from "the evidence was there and
        the classifier could not use it".

        Per PERIOD rather than per draw, so the line is continuous across the long
        stretches in which recycling is not practised at all -- a per-draw series would
        compress those to nothing and make the curve look far steeper than the budget it
        actually consumed. Dated by the END of each period, since the separation includes
        that period's own landings."""
        TossingRoomSplitScaling.require_per_draw_record(traces=traces)
        TossingRoomSplitScaling.steps_per_period(traces=traces)
        series: dict[int, list[tuple[int, float]]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            landed_targets: list[float] = []
            points: list[tuple[int, float]] = []
            for index, period in enumerate(run["periods"]):
                tally = period["skills"].get(skill)
                if tally is not None:
                    for target, landed in zip(
                        tally["throw_targets"], tally["throw_landed_flags"], strict=True
                    ):
                        if landed:
                            landed_targets.append(target)
                spread = (
                    max(landed_targets) - min(landed_targets) if len(landed_targets) > 1 else 0.0
                )
                points.append((
                    TossingRoomSplitScaling.transitions_before(run=run, index=index + 1),
                    spread,
                ))
            series[run["seed"]] = points
        return series

    @staticmethod
    def landings_series(*, traces: list[dict], skill: str) -> dict[int, list[tuple[int, int]]]:
        """Per seed, `(transitions, cumulative landings so far)` -- the denominator behind
        every claim on this page, plotted so a reader can see which seeds carry it."""
        TossingRoomSplitScaling.require_per_draw_record(traces=traces)
        TossingRoomSplitScaling.steps_per_period(traces=traces)
        series: dict[int, list[tuple[int, int]]] = {}
        for run in TossingRoomSplitThrowRates.runs(traces=traces):
            total = 0
            points: list[tuple[int, int]] = []
            for index, period in enumerate(run["periods"]):
                tally = period["skills"].get(skill)
                if tally is not None:
                    total += sum(tally["throw_landed_flags"])
                points.append((
                    TossingRoomSplitScaling.transitions_before(run=run, index=index + 1),
                    total,
                ))
            series[run["seed"]] = points
        return series

    @staticmethod
    def crosses_tolerance_at(*, traces: list[dict], skill: str) -> dict[int, int | None]:
        """Per seed, the transition count at which its landed targets first span more
        than `throw_tolerance`, or None if they never do."""
        crossings: dict[int, int | None] = {}
        for seed, points in TossingRoomSplitScaling.separation_series(
            traces=traces, skill=skill
        ).items():
            crossings[seed] = next(
                (transitions for transitions, spread in points if spread > _THROW_TOLERANCE), None
            )
        return crossings

    @staticmethod
    def cumulative_by_seed(
        *, draws: list[Draw], kind: str, checkpoints: list[int]
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Per seed, `(transitions, landed, attempts)` cumulative up to each checkpoint.

        The per-seed view has to be cumulative: a single seed contributes only a handful
        of informed draws per 2,500-transition window, and a windowed per-seed rate would
        be a plot of 0/1 and 1/1. Cumulative traces show the spread honestly while the
        pooled windowed comparison carries the inference."""
        series: dict[int, list[tuple[int, int, int]]] = {}
        for seed in sorted({draw.seed for draw in draws}):
            own = [draw for draw in draws if draw.seed == seed and draw.kind == kind]
            points = []
            for checkpoint in checkpoints:
                upto = [draw for draw in own if draw.transitions < checkpoint]
                points.append((checkpoint, sum(draw.landed for draw in upto), len(upto)))
            series[seed] = points
        return series

    # ------------------------------------------------------------------------ printing

    @staticmethod
    def format_p(*, p: float | None) -> str:
        return "--" if p is None else TossingRoomSplitThrowRates.format_p_value(p=p)

    @staticmethod
    def _print_comparisons(*, records: list[dict], header: str, holm: bool = True) -> None:
        """One table, with the Holm-Bonferroni threshold under any table of more than one
        comparison.

        Every table here is a family of tests, not one test, so the correction belongs
        under all of them rather than only under the transition windows -- and the whole
        report runs once per skill, which the printed family size makes visible."""
        print(f"\n{header}")
        print(
            f"{'comparison':>22} | {'informed':>12} | {'random':>12} | {'gap pp':>8} | "
            f"{'floor pp':>9} | {'MDE pp':>7} | {'p':>10}"
        )
        for record in records:
            informed = f"{record['informed'][0]}/{record['informed'][1]}"
            random_ = f"{record['random'][0]}/{record['random'][1]}"
            gap = "--" if record["gap"] is None else f"{record['gap']:+.2f}"
            floor = "--" if record["noise_floor"] is None else f"{record['noise_floor']:.2f}"
            mde = "--" if record["mde"] is None else f"{record['mde']:.2f}"
            print(
                f"{record['label']:>22} | {informed:>12} | {random_:>12} | {gap:>8} | "
                f"{floor:>9} | {mde:>7} | {TossingRoomSplitScaling.format_p(p=record['p']):>10}"
            )
        p_values = [record["p"] for record in records if record["p"] is not None]
        if holm and len(p_values) > 1:
            threshold = TossingRoomSplitScaling.holm_threshold(p_values=p_values)
            print(
                f"  Holm-Bonferroni over the {len(p_values)} comparisons in this table: "
                + (
                    f"largest surviving p = {TossingRoomSplitScaling.format_p(p=threshold)}"
                    if threshold is not None
                    else "none survives at alpha = 0.05"
                )
            )

    @staticmethod
    def _print_crossover(*, record: dict | None, axis: str, units: str) -> None:
        """The crossover line, stating plainly that its alpha is uncorrected and that
        persistence rather than the correction is what guards it."""
        print(
            f"  first separating {axis} (raw alpha = 0.05 AND every later "
            f"{axis} still positive; see the Holm line above for the corrected view): "
            + (
                f"{record['label']} {units} "
                f"({record['informed'][0]}/{record['informed'][1]} vs "
                f"{record['random'][0]}/{record['random'][1]}, "
                f"{record['gap']:+.2f}pp, p = {TossingRoomSplitScaling.format_p(p=record['p'])})"
                if record is not None
                else f"none -- no {axis} separates and stays separated"
            )
        )

    @staticmethod
    def print_report(*, traces: list[dict], bin_width: int, landing_threshold: int) -> None:
        for skill in _SKILL_STYLES:
            draws = TossingRoomSplitScaling.draws(traces=traces, skill=skill)
            fallback_landed, fallback_total = TossingRoomSplitScaling.rate(
                draws=draws, kind="fallback"
            )
            overall = TossingRoomSplitScaling.compare(draws=draws, label="whole run")
            print(f"\n{'=' * 78}\n{skill}: {len(draws)} attempts across all seeds\n{'=' * 78}")
            print(
                f"  uniform fallback draws: {fallback_landed}/{fallback_total} landed "
                f"(neither informed nor epsilon-random; reported apart, never pooled)"
            )
            TossingRoomSplitScaling._print_comparisons(
                records=[overall], header="Pooled over the whole run"
            )

            binned = TossingRoomSplitScaling.binned_by_transitions(draws=draws, bin_width=bin_width)
            TossingRoomSplitScaling._print_comparisons(
                records=binned,
                header=f"By elapsed transitions ({bin_width}-wide windows, each one standard run)",
            )
            TossingRoomSplitScaling._print_crossover(
                record=TossingRoomSplitScaling.first_separating_bin(records=binned),
                axis="window",
                units="transitions",
            )

            by_landings = TossingRoomSplitScaling.binned_by_prior_landings(draws=draws)
            TossingRoomSplitScaling._print_comparisons(
                records=by_landings, header="By landings already accumulated (the mechanism)"
            )
            TossingRoomSplitScaling._print_crossover(
                record=TossingRoomSplitScaling.first_separating_band(records=by_landings),
                axis="band",
                units="landings",
            )

            TossingRoomSplitScaling._print_comparisons(
                records=TossingRoomSplitScaling.split_by_separation(draws=draws),
                header="By whether the past landings already spanned the throw tolerance",
            )

            TossingRoomSplitScaling._print_comparisons(
                records=TossingRoomSplitScaling.stratified_by_transitions(
                    draws=draws, bin_width=bin_width, landing_floor=landing_threshold
                )[:4],
                header=f"Transition windows among draws with >= {landing_threshold} "
                f"landings behind them (does the clock still matter at fixed evidence?)",
            )

            cells = TossingRoomSplitScaling.factorial_split(
                draws=draws, landing_threshold=landing_threshold
            )
            midpoint = int(cells[0]["midpoint"]) if cells else 0
            TossingRoomSplitScaling._print_comparisons(
                records=cells,
                header=f"Transitions x accumulated landings, crossed at {landing_threshold} "
                f"landings and at the compared draws' median of {midpoint} transitions",
            )

            crossings = TossingRoomSplitScaling.crosses_tolerance_at(traces=traces, skill=skill)
            reached = [value for value in crossings.values() if value is not None]
            print(
                f"\n  landed-target separation exceeds the {_THROW_TOLERANCE} tolerance in "
                f"{len(reached)}/{len(crossings)} seeds"
                + (
                    f"; median at {int(statistics.median(reached))} transitions, "
                    f"range {min(reached)}-{max(reached)}"
                    if reached
                    else ""
                )
            )

    @staticmethod
    def print_attempt_cap(*, traces: list[dict]) -> None:
        """The structural claim the whole design rests on: recycling is capped at one
        attempt per practice period by the layout, so scaling PERIODS is the only way to
        buy it more practice and scaling period LENGTH would buy trash more and recycling
        none. Verified here rather than assumed, because getting it backwards would
        produce a null result for an entirely unrelated reason."""
        print(f"\n{'=' * 78}\nAttempts per practice period\n{'=' * 78}")
        for skill in _SKILL_STYLES:
            histogram = TossingRoomSplitThrowRates.attempts_per_period_histogram(
                traces=traces, skill=skill
            )
            periods = sum(histogram.values())
            shown = ", ".join(f"{count} at {attempts}" for attempts, count in histogram.items())
            print(f"  {skill}: {shown}  (of {periods} periods)")
        recycling = TossingRoomSplitThrowRates.attempts_per_period_histogram(
            traces=traces, skill="ThrowRecycling"
        )
        above = sum(count for attempts, count in recycling.items() if attempts >= 2)
        total = sum(recycling.values())
        print(
            f"  ThrowRecycling periods with 2 or more attempts: {above}/{total} "
            + ("-- the cap holds" if above == 0 else "-- THE CAP IS BROKEN")
        )

    # ------------------------------------------------------------------------ figures

    @staticmethod
    def _plot_binned(*, ax, records: list[dict], key: str, color: str, label: str) -> None:
        centres = [(record["start"] + record["end"]) / 2 for record in records]
        rates = [
            100.0 * record[key][0] / record[key][1] if record[key][1] else float("nan")
            for record in records
        ]
        ax.plot(centres, rates, marker="o", color=color, label=label, linewidth=2)

    @staticmethod
    def render_scaling(*, traces: list[dict], bin_width: int, output: Path) -> None:
        """The figure that matters: each skill's informed land rate against its own
        epsilon-random control, as a function of budget, with per-seed spread."""
        figure, axes = plt.subplots(2, 2, figsize=(14, 9))
        # Derived from the traces rather than hardcoded to this run's 25,000: the same
        # figure has to be correct if the budget is changed again, and a hardcoded
        # endpoint would silently truncate a longer run's curve instead of failing.
        steps = TossingRoomSplitScaling.steps_per_period(traces=traces)
        horizon = max(shard["num_cycles"] for shard in traces) * steps
        checkpoints = list(range(bin_width, horizon + 1, bin_width))

        for row, skill in enumerate(_SKILL_STYLES):
            draws = TossingRoomSplitScaling.draws(traces=traces, skill=skill)
            binned = TossingRoomSplitScaling.binned_by_transitions(draws=draws, bin_width=bin_width)

            windowed = axes[row][0]
            for seed_series, color in (
                (
                    TossingRoomSplitScaling.cumulative_by_seed(
                        draws=draws, kind=_INFORMED, checkpoints=checkpoints
                    ),
                    "tab:green",
                ),
                (
                    TossingRoomSplitScaling.cumulative_by_seed(
                        draws=draws, kind=_RANDOM, checkpoints=checkpoints
                    ),
                    "tab:grey",
                ),
            ):
                for points in seed_series.values():
                    windowed.plot(
                        [point[0] for point in points],
                        [
                            100.0 * point[1] / point[2] if point[2] else float("nan")
                            for point in points
                        ],
                        color=color,
                        alpha=0.25,
                        linewidth=0.9,
                    )
            TossingRoomSplitScaling._plot_binned(
                ax=windowed,
                records=binned,
                key="informed",
                color="tab:green",
                label="informed (pooled, windowed)",
            )
            TossingRoomSplitScaling._plot_binned(
                ax=windowed,
                records=binned,
                key="random",
                color="tab:grey",
                label="epsilon-random control (pooled, windowed)",
            )
            windowed.set_title(
                f"{skill}: informed draws vs their own coin flip\n"
                f"bold = pooled per {bin_width}-transition window; thin = per-seed cumulative"
            )
            windowed.set_xlabel("online transitions")
            windowed.set_ylabel("landed / attempts (%)")
            windowed.set_ylim(-3, 103)
            windowed.legend(fontsize=8, loc="upper left")
            windowed.grid(alpha=0.3)

            gap_axis = axes[row][1]
            centres = [(record["start"] + record["end"]) / 2 for record in binned]
            # `nan`, never 0.0: a window with an empty arm has no measurement, and
            # plotting it as a gap of exactly zero inside a zero-width MDE band draws a
            # perfectly-measured null where nothing was measured at all. `nan` breaks the
            # line instead, which is the honest rendering of a hole.
            gaps = [float("nan") if record["gap"] is None else record["gap"] for record in binned]
            mdes = [float("nan") if record["mde"] is None else record["mde"] for record in binned]
            gap_axis.axhline(0, color="black", linewidth=1)
            gap_axis.fill_between(
                centres,
                [-value for value in mdes],
                mdes,
                color="tab:red",
                alpha=0.12,
                label="below the window's own MDE (80% power)",
            )
            gap_axis.plot(
                centres, gaps, marker="o", color=_SKILL_STYLES[skill][0], linewidth=2, label="gap"
            )
            for centre, record in zip(centres, binned, strict=True):
                if record["p"] is not None and record["p"] < 0.05:
                    gap_axis.annotate(
                        "*",
                        (centre, record["gap"]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=14,
                    )
                    gap_axis.annotate(
                        f"{record['informed'][0]}/{record['informed'][1]}",
                        (centre, record["gap"]),
                        textcoords="offset points",
                        xytext=(0, -16),
                        ha="center",
                        fontsize=6,
                    )
            gap_axis.set_title(
                f"{skill}: informed minus control, per window\n"
                "* marks p < 0.05 (raw; see the Holm threshold in the printed table)"
            )
            gap_axis.set_xlabel("online transitions")
            gap_axis.set_ylabel("gap (percentage points)")
            gap_axis.legend(fontsize=8, loc="upper left")
            gap_axis.grid(alpha=0.3)

        figure.suptitle(
            "Tossing Room (split throws), 10x budget: does more practice fix "
            "ThrowRecycling's sampler?",
            fontsize=13,
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"wrote {output}")

    @staticmethod
    def render_mechanism(*, traces: list[dict], landing_threshold: int, output: Path) -> None:
        """The mechanism check: is the slope of the force/target relation even visible in
        each seed's own successes, and does the gap track that rather than the clock?"""
        figure, axes = plt.subplots(2, 2, figsize=(14, 9))

        spread_axis = axes[0][0]
        for skill, (color, style) in _SKILL_STYLES.items():
            series = TossingRoomSplitScaling.separation_series(traces=traces, skill=skill)
            for index, points in enumerate(series.values()):
                spread_axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linestyle=style,
                    alpha=0.45,
                    linewidth=1.0,
                    label=skill if index == 0 else None,
                )
        spread_axis.axhline(
            _THROW_TOLERANCE,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"throw tolerance ({_THROW_TOLERANCE})",
        )
        spread_axis.set_title(
            "Widest separation between two landed targets, per seed\n"
            "below the tolerance the successes are consistent with one flat band"
        )
        spread_axis.set_xlabel("online transitions")
        spread_axis.set_ylabel("max landed-target separation")
        spread_axis.legend(fontsize=8, loc="lower right")
        spread_axis.grid(alpha=0.3)

        landings_axis = axes[0][1]
        for skill, (color, style) in _SKILL_STYLES.items():
            series = TossingRoomSplitScaling.landings_series(traces=traces, skill=skill)
            for index, points in enumerate(series.values()):
                landings_axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    linestyle=style,
                    alpha=0.45,
                    linewidth=1.0,
                    label=skill if index == 0 else None,
                )
        landings_axis.set_title("Cumulative landings per seed (the classifier's positives)")
        landings_axis.set_xlabel("online transitions")
        landings_axis.set_ylabel("landings so far")
        landings_axis.legend(fontsize=8, loc="upper left")
        landings_axis.grid(alpha=0.3)

        for column, skill in enumerate(_SKILL_STYLES):
            axis = axes[1][column]
            draws = TossingRoomSplitScaling.draws(traces=traces, skill=skill)
            records = TossingRoomSplitScaling.binned_by_prior_landings(draws=draws)
            positions = range(len(records))
            width = 0.38
            for offset, key, color, label in (
                (-width / 2, "informed", "tab:green", "informed"),
                (width / 2, "random", "tab:grey", "epsilon-random control"),
            ):
                axis.bar(
                    [position + offset for position in positions],
                    [
                        100.0 * record[key][0] / record[key][1] if record[key][1] else 0.0
                        for record in records
                    ],
                    width=width,
                    color=color,
                    label=label,
                )
                for position, record in zip(positions, records, strict=True):
                    axis.annotate(
                        f"{record[key][0]}/{record[key][1]}",
                        (
                            position + offset,
                            100.0 * record[key][0] / record[key][1] if record[key][1] else 0.0,
                        ),
                        textcoords="offset points",
                        xytext=(0, 3),
                        ha="center",
                        fontsize=6,
                        rotation=90,
                    )
            axis.set_xticks(list(positions))
            axis.set_xticklabels([record["label"] for record in records])
            axis.set_title(f"{skill}: land rate by landings already accumulated")
            axis.set_xlabel("landings behind this draw")
            axis.set_ylabel("landed / attempts (%)")
            axis.set_ylim(0, 118)
            axis.legend(fontsize=8, loc="upper left")
            axis.grid(alpha=0.3, axis="y")

        figure.suptitle(
            "Is the force/target slope visible in each seed's own successes, and does "
            "the sampler track it?",
            fontsize=13,
        )
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism-output", type=Path, required=True)
    parser.add_argument(
        "--bin-width",
        type=int,
        default=_DEFAULT_BIN_WIDTH,
        help="Transitions per comparison window. The default is one standard run, so "
        "the first window replicates the published null at the published power.",
    )
    parser.add_argument(
        "--landing-threshold",
        type=int,
        default=2,
        help="Where the transitions x accumulated-landings 2x2 is cut. The default is 2 "
        "because the diagnosis says one landing pins the good region's location and two "
        "are the minimum that can pin its slope.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    traces = [json.loads(path.read_text()) for path in args.traces]
    problems = TossingRoomSplitThrowRates.check_against_sweep(
        traces=traces, results_root=args.results_root
    )
    if problems:
        print("consistency gate FAILED -- refusing to report:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    seeds = len(TossingRoomSplitThrowRates.runs(traces=traces))
    print(f"consistency gate: all {seeds} traced seeds reproduce their swept stats.json exactly")

    TossingRoomSplitScaling.print_attempt_cap(traces=traces)
    TossingRoomSplitScaling.print_report(
        traces=traces, bin_width=args.bin_width, landing_threshold=args.landing_threshold
    )
    TossingRoomSplitScaling.render_scaling(
        traces=traces, bin_width=args.bin_width, output=args.output
    )
    TossingRoomSplitScaling.render_mechanism(
        traces=traces, landing_threshold=args.landing_threshold, output=args.mechanism_output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
