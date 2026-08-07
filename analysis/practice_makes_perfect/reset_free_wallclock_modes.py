"""Post-run analysis for one thing the reset-free cycle-budget experiment reported as a
single number: the one-way `never` cell is **two populations, not one**.

**Background.** `docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget.md` (PR #166)
ran a budget x ledge x policy cube on Tossing Room and concluded that the reset-free arm
is *stranded* rather than starved: under the one-way ledge it logs 207 effective practice
attempts at 100 cycles, the identical 207 it logs at 10. Its cost paragraph then quoted a
per-run figure for that cell, and the quoted figures turned out to be medians labelled as
means. Chasing that label is what surfaced the real point: the mean is misleading there
for a reason deeper than arithmetic, because the cell's runs fall into two separated
groups and no single-number summary describes either.

**What this module establishes, and in which direction.** The predictor is
**stranding time**, read off `stats.json`; the outcome is **wall clock**, read off
`timing.json`. Those are different files produced by different mechanisms, which is what
keeps the agreement between the two partitions a finding rather than an identity --
`test_the_two_partitions_are_read_from_different_files` pins exactly that.

**The confound that had to be excluded first.** Wall clock is not a pure measure of work:
these runs shared a machine with other sweeps. `timing.json` records both concurrency
signals (`sweep_runs_in_flight_*` and a machine-wide `cli_processes` / load sample), so
each is tested rather than eyeballed. The decisive control is not any of those p-values
though -- it is that the **same seed-for-seed partition reproduces in a physically
separate sweep**, and at 1x all ten runs started in the same second under an identical
machine state, so there is no per-seed concurrency difference left to carry it.

**One residual this does not explain, and does not paper over.** Total practice attempts
are identical across all ten seeds, and one slow seed logs *fewer* planning attempts than
every fast seed while running 2.3x longer. So "more effective attempts" identifies the
*group*, but the per-second mechanism turning them into wall clock is not established
here. That is stated as a residual, not smoothed into a story.

**Statistics.** `PairedTests.sign_flip` for the cross-budget change (the cells share
seeds), `UnpairedTests.fisher_exact` for the agreement between the two partitions -- both
imported rather than reimplemented, since a second copy of a test is how a sign error gets
published. The concurrency covariates need a two-sample test on continuous values, which
neither helper covers, so `_permutation_p_value` enumerates all `C(10, 4) = 210`
relabellings in full: exact, no normal approximation, no scipy.

Stranding onsets come from `PickupWeightStranding`, reused rather than recopied: it
already encodes that stranding is **terminal-from-here** rather than "the first cycle
with no pile access", which a fresh implementation gets wrong.

Reads only already-produced output (CLAUDE.md's `analysis/` convention -- this never runs
a simulation or drives a `Method`). Each `--budget` points at the directory holding that
budget's `<seed>/stats.json`, because the committed sweeps nest differently
(`<cell>/<seed>` for the 10x runs, `<policy>/ees/<seed>` for the 1x ones) and guessing
between them is how the wrong sweep gets read.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import ClassVar

import matplotlib
from pydantic import BaseModel, ConfigDict

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.practice_makes_perfect.paired_tests import PairedTests  # noqa: E402
from analysis.practice_makes_perfect.pickup_weight_stranding import (  # noqa: E402
    PickupWeightStranding,
)
from analysis.practice_makes_perfect.tossing3d_ees_arms import UnpairedTests  # noqa: E402

# Skills that require the robot to be at the item pile, the same pair
# `reset_free_cycle_budget` counts. A stranded robot can still walk (`MoveRoom`) and press
# buttons (`Press*`) for a whole cycle, so those are excluded -- counting them would
# report a starved arm as busy.
_EFFECTIVE_PREFIXES = ("Pickup", "Throw")

# Colour carries the mode, marker the budget, so neither identity rests on hue alone.
# Chosen from the same colourblind-safe set the other figures in this folder use.
_MODE_COLORS = {"late": "#0072B2", "early": "#D55E00"}
_MODE_LABELS = {
    "early": "stranded in cycle 1",
    "late": "stranded in cycle 2 or 3",
}
_BUDGET_MARKERS = {"1x": "o", "10x": "s"}

# Cycles per budget, for the x tick labels only. Display text: the reader of a figure has
# no directory name in front of them, and "1x" alone does not say how many cycles that is.
_CYCLES = {"1x": 10, "10x": 100}

# The concurrency covariates `timing.json` records, and what each is called in the report.
# Every one of them is tested; none of them separates the modes.
_COVARIATES: tuple[tuple[str, str], ...] = (
    ("runs_in_flight_at_start", "this sweep's in-flight runs at start"),
    ("runs_in_flight_at_end", "this sweep's in-flight runs at end"),
    ("cli_processes_at_start", "machine-wide hitl_pmp.cli processes at start"),
    ("cli_processes_at_end", "machine-wide hitl_pmp.cli processes at end"),
    ("load_at_start", "1-minute load average at start"),
    ("load_at_end", "1-minute load average at end"),
)


class SeedRun(BaseModel):
    """One run, reduced to what separates the two modes and what might confound it."""

    model_config = ConfigDict(frozen=True)

    seed: int
    # From timing.json -- the outcome side.
    elapsed_seconds: float
    # From stats.json -- the predictor side.
    effective_attempts: int
    stranding_onset: int | None
    final_solved: int
    num_test_tasks: int
    # The learning curve, one entry per evaluation checkpoint (checkpoint 0 is taken
    # before any practice). Kept as two parallel lists rather than one list of pairs so
    # that either can serve as an x axis without unzipping at every call site.
    solved_per_checkpoint: list[int]
    transitions_per_checkpoint: list[int]
    # From timing.json -- the confound side.
    runs_in_flight_at_start: int
    runs_in_flight_at_end: int
    cli_processes_at_start: int
    cli_processes_at_end: int
    load_at_start: float
    load_at_end: float


class EffectiveAttemptChange(BaseModel):
    """The per-seed change in effective practice attempts between two budgets, with its
    exact paired p. Kept as a model rather than a bare tuple so "every difference was
    exactly zero" arrives with the test that says so."""

    model_config = ConfigDict(frozen=True)

    differences: list[int]
    p_value: float

    @property
    def num_unchanged(self) -> int:
        return sum(1 for difference in self.differences if difference == 0)


class ResetFreeWallclockModes:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # A run is "late-stranded" if its last pile access falls in cycle 2 or later. Cycle
    # indices here are 1-based to match the experiment log's prose, which is also what
    # `PickupWeightStranding.stranding_onset` returns for a terminal suffix.
    FIRST_LATE_ONSET: ClassVar[int] = 2

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_budgets(*, directories: dict[str, Path]) -> dict[str, list[SeedRun]]:
        """Each budget's per-seed runs, with the pairing check first.

        Every cross-budget statement below is per-seed, so unequal seed sets leave the
        pairing undefined; a reader that zipped them would silently compare one seed's
        1x run against another's 10x one.
        """
        budgets = {
            label: ResetFreeWallclockModes.read_budget(directory=directory)
            for label, directory in directories.items()
        }
        seed_sets = {label: sorted(run.seed for run in runs) for label, runs in budgets.items()}
        distinct = {tuple(seeds) for seeds in seed_sets.values()}
        if len(distinct) > 1:
            listed = "; ".join(f"{label}: {seeds}" for label, seeds in seed_sets.items())
            raise ValueError(
                f"budgets must run the same seeds, got {listed}. Every comparison here is "
                "paired within a seed, so an unequal seed set makes the pairing undefined."
            )
        return budgets

    @staticmethod
    def read_budget(*, directory: Path) -> list[SeedRun]:
        """One budget's runs, read from `directory/<seed>/`, in seed order."""
        seeds = sorted(int(path.parent.name) for path in directory.glob("*/stats.json"))
        if not seeds:
            raise ValueError(f"no <seed>/stats.json under {directory}")
        return [
            ResetFreeWallclockModes.read_run(directory=directory / str(seed), seed=seed)
            for seed in seeds
        ]

    @staticmethod
    def read_run(*, directory: Path, seed: int) -> SeedRun:
        """One run. A missing file raises: a reader that skips one silently reports a
        9-seed result as a 10-seed one."""
        timing_path = directory / "timing.json"
        if not timing_path.exists():
            raise FileNotFoundError(f"seed {seed}: no timing.json at {timing_path}")
        timing = json.loads(timing_path.read_text())
        stats = json.loads((directory / "stats.json").read_text())
        stranding = PickupWeightStranding.read_run(path=directory / "stats.json", seed=seed)
        periods = PickupWeightStranding.practice_periods(stats=stats)
        evaluations = stats["evaluations"]
        final = evaluations[-1]
        return SeedRun(
            seed=seed,
            elapsed_seconds=float(timing["elapsed_seconds"]),
            effective_attempts=sum(
                int(tally.get("num_attempts", 0))
                for period in periods
                for name, tally in period.items()
                if name.startswith(_EFFECTIVE_PREFIXES)
            ),
            stranding_onset=stranding.stranding_onset,
            final_solved=int(final[1]),
            num_test_tasks=int(final[2]),
            solved_per_checkpoint=[int(entry[1]) for entry in evaluations],
            transitions_per_checkpoint=[int(entry[0]) for entry in evaluations],
            runs_in_flight_at_start=int(timing["sweep_runs_in_flight_at_start"]),
            runs_in_flight_at_end=int(timing["sweep_runs_in_flight_at_end"]),
            cli_processes_at_start=int(timing["machine_at_start"]["cli_processes"]),
            cli_processes_at_end=int(timing["machine_at_end"]["cli_processes"]),
            load_at_start=float(timing["machine_at_start"]["load_average_1min"]),
            load_at_end=float(timing["machine_at_end"]["load_average_1min"]),
        )

    # ------------------------------------------------------------------ the two partitions

    @staticmethod
    def is_late_stranded(*, run: SeedRun) -> bool:
        """Stranded in cycle 2 or later -- including never stranded at all, which is
        "later" taken to its limit and which no run in this cell exhibits."""
        onset = run.stranding_onset
        return onset is None or onset >= ResetFreeWallclockModes.FIRST_LATE_ONSET

    @staticmethod
    def late_stranded_seeds(*, runs: list[SeedRun]) -> set[int]:
        """The partition taken off `stats.json`. Nothing from `timing.json` enters here."""
        return {run.seed for run in runs if ResetFreeWallclockModes.is_late_stranded(run=run)}

    @staticmethod
    def largest_gap(*, runs: list[SeedRun]) -> tuple[float, float]:
        """`(threshold, gap)` at the widest gap between consecutive sorted wall clocks.

        Data-derived rather than a hardcoded number of seconds, so the same partition
        survives a change of budget that scales every run by roughly 10x. The threshold
        is the midpoint of that gap.
        """
        elapsed = sorted(run.elapsed_seconds for run in runs)
        if len(elapsed) < 2:
            raise ValueError("a wall-clock split needs at least two runs")
        lower, upper = max(
            zip(elapsed, elapsed[1:], strict=False), key=lambda pair: pair[1] - pair[0]
        )
        return (lower + upper) / 2, upper - lower

    @staticmethod
    def slow_seeds(*, runs: list[SeedRun]) -> set[int]:
        """The partition taken off `timing.json`. Nothing from `stats.json` enters here."""
        threshold, _ = ResetFreeWallclockModes.largest_gap(runs=runs)
        return {run.seed for run in runs if run.elapsed_seconds > threshold}

    @staticmethod
    def mode_of(*, run: SeedRun) -> str:
        return "late" if ResetFreeWallclockModes.is_late_stranded(run=run) else "early"

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def agreement_table(*, runs: list[SeedRun]) -> tuple[tuple[int, int], tuple[int, int]]:
        """`((late & slow, late & fast), (early & slow, early & fast))`."""
        slow = ResetFreeWallclockModes.slow_seeds(runs=runs)
        late = ResetFreeWallclockModes.late_stranded_seeds(runs=runs)
        seeds = {run.seed for run in runs}
        return (
            (len(late & slow), len(late - slow)),
            (len((seeds - late) & slow), len((seeds - late) - slow)),
        )

    @staticmethod
    def agreement_p_value(*, runs: list[SeedRun]) -> float:
        """Two-sided Fisher exact on the agreement table.

        With a 6/4 margin the floor is `1 / C(10, 4) = 1/210`: under the two-sided
        total-probability convention only the single most extreme table clears the
        observed probability, so the usual doubling does not apply. Quoting the floor is
        what stops "all four slow seeds are the four late-stranded ones" reading as
        stronger evidence than ten seeds can carry -- p = 1/210 is the *best* this design
        can do, not a measure of how large the effect is.
        """
        return UnpairedTests.fisher_exact(table=ResetFreeWallclockModes.agreement_table(runs=runs))

    @staticmethod
    def effective_attempt_change(
        *, budgets: dict[str, list[SeedRun]], earlier: str, later: str
    ) -> EffectiveAttemptChange:
        """Per-seed `later - earlier` effective attempts, with its exact paired p.

        Zero differences are kept, never dropped: "every seed moved by exactly zero" is
        the finding, not a nuisance.
        """
        before = {run.seed: run.effective_attempts for run in budgets[earlier]}
        after = {run.seed: run.effective_attempts for run in budgets[later]}
        differences = [after[seed] - before[seed] for seed in sorted(before)]
        test = PairedTests.sign_flip(differences=[float(d) for d in differences])
        return EffectiveAttemptChange(differences=differences, p_value=test.p_value)

    @staticmethod
    def covariate_p_value(*, runs: list[SeedRun], name: str) -> float:
        """Exact two-sided permutation p for a covariate differing between the two modes.

        A null here does **not** establish that concurrency is absent -- with ten seeds
        almost nothing would. The load-bearing control is the cross-sweep reproduction;
        this is reported so the claim carries a number rather than an eyeballed overlap.
        """
        late = [
            getattr(run, name) for run in runs if ResetFreeWallclockModes.is_late_stranded(run=run)
        ]
        early = [
            getattr(run, name)
            for run in runs
            if not ResetFreeWallclockModes.is_late_stranded(run=run)
        ]
        return ResetFreeWallclockModes._permutation_p_value(group_a=late, group_b=early)

    @staticmethod
    def _permutation_p_value(*, group_a: list[float], group_b: list[float]) -> float:
        """Two-sided exact permutation test on the difference in means.

        Enumerates every way of splitting the pooled values into groups of the observed
        sizes -- `C(10, 4) = 210` here -- and counts how many reach a difference at least
        as large in absolute value. Exact by construction, so no normal approximation, no
        tie correction, and no scipy (not a dependency of this project).
        """
        pooled = list(group_a) + list(group_b)
        size = len(group_a)
        if size == 0 or len(group_b) == 0:
            return 1.0
        observed = abs(statistics.fmean(group_a) - statistics.fmean(group_b))
        total = sum(pooled)
        count = 0
        splits = 0
        for combo in itertools.combinations(range(len(pooled)), size):
            splits += 1
            head = sum(pooled[i] for i in combo)
            difference = abs(head / size - (total - head) / (len(pooled) - size))
            if difference >= observed - 1e-9:
                count += 1
        return count / splits

    @staticmethod
    def format_count(*, numerator: int, denominator: int) -> str:
        """`x/y`, never a bare percentage: the denominators here are small and uneven."""
        return f"{numerator}/{denominator}"

    @staticmethod
    def curves_by_mode(*, runs: list[SeedRun]) -> dict[str, list[SeedRun]]:
        """The runs split into the two modes, keyed `"early"` / `"late"`.

        Deliberately routed through `mode_of` rather than through a hardcoded seed list:
        the figure and the statistics must not be able to drift apart, and a seed list
        copied into a plotting function is exactly how they would.
        """
        grouped: dict[str, list[SeedRun]] = {"early": [], "late": []}
        for run in runs:
            grouped[ResetFreeWallclockModes.mode_of(run=run)].append(run)
        return grouped

    @staticmethod
    def transition_step(*, runs: list[SeedRun]) -> int | None:
        """The constant online-transition step between checkpoints, or `None` when it is
        not constant across every checkpoint of every run.

        Whether "by transitions" is merely "by cycle" rescaled is a property of the
        measured data, not of the domain -- an episode ending early would break it. The
        figure's caption turns on the answer, so it is computed rather than assumed.
        """
        steps = {
            later - earlier
            for run in runs
            for earlier, later in zip(
                run.transitions_per_checkpoint,
                run.transitions_per_checkpoint[1:],
                strict=False,
            )
        }
        return steps.pop() if len(steps) == 1 else None

    # ------------------------------------------------------------------ the figure

    @staticmethod
    def render(*, budgets: dict[str, list[SeedRun]], output: Path) -> None:
        """Four panels, every one of them per-seed.

        A bar chart of two means is the specific thing this figure exists to avoid: with
        a 6/4 split, an aggregate describes neither group and hides that there are two.
        """
        labels = list(budgets)
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
        ResetFreeWallclockModes._panel_attempts(axes=axes[0][0], budgets=budgets, labels=labels)
        ResetFreeWallclockModes._panel_wallclock(axes=axes[0][1], budgets=budgets, labels=labels)
        ResetFreeWallclockModes._panel_solved(axes=axes[1][0], budgets=budgets, labels=labels)
        ResetFreeWallclockModes._panel_concurrency(axes=axes[1][1], budgets=budgets, labels=labels)
        handles = [
            Line2D([], [], color=_MODE_COLORS[mode], marker="o", linestyle="-", label=text)
            for mode, text in _MODE_LABELS.items()
        ] + [
            Line2D(
                [],
                [],
                color="#444444",
                marker=_BUDGET_MARKERS.get(label, "^"),
                linestyle="none",
                label=f"{label} budget",
            )
            for label in labels
        ]
        figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        figure.suptitle(
            "One-way reset-free practice is two populations, not one: "
            "the wall-clock split is the stranding split",
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)

    @staticmethod
    def _panel_attempts(*, axes, budgets: dict[str, list[SeedRun]], labels: list[str]) -> None:
        """Per-seed slopegraph across budgets. Ten flat lines is the whole point: ten
        times the cycles bought zero additional effective attempts on any seed.

        Seeds that coincide are annotated as one group rather than as stacked labels --
        with six seeds on the same value, per-seed text overlaps into unreadability and
        hides that the six ARE identical, which is itself the finding.
        """
        positions = list(range(len(labels)))
        groups: dict[tuple[int, ...], list[int]] = {}
        for run in budgets[labels[0]]:
            series = tuple(
                next(r.effective_attempts for r in budgets[label] if r.seed == run.seed)
                for label in labels
            )
            groups.setdefault(series, []).append(run.seed)
        total_seeds = len(budgets[labels[0]])
        for series, seeds in sorted(groups.items()):
            run = next(r for r in budgets[labels[0]] if r.seed == seeds[0])
            colour = _MODE_COLORS[ResetFreeWallclockModes.mode_of(run=run)]
            axes.plot(positions, series, color=colour, marker="o", linewidth=1.6, alpha=0.9)
            listed = ", ".join(str(seed) for seed in seeds)
            share = ResetFreeWallclockModes.format_count(
                numerator=len(seeds), denominator=total_seeds
            )
            axes.annotate(
                f"seed{'s' if len(seeds) > 1 else ''} {listed}\n({share} of the cell)",
                (len(labels) - 1, series[-1]),
                textcoords="offset points",
                xytext=(9, -4),
                fontsize=8,
                color=colour,
                va="center",
            )
        axes.set_xticks(positions)
        axes.set_xticklabels([f"{label}\n({_CYCLES.get(label, '?')} cycles)" for label in labels])
        axes.set_xlim(-0.25, len(labels) - 0.35)
        axes.set_ylabel("effective practice attempts over the whole run\n(Pickup*/Throw*)")
        axes.set_title("(a) Ten times the budget bought nothing, on every seed")
        axes.grid(axis="y", alpha=0.25)

    @staticmethod
    def _panel_wallclock(*, axes, budgets: dict[str, list[SeedRun]], labels: list[str]) -> None:
        """Wall clock against stranding cycle, per seed, both budgets on one log axis --
        the budgets differ by roughly 10x in seconds and the same gap opens in both."""
        for label in labels:
            runs = budgets[label]
            for index, run in enumerate(runs):
                onset = run.stranding_onset
                axes.scatter(
                    (onset if onset is not None else 0) + 0.10 * ((index % 3) - 1),
                    run.elapsed_seconds,
                    color=_MODE_COLORS[ResetFreeWallclockModes.mode_of(run=run)],
                    marker=_BUDGET_MARKERS.get(label, "^"),
                    s=58,
                    zorder=3,
                )
            threshold, gap = ResetFreeWallclockModes.largest_gap(runs=runs)
            slow = ResetFreeWallclockModes.slow_seeds(runs=runs)
            axes.axhline(threshold, color="#777777", linestyle=(0, (3, 3)), linewidth=1.0)
            share = ResetFreeWallclockModes.format_count(numerator=len(slow), denominator=len(runs))
            axes.annotate(
                f"{label}: widest gap {gap:.0f} s, {share} above it",
                (0.015, threshold),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color="#555555",
                va="bottom",
            )
        axes.set_yscale("log")
        axes.set_xlabel("cycle of the last effective practice attempt")
        axes.set_ylabel("wall clock (s, log scale)")
        axes.set_xticks([1, 2, 3])
        axes.set_xlim(0.5, 3.5)
        axes.set_title("(b) The wall-clock modes are the stranding cycles")
        axes.grid(axis="y", alpha=0.25, which="both")

    @staticmethod
    def _panel_solved(*, axes, budgets: dict[str, list[SeedRun]], labels: list[str]) -> None:
        """Final-checkpoint tasks solved, per seed, by mode. The empty band between the
        two groups is the scientific content: the pooled figure is a mixture of two
        populations, not a performance level."""
        total = budgets[labels[0]][0].num_test_tasks
        for offset, label in enumerate(labels):
            runs = budgets[label]
            for index, run in enumerate(runs):
                mode = ResetFreeWallclockModes.mode_of(run=run)
                axes.scatter(
                    offset + (0.18 if mode == "late" else -0.18) + 0.045 * ((index % 3) - 1),
                    run.final_solved,
                    color=_MODE_COLORS[mode],
                    marker=_BUDGET_MARKERS.get(label, "^"),
                    s=58,
                    zorder=3,
                )
            # The early group's label hangs below its lowest point and the late group's
            # above its highest, so neither lands in the empty band between them -- which
            # is where the band's own annotation goes.
            for mode, anchor, shift in (("early", min, -26), ("late", max, 16)):
                values = [
                    r.final_solved for r in runs if ResetFreeWallclockModes.mode_of(run=r) == mode
                ]
                axes.annotate(
                    ResetFreeWallclockModes.format_count(
                        numerator=sum(values), denominator=total * len(values)
                    )
                    + f"\npooled over {len(values)} seeds",
                    (offset + (0.18 if mode == "late" else -0.18), anchor(values)),
                    textcoords="offset points",
                    xytext=(0, shift),
                    ha="center",
                    fontsize=8,
                    color=_MODE_COLORS[mode],
                )
            early = [
                r.final_solved for r in runs if ResetFreeWallclockModes.mode_of(run=r) == "early"
            ]
            late = [
                r.final_solved for r in runs if ResetFreeWallclockModes.mode_of(run=r) == "late"
            ]
            if early and late and max(early) < min(late):
                axes.fill_between(
                    [offset - 0.45, offset + 0.45],
                    max(early),
                    min(late),
                    color="#999999",
                    alpha=0.16,
                    zorder=1,
                )
                axes.annotate(
                    f"no overlap: {max(early)}/{total} to {min(late)}/{total} empty",
                    (offset - 0.43, (max(early) + min(late)) / 2),
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#555555",
                )
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels([f"{label} budget" for label in labels])
        axes.set_xlim(-0.65, len(labels) - 0.35)
        axes.set_ylim(-4, total + 1)
        axes.set_ylabel(f"tasks solved at the final checkpoint (x/{total})")
        axes.set_title("(c) The modes do not overlap on score either")
        axes.grid(axis="y", alpha=0.25)

    @staticmethod
    def _panel_concurrency(*, axes, budgets: dict[str, list[SeedRun]], labels: list[str]) -> None:
        """The confound, answered rather than asserted: wall clock against the machine
        load each run actually started against.

        The 1x sweep is the control that matters. All ten of its runs launched in the
        same second at `--max-workers 10`, so every one of them met an identical machine
        state -- and the same four seeds still came out slow. No per-seed difference in
        starting concurrency exists there to carry the split.
        """
        for label in labels:
            runs = budgets[label]
            for run in runs:
                axes.scatter(
                    run.load_at_start,
                    run.elapsed_seconds,
                    color=_MODE_COLORS[ResetFreeWallclockModes.mode_of(run=run)],
                    marker=_BUDGET_MARKERS.get(label, "^"),
                    s=58,
                    zorder=3,
                )
            starts = {round(run.load_at_start, 4) for run in runs}
            slow = ResetFreeWallclockModes.slow_seeds(runs=runs)
            identical = len(starts) == 1
            share = ResetFreeWallclockModes.format_count(numerator=len(slow), denominator=len(runs))
            note = (
                f"{label}: all {len(runs)} runs launched in the same second\n"
                f"against an identical load of {next(iter(starts)):.2f} -- and still\n"
                f"{share} came out slow"
                if identical
                else f"{label}: load {min(starts):.1f}-{max(starts):.1f} at start,\n"
                f"exact permutation p = "
                f"{ResetFreeWallclockModes.covariate_p_value(runs=runs, name='load_at_start'):.2f}"
                "\nbetween the two modes"
            )
            axes.annotate(
                note,
                (
                    min(run.load_at_start for run in runs),
                    max(run.elapsed_seconds for run in runs),
                ),
                textcoords="offset points",
                xytext=(14, 22) if identical else (0, 26),
                ha="left",
                fontsize=8,
                color="#444444",
            )
        axes.set_yscale("log")
        axes.set_xlabel("1-minute load average when the run started")
        axes.set_ylabel("wall clock (s, log scale)")
        axes.set_title("(d) Concurrency does not sort the modes")
        axes.set_ylim(top=2400)
        axes.grid(alpha=0.25, which="both")

    # ------------------------------------------------------------------ the report

    @staticmethod
    def report(*, budgets: dict[str, list[SeedRun]]) -> str:
        """Everything the log entry quotes, as `x/y` counts and exact p-values."""
        labels = list(budgets)
        lines: list[str] = []
        for label in labels:
            runs = budgets[label]
            slow = ResetFreeWallclockModes.slow_seeds(runs=runs)
            late = ResetFreeWallclockModes.late_stranded_seeds(runs=runs)
            threshold, gap = ResetFreeWallclockModes.largest_gap(runs=runs)
            elapsed = [run.elapsed_seconds for run in runs]
            count = ResetFreeWallclockModes.format_count
            agreed = len(slow & late) + len(runs) - len(slow | late)
            lines.append(f"=== {label} budget, {len(runs)} runs ===")
            lines.append(
                f"  wall clock  median {statistics.median(elapsed):.1f} s, "
                f"mean {statistics.fmean(elapsed):.1f} s, "
                f"widest gap {gap:.1f} s at {threshold:.1f} s"
            )
            lines.append(
                "  slow seeds        "
                f"{count(numerator=len(slow), denominator=len(runs))} {sorted(slow)}"
            )
            lines.append(
                "  late-stranded     "
                f"{count(numerator=len(late), denominator=len(runs))} {sorted(late)}"
            )
            lines.append(
                "  partitions agree  "
                f"{count(numerator=agreed, denominator=len(runs))}, Fisher exact p = "
                f"{ResetFreeWallclockModes.agreement_p_value(runs=runs):.5f}"
            )
            for run in runs:
                solved = count(numerator=run.final_solved, denominator=run.num_test_tasks)
                lines.append(
                    f"    seed {run.seed}: {run.elapsed_seconds:8.1f} s, "
                    f"{run.effective_attempts:3d} effective attempts, "
                    f"last effective cycle {run.stranding_onset}, solved {solved}"
                )
            lines.append("  concurrency covariates (exact permutation, late against early):")
            for name, description in _COVARIATES:
                p_value = ResetFreeWallclockModes.covariate_p_value(runs=runs, name=name)
                late_values = [
                    getattr(r, name)
                    for r in runs
                    if ResetFreeWallclockModes.mode_of(run=r) == "late"
                ]
                early_values = [
                    getattr(r, name)
                    for r in runs
                    if ResetFreeWallclockModes.mode_of(run=r) == "early"
                ]
                lines.append(
                    f"    {description}: late {min(late_values):g}-{max(late_values):g}, "
                    f"early {min(early_values):g}-{max(early_values):g}, p = {p_value:.4f}"
                )
        if len(labels) >= 2:
            change = ResetFreeWallclockModes.effective_attempt_change(
                budgets=budgets, earlier=labels[0], later=labels[-1]
            )
            lines.append(f"=== {labels[0]} -> {labels[-1]} effective-attempt change, per seed ===")
            lines.append(f"  differences {change.differences}")
            unchanged = ResetFreeWallclockModes.format_count(
                numerator=change.num_unchanged, denominator=len(change.differences)
            )
            lines.append(
                f"  unchanged   {unchanged}, exact paired sign-flip p = {change.p_value:.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def render_curves(*, budgets: dict[str, list[SeedRun]], output: Path) -> None:
        """The learning curve itself, grouped by stranding mode: one row per budget, and
        the two x axes the house style pairs -- practice cycle and online transitions.

        **The two columns are affine on this data** (`transition_step` returns a constant
        across every run), so the right column is the left one rescaled rather than a
        second measurement. It is drawn on a symlog axis for exactly that reason: the
        divergence between the modes is complete within three cycles of a hundred, which
        a linear axis compresses into the leftmost few percent of the panel and hides.

        Per-seed traces are drawn faint underneath a bold group mean. With a 6/4 split a
        mean-only plot would hide the finding, which is that there are two groups at all.
        """
        labels = list(budgets)
        figure, axes = plt.subplots(
            len(labels), 2, figsize=(13.0, 4.6 * len(labels)), squeeze=False
        )
        for row, label in enumerate(labels):
            runs = budgets[label]
            step = ResetFreeWallclockModes.transition_step(runs=runs)
            for column, by_transitions in ((0, False), (1, True)):
                ResetFreeWallclockModes._curve_panel(
                    axes=axes[row][column],
                    runs=runs,
                    label=label,
                    by_transitions=by_transitions,
                )
            axes[row][1].annotate(
                f"{step} transitions per cycle on "
                + ResetFreeWallclockModes.format_count(numerator=len(runs), denominator=len(runs))
                + " runs, so this column is\nthe left one rescaled"
                if step is not None
                else "transitions per cycle are not constant here",
                (0.98, 0.06),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                color="#555555",
            )
        handles = [
            Line2D([], [], color=_MODE_COLORS[mode], linewidth=2.4, label=f"{text} (group mean)")
            for mode, text in _MODE_LABELS.items()
        ] + [Line2D([], [], color="#777777", linewidth=1.0, alpha=0.5, label="individual seeds")]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        figure.suptitle(
            "The learning curve splits by stranding cycle, and neither group moves with the budget",
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)

    @staticmethod
    def _curve_panel(*, axes, runs: list[SeedRun], label: str, by_transitions: bool) -> None:
        """One panel: faint per-seed traces under a bold per-mode mean."""
        grouped = ResetFreeWallclockModes.curves_by_mode(runs=runs)
        total = runs[0].num_test_tasks
        for mode, members in grouped.items():
            if not members:
                continue
            colour = _MODE_COLORS[mode]
            for run in members:
                axes.plot(
                    run.transitions_per_checkpoint
                    if by_transitions
                    else list(range(len(run.solved_per_checkpoint))),
                    run.solved_per_checkpoint,
                    color=colour,
                    alpha=0.32,
                    linewidth=1.0,
                )
            length = min(len(run.solved_per_checkpoint) for run in members)
            mean = [
                statistics.fmean([run.solved_per_checkpoint[i] for run in members])
                for i in range(length)
            ]
            axes.plot(
                members[0].transitions_per_checkpoint[:length]
                if by_transitions
                else list(range(length)),
                mean,
                color=colour,
                linewidth=2.6,
                label=(
                    _MODE_LABELS[mode]
                    + ": "
                    + ResetFreeWallclockModes.format_count(
                        numerator=len(members), denominator=len(runs)
                    )
                    + " seeds, ending "
                    + ResetFreeWallclockModes.format_count(
                        numerator=sum(run.final_solved for run in members),
                        denominator=total * len(members),
                    )
                ),
            )
        if by_transitions:
            axes.set_xscale("symlog", linthresh=runs[0].transitions_per_checkpoint[1] or 1)
            axes.set_xlabel("online transitions (symlog -- the split happens at the far left)")
        else:
            axes.set_xlabel("practice cycle")
        axes.set_ylim(0, total)
        axes.set_ylabel(f"tasks solved (x/{total})")
        axes.set_title(
            f"{label} budget, {'by online transitions' if by_transitions else 'by practice cycle'}"
        )
        axes.grid(alpha=0.25)
        axes.legend(fontsize=8, loc="upper left", framealpha=0.85)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--budget",
            action="append",
            required=True,
            metavar="LABEL=DIR",
            help="a budget's label and the directory holding its <seed>/stats.json",
        )
        parser.add_argument("--output", type=Path, required=True, help="modes figure path (.png)")
        parser.add_argument(
            "--curves-output", type=Path, help="learning-curve-by-mode figure path (.png)"
        )
        args = parser.parse_args()
        directories: dict[str, Path] = {}
        for entry in args.budget:
            label, _, path = entry.partition("=")
            if not path:
                raise ValueError(f"--budget wants LABEL=DIR, got {entry!r}")
            directories[label] = Path(path)
        budgets = ResetFreeWallclockModes.load_budgets(directories=directories)
        print(ResetFreeWallclockModes.report(budgets=budgets))
        ResetFreeWallclockModes.render(budgets=budgets, output=args.output)
        print(f"wrote {args.output}")
        if args.curves_output is not None:
            ResetFreeWallclockModes.render_curves(budgets=budgets, output=args.curves_output)
            print(f"wrote {args.curves_output}")


if __name__ == "__main__":
    ResetFreeWallclockModes.main()
