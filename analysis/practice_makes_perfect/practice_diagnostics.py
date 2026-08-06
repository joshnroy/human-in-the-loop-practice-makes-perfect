"""Post-run analysis for the two per-window diagnostic records in `stats.json`:
`practice_outcomes_per_cycle` (what practicing each lifted skill actually did) and
`planning_failures_per_cycle`/`planning_attempts_per_cycle` (how often the planner was
asked and how often it came back empty).

**The question this is for.** A learning curve says a method scored 21/100. It cannot
say *why*, and the two candidate answers need different fixes: the samplers were never
given enough labels (starvation -- buy more transitions), or they have the labels and
cannot fit them (inability -- change the representation). Those look identical in
`evaluations` and different here. Read the per-skill panels as: no line at all means the
skill was never practiced; a line whose *informed* attempts stay at zero means the
sampler was asked but never had a classifier that could rank its candidates; a healthy
informed count with a flat informed success count means it was asked and missed.

Domain- and method-agnostic: it keys on nothing but the lifted skill names a run
happens to record, so the same script serves Light Switch, both Tossing Rooms, Ball-Ring
and Tossing3D. That is the whole point -- the previous route to these numbers was
`scripts/tossingroomsplit_skill_traces.py`, which imports one domain's `Environment`
directly and could not be pointed at another.

Never runs a simulation or drives a `Problem`/`Method` itself (CLAUDE.md's `analysis/`
convention): it reads back what `hitl_pmp.method_runner.MethodRunner.run` already wrote
under `--output-dir`, laid out as `<results-root>/<method>/<seed>/stats.json` -- the
layout `scripts/run_sweep.py` produces.

**Per-seed spread, not just a mean.** Every panel draws one faint line per seed under
the mean, because with ten seeds a mean can be one seed's behaviour: a single run that
practiced a skill 300 times while nine practiced it twice produces a mean of 32, which
describes nothing that happened.

**The x-axis offset is real and is handled here.** Both records are bucketed by
`PracticeLoop`'s `on_cycle_end`, which fires *before* each evaluation sweep, so bucket
`i` covers sweep `i` and the practice period after it -- it holds the practice that runs
between `evaluations[i]` and `evaluations[i+1]`. Plotting bucket `i` at
`evaluations[i]`'s transition count would shift a whole practice period one checkpoint
left. `window_transitions` puts each bucket at the transition count it *ends* at
instead, which is where the practice it describes actually happened. The trailing
bucket covers the final sweep alone and contains no practice at all -- see
`practice_window_count` for why the practice panels stop before it and the planning
panel does not.

Counts, never bare percentages: every printed cell and every axis label is `x/y`.
"""

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.method.types import SkillPracticeTally  # noqa: E402
from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402


class PracticeDiagnostics:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_runs(*, method_dir: Path) -> list[Metrics]:
        """Every seed's `Metrics` under one method directory, in seed-directory order.

        Reconstructed as `Metrics` rather than parsed as raw JSON so the per-run
        arithmetic (`total_practice_outcomes`, `total_planning_outcomes`) has exactly
        one implementation, in the class that owns the fields."""
        return [
            Metrics.model_validate_json(stats_path.read_text())
            for stats_path in sorted(method_dir.glob("*/stats.json"))
        ]

    @staticmethod
    def summarize(*, results_root: Path) -> dict[str, list[Metrics]]:
        return {
            method_dir.name: PracticeDiagnostics.load_runs(method_dir=method_dir)
            for method_dir in sorted(results_root.iterdir())
            if method_dir.is_dir()
        }

    @staticmethod
    def window_transitions(*, metrics: Metrics) -> list[int]:
        """The transition count each per-window bucket *ends* at.

        Bucket `i` covers evaluation sweep `i` and the practice period after it, so the
        practice it describes finished at `evaluations[i + 1]`'s transition count. The
        trailing bucket covers the final sweep alone and contains no practice, so it
        sits at the final transition count. Getting this wrong is a one-checkpoint
        shift, which on a short run is the whole effect."""
        transitions = [entry[0] for entry in metrics.evaluations]
        if not transitions:
            return []
        return [*transitions[1:], transitions[-1]]

    @staticmethod
    def skill_names(*, runs: Sequence[Metrics]) -> list[str]:
        """Every lifted skill any seed practiced, sorted. Taken over the union rather
        than over one seed, because a skill that only some seeds ever reached is
        precisely the kind of thing worth seeing."""
        names: set[str] = set()
        for metrics in runs:
            for window in metrics.practice_outcomes_per_cycle:
                names.update(window)
        return sorted(names)

    @staticmethod
    def per_seed_series(*, runs: Sequence[Metrics], skill_name: str, field: str) -> list[list[int]]:
        """One per-window series per seed, for one lifted skill and one counter.

        A seed that never practiced the skill still contributes a series -- of zeros,
        the same length as its own bucket list -- because "this seed never touched it"
        is data, and dropping it would quietly shrink the denominator of the mean."""
        series: list[list[int]] = []
        for metrics in runs:
            series.append([
                getattr(window.get(skill_name, SkillPracticeTally()), field)
                for window in metrics.practice_outcomes_per_cycle
            ])
        return series

    @staticmethod
    def mean_series(*, series: Sequence[Sequence[int]]) -> list[float]:
        """Truncated to the shortest seed, so a partially-finished run cannot lengthen
        or skew the others."""
        usable = [entry for entry in series if entry]
        if not usable:
            return []
        length = min(len(entry) for entry in usable)
        return [
            statistics.mean([float(entry[index]) for entry in usable]) for index in range(length)
        ]

    @staticmethod
    def totals(*, runs: Sequence[Metrics]) -> dict[str, SkillPracticeTally]:
        """Every seed's practice pooled per lifted skill -- the x/y counts the table
        prints. Pooled rather than averaged: a mean of rates over seeds with different
        attempt counts weights a two-attempt seed like a two-hundred-attempt one."""
        pooled: dict[str, SkillPracticeTally] = {}
        for metrics in runs:
            for name, tally in metrics.total_practice_outcomes().items():
                pooled[name] = pooled.get(name, SkillPracticeTally()).plus(other=tally)
        return {name: pooled[name] for name in sorted(pooled)}

    @staticmethod
    def print_table(*, summary: dict[str, list[Metrics]]) -> None:
        """Every cell an x/y, pooled across seeds, with the seed count stated so the
        denominators are recoverable."""
        for method, runs in summary.items():
            print(f"\n=== {method} ({len(runs)} seeds) ===")
            failures = sum(metrics.total_planning_outcomes()[0] for metrics in runs)
            attempts = sum(metrics.total_planning_outcomes()[1] for metrics in runs)
            print(f"planning: {failures}/{attempts} planner calls found no plan")
            pooled = PracticeDiagnostics.totals(runs=runs)
            if not pooled:
                print("practice: nothing recorded (this Method does not measure practice)")
                continue
            header = (
                f"{'skill':<22}{'succeeded':>14}{'informed':>14}"
                f"{'epsilon-random':>18}{'fallback':>14}"
            )
            print(header)
            for name, tally in pooled.items():
                print(
                    f"{name:<22}"
                    f"{f'{tally.num_successes}/{tally.num_attempts}':>14}"
                    f"{f'{tally.num_informed_successes}/{tally.num_informed_attempts}':>14}"
                    f"{f'{tally.num_random_successes}/{tally.num_random_attempts}':>18}"
                    f"{f'{tally.num_fallback_successes()}/{tally.num_fallback_attempts()}':>14}"
                )

    @staticmethod
    def _draw_panel(
        *,
        axis: plt.Axes,
        runs: Sequence[Metrics],
        skill_name: str,
        field: str,
        color: str,
        label: str,
        width: float,
    ) -> None:
        """One counter for one skill: a faint line per seed, the mean on top.

        The per-seed lines are the point of the panel rather than decoration -- see the
        module docstring on why a mean alone can describe nothing that happened.

        `width` descends across the four series a caller draws, so a skill whose
        successes equal its attempts (every deterministic skill) shows both rather than
        hiding one exactly under the other."""
        series = PracticeDiagnostics.per_seed_series(runs=runs, skill_name=skill_name, field=field)
        for metrics, entry in zip(runs, series, strict=True):
            transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
            length = min(
                len(transitions),
                len(entry),
                PracticeDiagnostics.practice_window_count(metrics=metrics),
            )
            if length:
                axis.plot(transitions[:length], entry[:length], color=color, alpha=0.18, lw=1)
        mean = PracticeDiagnostics.mean_series(series=series)
        if mean and runs:
            transitions = PracticeDiagnostics.window_transitions(metrics=runs[0])
            length = min(
                len(transitions),
                len(mean),
                PracticeDiagnostics.practice_window_count(metrics=runs[0]),
            )
            axis.plot(
                transitions[:length],
                mean[:length],
                color=color,
                lw=width,
                marker="o",
                ms=3,
                label=label,
            )

    @staticmethod
    def practice_window_count(*, metrics: Metrics) -> int:
        """How many buckets describe a practice period.

        Every bucket but the last does. The trailing one covers the final evaluation
        sweep alone -- no practice runs in it *by construction*, and it sits at the same
        transition count as the bucket before it, since an evaluation sweep charges no
        transitions. Plotting it in a practice panel therefore draws a structural zero
        at a duplicated x, which renders as a cliff at the right-hand edge and reads as
        a collapse. It is excluded from the practice panels for that reason, and kept in
        the planning panel, where the final sweep really does make planner calls."""
        return max(len(metrics.practice_outcomes_per_cycle) - 1, 0)

    @staticmethod
    def plot(*, summary: dict[str, list[Metrics]], output_path: Path) -> None:
        """One row per (method, lifted skill) for practice, plus one row per method for
        planning. Every panel shares the transitions x-axis the learning curve uses, so
        a reader can lay this beside `analysis/practice_makes_perfect/ees.py`'s figure
        and read the two at the same checkpoints."""
        rows: list[tuple[str, list[Metrics], str | None]] = []
        for method, runs in summary.items():
            for name in PracticeDiagnostics.skill_names(runs=runs):
                rows.append((method, runs, name))
            rows.append((method, runs, None))
        if not rows:
            raise ValueError("nothing to plot: no stats.json under --results-root")
        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 3.1 * len(rows)), squeeze=False)
        try:
            for axis, (method, runs, name) in zip(axes[:, 0], rows, strict=True):
                if name is None:
                    PracticeDiagnostics._draw_planning_panel(axis=axis, runs=runs)
                    axis.set_title(f"{method}: planning calls per window ({len(runs)} seeds)")
                    axis.set_ylabel("planner calls\n(failed / attempted)")
                else:
                    for field, color, label, width in (
                        ("num_attempts", "tab:blue", "attempts", 3.2),
                        ("num_successes", "tab:green", "successes", 2.4),
                        ("num_informed_attempts", "tab:orange", "informed attempts", 1.8),
                        ("num_informed_successes", "tab:red", "informed successes", 1.2),
                    ):
                        PracticeDiagnostics._draw_panel(
                            axis=axis,
                            runs=runs,
                            skill_name=name,
                            field=field,
                            color=color,
                            label=label,
                            width=width,
                        )
                    axis.set_title(f"{method}: {name} practice per window ({len(runs)} seeds)")
                    axis.set_ylabel("executions per window\n(count, not a rate)")
                axis.set_xlabel("Number of online transitions at the end of the window")
                axis.legend(fontsize=8)
                axis.margins(y=0.15)
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

    @staticmethod
    def _draw_planning_panel(*, axis: plt.Axes, runs: Sequence[Metrics]) -> None:
        """#106's counters, which nothing plotted until now. Failures and attempts on
        one axis rather than a ratio: EES plans speculatively, so a healthy run reports
        a nonzero, workload-dependent failure count and only `failures` tracking
        `attempts` means anything is wrong -- a shape a ratio hides."""
        for field, color, label in (
            ("planning_attempts_per_cycle", "tab:blue", "attempts"),
            ("planning_failures_per_cycle", "tab:red", "failures"),
        ):
            for metrics in runs:
                transitions = PracticeDiagnostics.window_transitions(metrics=metrics)
                values = getattr(metrics, field)
                length = min(len(transitions), len(values))
                if length:
                    axis.plot(transitions[:length], values[:length], color=color, alpha=0.18, lw=1)
            series = [getattr(metrics, field) for metrics in runs]
            mean = PracticeDiagnostics.mean_series(series=series)
            if mean and runs:
                transitions = PracticeDiagnostics.window_transitions(metrics=runs[0])
                length = min(len(transitions), len(mean))
                axis.plot(
                    transitions[:length],
                    mean[:length],
                    color=color,
                    lw=3.2 if field.endswith("attempts_per_cycle") else 2.0,
                    marker="o",
                    ms=3,
                    label=label,
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Sweep directory laid out as <results-root>/<method>/<seed>/stats.json.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional diagnostics PNG.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = PracticeDiagnostics.summarize(results_root=args.results_root)
    PracticeDiagnostics.print_table(summary=summary)
    if args.output is not None:
        PracticeDiagnostics.plot(summary=summary, output_path=args.output)
        print(f"\nWrote plot to {args.output}")


if __name__ == "__main__":
    main()
