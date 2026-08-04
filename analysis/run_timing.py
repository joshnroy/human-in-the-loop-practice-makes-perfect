"""Post-run analysis of run wall-clock vs. observed concurrency: reads the
`timing.json` files `scripts/run_sweep.py` writes next to each run's `stats.json`
and answers "how long does a run take, and does adding concurrency help?" from
recorded data rather than a fresh hour of hand measurement.

Never runs a simulation -- see CLAUDE.md's analysis/ convention. It only reads
`--results-root DIR` laid out as `DIR/<method>/<seed>/timing.json`, found by
**filename glob**, never by file mtime: mtimes were the only available signal
before this record existed, and they answer a different question (when a file was
last written, not when its run began).

It imports `RunTiming` from `scripts.run_sweep` rather than redeclaring it. The
boundary `scripts/` protects is that nothing there imports `hitl_pmp`; a reader
importing the writer's schema is the opposite direction, and one definition of
the record beats a second copy that can drift out of step with the writer.

Concurrency here means the **machine-wide** `hitl_pmp.cli` process count, not this
sweep's own in-flight count, because several agents' sweeps share this box: a run
launched by a 4-worker sweep while another sweep runs 24 competed with 28 runs,
not 4. Where the machine-wide sample is unavailable (no /proc), the sweep-local
count is used as an explicit, labelled lower bound instead.
"""

import argparse
import statistics
from pathlib import Path

from scripts.run_sweep import RunTiming


class RunTimingAnalysis:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load(*, results_root: Path) -> list[RunTiming]:
        """Every recorded run under a results root, oldest first. Sorted by start
        time so a printed table reads as the sweep actually unfolded."""
        timings = [
            RunTiming.model_validate_json(path.read_text())
            for path in sorted(results_root.glob("*/*/timing.json"))
        ]
        return sorted(timings, key=lambda timing: timing.start_epoch_seconds)

    @staticmethod
    def observed_concurrency(*, timing: RunTiming) -> tuple[float, bool]:
        """How many runs were in flight alongside this one, counting itself, and
        whether that number is machine-wide (True) or only this sweep's own
        children (False, a lower bound because other agents share the box).

        Averages the entry and exit samples: both are point samples, and their
        mean is the cheapest unbiased summary of a run's whole life.

        The `+ 1` is not a fudge. The machine-wide samples are taken before this
        run's own child is spawned and after it has exited (subprocess.run blocks
        in between), so they exclude it, while the sweep-local counter includes
        it. Without the correction the two branches would return numbers on
        different scales and a mixed-platform results root would silently compare
        apples to oranges.
        """
        start = timing.machine_at_start.cli_processes
        end = timing.machine_at_end.cli_processes
        if start is not None and end is not None:
            return (start + end) / 2 + 1, True
        return (
            (timing.sweep_runs_in_flight_at_start + timing.sweep_runs_in_flight_at_end) / 2,
            False,
        )

    @staticmethod
    def group_by_concurrency(*, timings: list[RunTiming]) -> dict[int, list[float]]:
        """Elapsed seconds bucketed by observed concurrency, rounded to an integer
        number of concurrent runs. Buckets rather than a fit: with a handful of
        runs per level a median is honest and a regression line is not."""
        buckets: dict[int, list[float]] = {}
        for timing in timings:
            concurrency, _ = RunTimingAnalysis.observed_concurrency(timing=timing)
            buckets.setdefault(round(concurrency), []).append(timing.elapsed_seconds)
        return buckets

    @staticmethod
    def print_runs(*, timings: list[RunTiming]) -> None:
        """`observed` is the number the aggregates below bucket on (machine-wide
        where available, this run included); `sweep` is this sweep's own in-flight
        count at the run's exit, printed alongside so the gap between "my workers"
        and "everyone's" is visible per run rather than only in aggregate."""
        print(
            f"{'method':<16}{'seed':>5}{'elapsed s':>11}{'observed':>10}{'sweep':>7}"
            f"{'load':>8}  {'started':<32}{'ok':>4}"
        )
        for timing in timings:
            observed, _ = RunTimingAnalysis.observed_concurrency(timing=timing)
            load = timing.machine_at_end.load_average_1min
            print(
                f"{timing.method:<16}{timing.seed:>5}{timing.elapsed_seconds:>11.1f}"
                f"{observed:>10.1f}{timing.sweep_runs_in_flight_at_end:>7}"
                f"{'?' if load is None else f'{load:.2f}':>8}  "
                f"{timing.start_time:<32}{'y' if timing.succeeded else 'n':>4}"
            )

    @staticmethod
    def print_concurrency_table(*, timings: list[RunTiming]) -> None:
        """The question this record exists to answer: per-run wall-clock as a
        function of how many runs were in flight alongside it."""
        buckets = RunTimingAnalysis.group_by_concurrency(timings=timings)
        machine_wide = all(
            RunTimingAnalysis.observed_concurrency(timing=timing)[1] for timing in timings
        )
        scope = "machine-wide" if machine_wide else "this sweep only (lower bound)"
        print(f"\nElapsed vs. observed concurrency ({scope}):")
        print(f"{'concurrent runs':>16}{'runs':>7}{'median s':>11}{'mean s':>10}{'max s':>10}")
        for concurrency, elapsed in sorted(buckets.items()):
            print(
                f"{concurrency:>16}{len(elapsed):>7}{statistics.median(elapsed):>11.1f}"
                f"{statistics.mean(elapsed):>10.1f}{max(elapsed):>10.1f}"
            )

    @staticmethod
    def print_sweep_summary(*, timings: list[RunTiming]) -> None:
        """Per sweep, because two sweeps sharing a results root ran under
        different `--max-workers` and must not be pooled."""
        by_sweep: dict[str, list[RunTiming]] = {}
        for timing in timings:
            by_sweep.setdefault(timing.sweep_id, []).append(timing)
        print(f"\n{'sweep':<28}{'runs':>6}{'workers':>9}{'cpus':>6}{'wall s':>10}{'run s':>10}")
        for sweep_id, sweep_timings in sorted(by_sweep.items()):
            # Sweep wall-clock is last end minus first start, i.e. the elapsed
            # time a caller waited -- not the sum of the runs, which counts
            # concurrent work several times over.
            wall = max(timing.end_epoch_seconds for timing in sweep_timings) - min(
                timing.start_epoch_seconds for timing in sweep_timings
            )
            cpus = sweep_timings[0].cpu_count
            print(
                f"{sweep_id:<28}{len(sweep_timings):>6}{sweep_timings[0].max_workers:>9}"
                f"{'?' if cpus is None else cpus:>6}{wall:>10.1f}"
                f"{statistics.median([t.elapsed_seconds for t in sweep_timings]):>10.1f}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also print one row per run, not just the aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timings = RunTimingAnalysis.load(results_root=args.results_root)
    if not timings:
        print(f"No timing.json found under {args.results_root} (sweeps predating them?)")
        return
    if args.per_run:
        RunTimingAnalysis.print_runs(timings=timings)
    RunTimingAnalysis.print_concurrency_table(timings=timings)
    RunTimingAnalysis.print_sweep_summary(timings=timings)


if __name__ == "__main__":
    main()
