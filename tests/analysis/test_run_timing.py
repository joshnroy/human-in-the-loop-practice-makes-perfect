"""`analysis/run_timing.py` exists to answer one question -- per-run wall-clock as a
function of the concurrency a run actually experienced -- and the way that answer goes
wrong is silently: pick the sweep-local concurrency count on a shared box and every
number is attributed to the wrong cause, with no crash and no implausible output. So
the tests pin *which* signal is used, and what happens when it is missing.
"""

from pathlib import Path

from analysis.run_timing import RunTimingAnalysis
from scripts.run_sweep import MachineSample, RunTiming


def _write_timing(
    *,
    results_root: Path,
    method: str,
    seed: int,
    elapsed_seconds: float,
    machine: int | None,
    sweep_in_flight: int,
    sweep_id: str = "sweep-a",
    start_epoch_seconds: float = 1000.0,
) -> RunTiming:
    timing = RunTiming(
        sweep_id=sweep_id,
        method=method,
        seed=seed,
        start_time="2026-08-04T12:00:00-04:00",
        end_time="2026-08-04T12:01:00-04:00",
        start_epoch_seconds=start_epoch_seconds,
        end_epoch_seconds=start_epoch_seconds + elapsed_seconds,
        elapsed_seconds=elapsed_seconds,
        returncode=0,
        succeeded=True,
        max_workers=4,
        cpu_count=24,
        sweep_runs_in_flight_at_start=sweep_in_flight,
        sweep_runs_in_flight_at_end=sweep_in_flight,
        machine_at_start=MachineSample(cli_processes=machine, load_average_1min=1.0),
        machine_at_end=MachineSample(cli_processes=machine, load_average_1min=1.0),
    )
    run_dir = results_root / method / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "timing.json").write_text(timing.model_dump_json())
    return timing


def test_load_finds_every_run_by_filename_not_by_mtime(*, tmp_path: Path) -> None:
    """Directory mtimes were the only signal before this record existed, and they
    answer a different question (when a file was last written, not when its run
    began). The glob is over the layout run_sweep writes, nothing else."""
    _write_timing(
        results_root=tmp_path,
        method="ees",
        seed=0,
        elapsed_seconds=10.0,
        machine=8,
        sweep_in_flight=4,
    )
    _write_timing(
        results_root=tmp_path,
        method="random-skills",
        seed=1,
        elapsed_seconds=20.0,
        machine=8,
        sweep_in_flight=4,
    )
    timings = RunTimingAnalysis.load(results_root=tmp_path)
    assert {(timing.method, timing.seed) for timing in timings} == {
        ("ees", 0),
        ("random-skills", 1),
    }


def test_load_returns_runs_oldest_first(*, tmp_path: Path) -> None:
    """Sorted by start time, not by directory name, so a printed table reads as the
    sweep actually unfolded rather than alphabetically."""
    _write_timing(
        results_root=tmp_path,
        method="zzz",
        seed=0,
        elapsed_seconds=1.0,
        machine=1,
        sweep_in_flight=1,
        start_epoch_seconds=100.0,
    )
    _write_timing(
        results_root=tmp_path,
        method="aaa",
        seed=0,
        elapsed_seconds=1.0,
        machine=1,
        sweep_in_flight=1,
        start_epoch_seconds=200.0,
    )
    assert [timing.method for timing in RunTimingAnalysis.load(results_root=tmp_path)] == [
        "zzz",
        "aaa",
    ]


def test_observed_concurrency_prefers_the_machine_wide_count(*, tmp_path: Path) -> None:
    """Several agents' sweeps share this box: a run launched by a 4-worker sweep
    while another sweep runs 24 competed with ~28 runs, not 4. Using the sweep-local
    count here is exactly the mistake that would make the answer wrong."""
    timing = _write_timing(
        results_root=tmp_path,
        method="ees",
        seed=0,
        elapsed_seconds=10.0,
        machine=27,
        sweep_in_flight=4,
    )
    concurrency, machine_wide = RunTimingAnalysis.observed_concurrency(timing=timing)
    assert concurrency == 28
    assert machine_wide is True


def test_observed_concurrency_counts_the_run_itself(*, tmp_path: Path) -> None:
    """The machine-wide samples are taken before this run's child is spawned and
    after it exits, so they omit it, while the sweep-local counter includes it.
    Untreated, the two branches would return numbers on different scales -- a run
    alone on the box would read 0 machine-wide and 1 sweep-locally for the same
    situation."""
    timing = _write_timing(
        results_root=tmp_path,
        method="ees",
        seed=0,
        elapsed_seconds=10.0,
        machine=0,
        sweep_in_flight=1,
    )
    assert RunTimingAnalysis.observed_concurrency(timing=timing) == (1, True)


def test_observed_concurrency_falls_back_to_a_labelled_lower_bound(*, tmp_path: Path) -> None:
    """Without /proc the machine-wide count is unknown, and the sweep-local count is
    only a lower bound -- so it is used, but reported as such rather than passed off
    as the real thing."""
    timing = _write_timing(
        results_root=tmp_path,
        method="ees",
        seed=0,
        elapsed_seconds=10.0,
        machine=None,
        sweep_in_flight=4,
    )
    concurrency, machine_wide = RunTimingAnalysis.observed_concurrency(timing=timing)
    assert concurrency == 4
    assert machine_wide is False


def test_grouping_buckets_elapsed_by_observed_concurrency(*, tmp_path: Path) -> None:
    """The output that motivated the whole record: wall-clock per run at each level
    of concurrency, which is what a knee would show up in. Bucket labels count the
    run itself, so a recorded machine count of 3 lands in bucket 4."""
    for seed, (machine, elapsed) in enumerate([(3, 10.0), (3, 12.0), (23, 30.0)]):
        _write_timing(
            results_root=tmp_path,
            method="ees",
            seed=seed,
            elapsed_seconds=elapsed,
            machine=machine,
            sweep_in_flight=4,
        )
    buckets = RunTimingAnalysis.group_by_concurrency(
        timings=RunTimingAnalysis.load(results_root=tmp_path)
    )
    assert buckets == {4: [10.0, 12.0], 24: [30.0]}


def test_a_results_root_with_no_timings_reports_that_rather_than_crashing(
    *, tmp_path: Path
) -> None:
    """Every sweep run before this record existed has none, and that has to read as
    "not recorded" rather than as an empty result or a traceback."""
    assert RunTimingAnalysis.load(results_root=tmp_path) == []
