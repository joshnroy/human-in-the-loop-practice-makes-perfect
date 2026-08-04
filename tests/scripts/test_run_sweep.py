import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from scripts.run_sweep import (
    InFlightCounter,
    MachineSampler,
    RunTiming,
    SpawnRetryPolicy,
    SweepRun,
    SweepRunner,
)

# Tests never wait out a real backoff -- the schedule is exercised by
# test_backoff_grows_exponentially, not by sleeping through it.
_NO_BACKOFF = SpawnRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)


def test_plan_produces_one_run_per_method_seed_pair() -> None:
    runs = SweepRunner.plan(
        env="lightswitch",
        methods=["ees", "random-skills"],
        seeds=[0, 1, 2],
        results_root=Path("/tmp/results"),
        shared_args=[],
        method_args={},
    )
    assert len(runs) == 6
    assert {(run.method, run.seed) for run in runs} == {
        (method, seed) for method in ("ees", "random-skills") for seed in (0, 1, 2)
    }


def test_plan_writes_into_the_layout_the_analysis_scripts_expect() -> None:
    """analysis/practice_makes_perfect/ees.py globs <method>/<seed>/stats.json --
    the sweep has to produce exactly that tree or the analysis silently finds
    nothing."""
    (run,) = SweepRunner.plan(
        env="lightswitch",
        methods=["ees"],
        seeds=[3],
        results_root=Path("/tmp/results"),
        shared_args=[],
        method_args={},
    )
    assert run.output_dir == Path("/tmp/results/ees/3")


def test_plan_threads_seed_env_and_method_into_the_command() -> None:
    (run,) = SweepRunner.plan(
        env="lightswitch",
        methods=["ees"],
        seeds=[7],
        results_root=Path("/tmp/results"),
        shared_args=[],
        method_args={},
    )
    assert run.command[:3] == [sys.executable, "-m", "hitl_pmp.cli"]
    assert "--env" in run.command and "lightswitch" in run.command
    assert "--method" in run.command and "ees" in run.command
    assert run.command[run.command.index("--seed") + 1] == "7"


def test_plan_applies_shared_args_to_every_run() -> None:
    runs = SweepRunner.plan(
        env="lightswitch",
        methods=["ees", "skill-oracle"],
        seeds=[0],
        results_root=Path("/tmp/results"),
        shared_args=["--grid-size", "25"],
        method_args={},
    )
    for run in runs:
        assert run.command[run.command.index("--grid-size") + 1] == "25"


def test_plan_applies_method_args_only_to_that_method() -> None:
    """Methods do not share a flag set -- `--method skill-oracle` rejects
    --num-cycles outright, so per-method args are not optional sugar."""
    runs = SweepRunner.plan(
        env="lightswitch",
        methods=["ees", "skill-oracle"],
        seeds=[0],
        results_root=Path("/tmp/results"),
        shared_args=[],
        method_args={"ees": ["--num-cycles", "10"]},
    )
    by_method = {run.method: run.command for run in runs}
    assert "--num-cycles" in by_method["ees"]
    assert "--num-cycles" not in by_method["skill-oracle"]


def test_default_seeds_are_a_fixed_contiguous_range() -> None:
    """Seeds are fixed and deterministic (0..n-1), never drawn randomly -- a sweep
    has to be re-runnable to the same numbers months later, and the paper's own
    protocol is 'we run 10 random seeds of each approach', i.e. a fixed set."""
    assert SweepRunner.default_seeds(num_seeds=10) == list(range(10))


def test_parse_method_args_accepts_method_equals_flags() -> None:
    parsed = SweepRunner.parse_method_args(
        raw=["ees=--num-cycles 10 --max-steps-per-interaction 150"]
    )
    assert parsed == {"ees": ["--num-cycles", "10", "--max-steps-per-interaction", "150"]}


def test_parse_method_args_rejects_a_malformed_entry() -> None:
    with pytest.raises(ValueError, match="method=args"):
        SweepRunner.parse_method_args(raw=["no-equals-sign"])


def test_execute_runs_every_command_and_reports_success(*, tmp_path: Path) -> None:
    runs = [
        SweepRun(
            method="fake",
            seed=seed,
            output_dir=tmp_path / "fake" / str(seed),
            command=[sys.executable, "-c", "print('ok')"],
        )
        for seed in range(3)
    ]
    outcomes = SweepRunner.execute(runs=runs, max_workers=3)
    assert len(outcomes) == 3
    assert all(outcome.succeeded for outcome in outcomes)
    # Output directories are created up front so a run can write straight into them.
    for run in runs:
        assert run.output_dir.is_dir()


def test_execute_reports_a_failing_run_without_taking_the_sweep_down(*, tmp_path: Path) -> None:
    """One bad seed must not abort the other 29 -- an interrupted sweep is far
    more expensive to recover from than a single missing datapoint."""
    runs = [
        SweepRun(
            method="fake",
            seed=0,
            output_dir=tmp_path / "fake" / "0",
            command=[sys.executable, "-c", "raise SystemExit(3)"],
        ),
        SweepRun(
            method="fake",
            seed=1,
            output_dir=tmp_path / "fake" / "1",
            command=[sys.executable, "-c", "print('fine')"],
        ),
    ]
    outcomes = sorted(SweepRunner.execute(runs=runs, max_workers=2), key=lambda o: o.run.seed)
    assert outcomes[0].succeeded is False
    assert outcomes[0].returncode == 3
    assert outcomes[1].succeeded is True


def test_execute_pins_single_threaded_math_in_children(*, tmp_path: Path) -> None:
    """Workers run concurrently, so each child must not also try to grab every
    core -- that oversubscribes and slows the sweep down rather than speeding it
    up."""
    output = tmp_path / "env.txt"
    run = SweepRun(
        method="fake",
        seed=0,
        output_dir=tmp_path / "fake" / "0",
        command=[
            sys.executable,
            "-c",
            f"import os, pathlib; pathlib.Path({str(output)!r}).write_text("
            "os.environ.get('OMP_NUM_THREADS', 'unset'))",
        ],
    )
    SweepRunner.execute(runs=[run], max_workers=1)
    assert output.read_text() == "1"


def _fake_run(*, tmp_path: Path, seed: int, command: list[str]) -> SweepRun:
    return SweepRun(
        method="fake", seed=seed, output_dir=tmp_path / "fake" / str(seed), command=command
    )


def _timing_of(*, run: SweepRun) -> RunTiming:
    return RunTiming.model_validate_json((run.output_dir / "timing.json").read_text())


def test_execute_records_timing_beside_the_run_it_timed(*, tmp_path: Path) -> None:
    """The record exists so "how long does a run take?" is answerable from data
    instead of re-measured by hand -- which it has been, repeatedly, because the
    only previous signal was output-directory mtimes."""
    run = _fake_run(
        tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "import time; time.sleep(0.05)"]
    )
    SweepRunner.execute(runs=[run], max_workers=1)
    timing = _timing_of(run=run)
    assert timing.method == "fake"
    assert timing.seed == 0
    assert timing.succeeded is True
    assert timing.returncode == 0
    assert timing.elapsed_seconds >= 0.05


def test_timing_is_written_for_a_failing_run_too(*, tmp_path: Path) -> None:
    """A failed run's wall-clock is data. Dropping it would bias any later
    concurrency analysis toward whatever happened to succeed."""
    run = _fake_run(
        tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "raise SystemExit(3)"]
    )
    SweepRunner.execute(runs=[run], max_workers=1)
    timing = _timing_of(run=run)
    assert timing.returncode == 3
    assert timing.succeeded is False


def test_timing_is_written_even_when_the_child_cannot_be_launched(*, tmp_path: Path) -> None:
    """subprocess.run itself raising (no such executable) is the one path that
    would otherwise skip the record entirely -- hence the try/finally."""
    run = _fake_run(tmp_path=tmp_path, seed=0, command=["definitely-not-a-real-executable"])
    outcomes = SweepRunner.execute(runs=[run], max_workers=1, retry_policy=_NO_BACKOFF)
    assert outcomes[0].returncode == -1
    timing = _timing_of(run=run)
    assert timing.returncode == -1
    assert timing.succeeded is False


def test_a_run_that_cannot_be_launched_is_reported_not_raised(*, tmp_path: Path) -> None:
    """A spawn that raises must not abort the sweep. `executor.map` re-raises the
    first exception any worker raised, so letting it out would take down every
    other run in the grid -- the exact opposite of SweepOutcome's contract, and
    reachable in practice: under memory pressure fork() raises OSError.

    The original exception has to survive as data, not be swallowed: the whole
    point of catching it is that the sweep continues, so the traceback is the
    only remaining evidence of what went wrong."""
    broken = _fake_run(tmp_path=tmp_path, seed=0, command=["definitely-not-a-real-executable"])
    healthy = _fake_run(tmp_path=tmp_path, seed=1, command=[sys.executable, "-c", "print('fine')"])

    outcomes = sorted(
        SweepRunner.execute(runs=[broken, healthy], max_workers=2, retry_policy=_NO_BACKOFF),
        key=lambda o: o.run.seed,
    )

    # The failed spawn is reported as an ordinary failed outcome...
    assert outcomes[0].succeeded is False
    assert outcomes[0].returncode == -1
    # ...carrying the original exception, not a substitute for it. Nothing here
    # may mention UnboundLocalError: that would mean the real cause was masked.
    assert "FileNotFoundError" in outcomes[0].output
    assert "UnboundLocalError" not in outcomes[0].output
    # ...written to that run's log.txt, exactly as a child's own stderr would be.
    assert "FileNotFoundError" in (broken.output_dir / "log.txt").read_text()

    # ...and the sibling run in the same sweep still ran to completion. This is
    # the assertion that pins the actual contract.
    assert outcomes[1].succeeded is True
    assert "fine" in outcomes[1].output


def test_a_run_that_cannot_be_launched_still_fails_the_sweep(*, tmp_path: Path) -> None:
    """Reporting instead of raising must not downgrade a hard failure to a silent
    pass: `main` exits non-zero if any outcome failed, and rc=-1 is a failure."""
    run = _fake_run(tmp_path=tmp_path, seed=0, command=["definitely-not-a-real-executable"])
    outcomes = SweepRunner.execute(runs=[run], max_workers=1, retry_policy=_NO_BACKOFF)
    assert [outcome for outcome in outcomes if not outcome.succeeded] == outcomes


# Captured at import, before any test monkeypatches subprocess.run, so the fake
# below can delegate to the genuine article without recursing into itself.
_REAL_SUBPROCESS_RUN = subprocess.run


class _FlakySpawn(BaseModel):
    """Stands in for `subprocess.run` and fails the *spawn* of one specific
    command on its first `failures` attempts -- what fork() does under memory
    pressure. Any other command is handed straight to the real thing, so a single
    sweep can contain both a flaky run and a healthy sibling.

    This has to be faked rather than provoked: the real trigger is the machine
    running out of memory, which a test may not do to the box it runs on."""

    command: list[str]
    failures: int
    attempts: int = 0

    def __call__(self, *args, **kwargs):
        if list(args[0]) != self.command:
            return _REAL_SUBPROCESS_RUN(*args, **kwargs)
        self.attempts += 1
        if self.attempts <= self.failures:
            # Numbered so a test can prove *which* attempt's traceback survived.
            raise OSError(f"synthetic spawn failure {self.attempts}")
        return _REAL_SUBPROCESS_RUN(*args, **kwargs)


def test_a_transient_spawn_failure_is_retried_until_the_run_launches(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The motivating case: fork() fails under memory pressure, a sibling worker
    exits a second later, and the same command then launches fine. Losing a
    ~40-minute run to that is pure waste, so it is retried."""
    run = _fake_run(tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "print('fine')"])
    flaky = _FlakySpawn(command=run.command, failures=2)
    monkeypatch.setattr(subprocess, "run", flaky)

    outcomes = SweepRunner.execute(runs=[run], max_workers=1, retry_policy=_NO_BACKOFF)

    assert flaky.attempts == 3
    assert outcomes[0].succeeded is True
    assert "fine" in outcomes[0].output
    # The retries are recorded, not silent: several runs needing them is a
    # machine-health signal, so it has to outlive the scrollback.
    assert outcomes[0].spawn_attempts == 3
    assert _timing_of(run=run).spawn_attempts == 3
    # Every failed attempt's traceback is kept, not just the last one.
    log = (run.output_dir / "log.txt").read_text()
    assert "synthetic spawn failure 1" in log
    assert "synthetic spawn failure 2" in log


def test_a_spawn_that_fails_every_attempt_is_reported_with_the_last_traceback(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting the retries changes nothing about the contract: still reported
    as rc=-1 with the traceback as output, still never raised, and the other runs
    in the grid still complete."""
    broken = _fake_run(tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "print('never')"])
    healthy = _fake_run(tmp_path=tmp_path, seed=1, command=[sys.executable, "-c", "print('fine')"])
    flaky = _FlakySpawn(command=broken.command, failures=99)
    monkeypatch.setattr(subprocess, "run", flaky)

    outcomes = sorted(
        SweepRunner.execute(runs=[broken, healthy], max_workers=1, retry_policy=_NO_BACKOFF),
        key=lambda outcome: outcome.run.seed,
    )

    assert flaky.attempts == 3  # bounded: it does not retry forever
    assert outcomes[0].succeeded is False
    assert outcomes[0].returncode == -1
    assert outcomes[0].spawn_attempts == 3
    # The *last* attempt's traceback is the one that has to survive -- it is the
    # state the run was actually abandoned in.
    assert "OSError" in outcomes[0].output
    assert "synthetic spawn failure 3" in outcomes[0].output
    assert "synthetic spawn failure 3" in (broken.output_dir / "log.txt").read_text()

    assert outcomes[1].succeeded is True
    assert "fine" in outcomes[1].output


def test_a_run_that_started_and_exited_non_zero_is_never_retried(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the retry exists to respect. A child that *ran* and failed
    is deterministic in `--seed`, so re-running it just burns another ~40 minutes
    reproducing the identical failure -- one broken seed must cost 1x, not 3x.
    Only a spawn that never produced an exit code at all is transient."""
    run = _fake_run(
        tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "raise SystemExit(3)"]
    )
    # failures=0: the spawn always succeeds, so every attempt here is the child
    # itself exiting non-zero.
    counting = _FlakySpawn(command=run.command, failures=0)
    monkeypatch.setattr(subprocess, "run", counting)

    outcomes = SweepRunner.execute(runs=[run], max_workers=1, retry_policy=_NO_BACKOFF)

    assert counting.attempts == 1  # exactly one spawn, despite max_attempts=3
    assert outcomes[0].returncode == 3
    assert outcomes[0].spawn_attempts == 1


def test_max_attempts_of_one_disables_retrying(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--max-spawn-attempts 1` is the documented escape hatch, so it has to
    really mean "spawn once and report whatever happened"."""
    run = _fake_run(tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "print('never')"])
    flaky = _FlakySpawn(command=run.command, failures=99)
    monkeypatch.setattr(subprocess, "run", flaky)

    outcomes = SweepRunner.execute(
        runs=[run], max_workers=1, retry_policy=SpawnRetryPolicy(max_attempts=1)
    )

    assert flaky.attempts == 1
    assert outcomes[0].returncode == -1
    assert outcomes[0].spawn_attempts == 1


def test_retry_policy_backs_off_exponentially_from_a_bounded_default() -> None:
    """Exponential rather than fixed: if the box really is out of memory, backing
    off harder is what gives the resident workers time to drain. The bound has a
    floor of 1 because argparse will not check it, and 0 would silently mean
    "never even try" -- an rc=-1 with no exception to explain it."""
    policy = SpawnRetryPolicy()
    assert policy.max_attempts == 3
    assert policy.backoff_seconds(attempt=1) == 2.0
    assert policy.backoff_seconds(attempt=2) == 4.0
    with pytest.raises(ValidationError):
        SpawnRetryPolicy(max_attempts=0)


def test_timing_timestamps_are_timezone_aware_iso_8601(*, tmp_path: Path) -> None:
    """A naive timestamp cannot be compared against another machine's, or across a
    DST boundary. The epoch fields are the derived convenience, so they have to
    agree with the primary ISO ones."""
    run = _fake_run(tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "pass"])
    SweepRunner.execute(runs=[run], max_workers=1)
    timing = _timing_of(run=run)
    start = datetime.fromisoformat(timing.start_time)
    end = datetime.fromisoformat(timing.end_time)
    assert start.tzinfo is not None and end.tzinfo is not None
    assert start.timestamp() == pytest.approx(timing.start_epoch_seconds)
    assert end >= start


def test_elapsed_agrees_with_the_wall_clock_absent_a_clock_step(*, tmp_path: Path) -> None:
    """Elapsed is measured with time.monotonic() rather than by subtracting the
    two timestamps, because a 40-minute run is long enough for NTP to step the
    wall clock underneath it. This can only check that the two agree when nothing
    steps the clock -- the divergence it guards against is not reproducible in a
    test -- so it is a sanity check on the units, not a proof of the clock source."""
    run = _fake_run(
        tmp_path=tmp_path, seed=0, command=[sys.executable, "-c", "import time; time.sleep(0.05)"]
    )
    SweepRunner.execute(runs=[run], max_workers=1)
    timing = _timing_of(run=run)
    wall_elapsed = timing.end_epoch_seconds - timing.start_epoch_seconds
    assert timing.elapsed_seconds == pytest.approx(wall_elapsed, abs=0.5)


def test_timing_records_both_concurrency_signals_separately(*, tmp_path: Path) -> None:
    """The sweep-local count and the machine-wide count answer different
    questions: other agents' sweeps load the same box, so the sweep-local number
    is only a lower bound on what a run actually competed with. Conflating them
    would misattribute another sweep's load to this one's --max-workers."""
    runs = [
        _fake_run(
            tmp_path=tmp_path,
            seed=seed,
            command=[sys.executable, "-c", "import time; time.sleep(0.05)"],
        )
        for seed in range(3)
    ]
    SweepRunner.execute(runs=runs, max_workers=3)
    for run in runs:
        timing = _timing_of(run=run)
        assert 1 <= timing.sweep_runs_in_flight_at_start <= 3
        assert 1 <= timing.sweep_runs_in_flight_at_end <= 3
        assert timing.max_workers == 3
        # Machine-wide samples are Optional by design ("unknown" is not "zero"),
        # but the field pair always exists at both ends of the run.
        for sample in (timing.machine_at_start, timing.machine_at_end):
            assert sample.cli_processes is None or sample.cli_processes >= 0
            assert sample.load_average_1min is None or sample.load_average_1min >= 0


def test_every_run_of_one_sweep_shares_a_sweep_id(*, tmp_path: Path) -> None:
    """Grouping by sweep is what makes --max-workers interpretable: runs of one
    sweep shared a worker budget, runs of two overlapping sweeps did not."""
    runs = [
        _fake_run(tmp_path=tmp_path, seed=seed, command=[sys.executable, "-c", "pass"])
        for seed in range(2)
    ]
    SweepRunner.execute(runs=runs, max_workers=2)
    assert len({_timing_of(run=run).sweep_id for run in runs}) == 1

    second = _fake_run(tmp_path=tmp_path / "again", seed=0, command=[sys.executable, "-c", "pass"])
    SweepRunner.execute(runs=[second], max_workers=1)
    assert _timing_of(run=second).sweep_id != _timing_of(run=runs[0]).sweep_id


def test_timing_does_not_touch_what_the_run_itself_wrote(*, tmp_path: Path) -> None:
    """The reason timing lives in its own file: stats.json's byte-stability is how
    this project verifies that a change did not alter results, so nothing here may
    reach into it."""
    stats = tmp_path / "fake" / "0" / "stats.json"
    run = _fake_run(
        tmp_path=tmp_path,
        seed=0,
        command=[
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(stats)!r}).write_text('{{\"evaluations\": []}}')",
        ],
    )
    SweepRunner.execute(runs=[run], max_workers=1)
    assert stats.read_text() == '{"evaluations": []}'
    assert (run.output_dir / "timing.json").is_file()


def test_in_flight_counter_reports_the_calling_run_inclusively() -> None:
    """Entry and exit samples have to be on the same scale, or a run that starts
    alone and finishes alone would read 1 and 0."""
    counter = InFlightCounter()
    assert counter.enter() == 1
    assert counter.enter() == 2
    assert counter.leave() == 2
    assert counter.leave() == 1


def test_machine_sampler_counts_cli_processes_from_a_process_table(*, tmp_path: Path) -> None:
    """Counts every `hitl_pmp.cli` on the box, including other agents' sweeps --
    that is the point of the machine-wide signal. Reads /proc directly rather than
    spawning `pgrep`, whose `-c` prints 0 *and* exits non-zero on no match."""
    for pid, cmdline in (
        ("11", b"python\x00-m\x00hitl_pmp.cli\x00--env\x00lightswitch\x00"),
        ("12", b"python\x00-m\x00hitl_pmp.cli\x00"),
        ("13", b"python\x00-m\x00scripts.run_sweep\x00"),
        ("self", b"not-a-pid-directory\x00"),
    ):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(cmdline)
    assert MachineSampler.count_cli_processes(proc_dir=tmp_path) == 2


def test_machine_sampler_reports_unknown_rather_than_zero_without_proc(*, tmp_path: Path) -> None:
    """ "No /proc" (macOS) and "no runs in flight" are different facts, and a zero
    in place of the first would silently corrupt a concurrency regression."""
    assert MachineSampler.count_cli_processes(proc_dir=tmp_path / "nonexistent") is None


def test_sweep_output_round_trips_through_the_analysis_layout(*, tmp_path: Path) -> None:
    """End-to-end shape check: a planned run's output_dir is exactly where a
    stats.json has to land for `analysis` to find it by globbing."""
    (run,) = SweepRunner.plan(
        env="lightswitch",
        methods=["random-skills"],
        seeds=[0],
        results_root=tmp_path,
        shared_args=[],
        method_args={},
    )
    run.output_dir.mkdir(parents=True)
    (run.output_dir / "stats.json").write_text(
        json.dumps({"evaluations": [[0, 1, 2]], "task_name": "default"})
    )
    assert sorted(tmp_path.glob("*/*/stats.json")) == [run.output_dir / "stats.json"]
