"""Run a (method x seed) experiment sweep in parallel, writing results into the
layout `analysis/` already expects.

Every experiment in this project is the same shape: run `hitl_pmp.cli` once per
(method, seed), point each at `<results-root>/<method>/<seed>/`, then hand that
tree to an `analysis/` script. This exists so that is a supported, reviewable,
re-runnable command rather than a throwaway shell loop rewritten per experiment.

Why it isn't in `analysis/`: `analysis/` is strictly post-run (see its README and
CLAUDE.md) -- it reads `--output-dir` output and never drives a simulation. This
*does* drive simulations, so it lives in `scripts/` instead, mirroring the sibling
`hitl-practice` repo's own `scripts/` convention. It only ever shells out to the
CLI; it imports nothing from `hitl_pmp`, so it cannot accidentally reach past that
boundary.

Seeds are **fixed and deterministic** -- `--num-seeds 10` means exactly seeds
0..9, never a random draw. A sweep has to reproduce to the same numbers when
re-run months later, and the paper's own protocol ("we run 10 random seeds of
each approach") is a fixed set of seeds, not randomly chosen ones. Every source
of randomness downstream is already seeded from that one integer (task sampling,
skill/parameter sampling, and torch), which `tests/scripts/test_reproducibility.py`
pins.

Example -- the full EES reproduction at the paper's Light Switch protocol:

    python -m scripts.run_sweep \\
        --env lightswitch \\
        --methods ees random-skills skill-oracle \\
        --num-seeds 10 \\
        --results-root results/ees \\
        --shared-args "--grid-size 25 --num-test-tasks 10" \\
        --method-args "ees=--num-cycles 10 --max-steps-per-interaction 150" \\
        --method-args "random-skills=--num-cycles 10 --max-steps-per-interaction 150"

Per-method args are not sugar: methods genuinely do not share a flag set (a
`--method skill-oracle` run rejects `--num-cycles` outright), so a single shared
argument string cannot express a real sweep.

Every run also writes a `timing.json` next to its `stats.json` (see `RunTiming`)
recording when it ran, how long it took, and how loaded the machine was while it
ran -- so "how long does a run take, and does concurrency help?" is answerable
from recorded data instead of re-measured by hand. It is a separate file
precisely so `stats.json` stays byte-identical run to run; nothing in it is ever
an input to a reproducibility comparison.
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr


class SweepRun(BaseModel):
    """One planned (method, seed) invocation. Planning is separated from running
    so the whole sweep can be inspected -- or asserted about in tests -- without
    executing anything."""

    model_config = ConfigDict(frozen=True)

    method: str
    seed: int
    output_dir: Path
    command: list[str]


class SweepOutcome(BaseModel):
    """What actually happened to one run. A failure is reported, not raised: one
    bad seed must not abort the other 29, since an interrupted sweep costs far
    more to recover from than a single missing datapoint. That holds for a run
    that never *started* too -- a failed spawn is reported as `returncode == -1`
    with the traceback as `output`, not as an exception out of the sweep (see
    `SweepRunner._execute_one`)."""

    model_config = ConfigDict(frozen=True)

    run: SweepRun
    returncode: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class RunTiming(BaseModel):
    """When one run ran, how long it took, and how busy the machine was while it
    ran. Serialized to `timing.json` inside that run's own `--output-dir`, beside
    its `stats.json`.

    **Why a separate file.** `stats.json` is the serialized `core.Metrics`, and
    its byte-stability is load-bearing: this project verifies a change by running
    a fixed seed before and after and diffing that file. A timestamp inside it
    would make every run differ from every other by construction, and would break
    that check for every open PR that uses it. So timing lands beside it and is
    excluded, by filename, from any reproducibility comparison. `sweep_id`,
    timestamps and load samples are all non-reproducible on purpose -- that is
    what makes them useful and exactly why they may not live in `stats.json`.

    **Two different concurrency signals, which must not be conflated.**
    `sweep_runs_in_flight_*` counts only *this* sweep's own child runs. Other
    agents on the same box run their own sweeps, so it is a lower bound on the
    real competition for the CPU. `machine_*` samples the whole machine
    (`hitl_pmp.cli` process count, 1-minute load average) and therefore includes
    this sweep's children plus everyone else's. Regressing wall-clock on the
    sweep-local count when the box was shared would attribute another agent's
    load to this sweep's `--max-workers`, which is the specific mistake this
    field pair exists to prevent.
    """

    model_config = ConfigDict(frozen=True)

    sweep_id: str
    method: str
    seed: int

    # ISO 8601 with a UTC offset is the *primary* timestamp: it is unambiguous
    # across machines and DST, sorts lexicographically within one offset, and is
    # readable in a report without conversion. Epoch seconds are recorded too,
    # for arithmetic without parsing, but they are the derived convenience.
    start_time: str
    end_time: str
    start_epoch_seconds: float
    end_epoch_seconds: float
    # Measured with time.monotonic(), NOT end_epoch - start_epoch: the wall clock
    # can be stepped by NTP mid-run, and a 40-minute run is long enough for that
    # to matter. The two will disagree slightly; monotonic is the truthful one.
    elapsed_seconds: float

    returncode: int
    succeeded: bool

    # The sweep's own configuration, so a record is interpretable on its own.
    max_workers: int
    cpu_count: int | None

    # This sweep's own children only -- see the class docstring. Point samples at
    # this run's entry and exit, not a time-weighted average: a run that starts
    # alone and finishes among 22 siblings shows 1 and 22, and the truth in
    # between is bounded above by max_workers. Both **include this run itself**.
    sweep_runs_in_flight_at_start: int
    sweep_runs_in_flight_at_end: int

    # Machine-wide -- this sweep's children *and* every other agent's. Unlike the
    # pair above these **exclude this run itself**, and unavoidably so: the start
    # sample is taken before its child is spawned and the end sample after that
    # child has exited, because subprocess.run blocks in between. A reader
    # comparing the two signals has to add one to put them on the same scale
    # (analysis/run_timing.py does).
    machine_at_start: "MachineSample"
    machine_at_end: "MachineSample"


class MachineSample(BaseModel):
    """The whole box at one instant, sampled twice per run (entry and exit).

    Both fields are machine-wide, never sweep-local. Either can be None where the
    platform does not expose it (no /proc, no getloadavg), which is a real state a
    reader has to handle rather than something to fake with a zero.
    """

    model_config = ConfigDict(frozen=True)

    # Count of running processes whose command line mentions `hitl_pmp.cli`, i.e.
    # the instantaneous number of runs in flight *anywhere on the machine*.
    cli_processes: int | None
    # os.getloadavg()[0]. Note this is a ~1-minute damped average, so it lags:
    # sampled at a run's start it describes the machine the run *entered*, and
    # sampled at its end it approximates what the run actually *experienced*.
    load_average_1min: float | None


class MachineSampler:
    """A static-method container, never instantiated -- no state between calls."""

    @staticmethod
    def sample() -> MachineSample:
        return MachineSample(
            cli_processes=MachineSampler.count_cli_processes(),
            load_average_1min=MachineSampler.load_average_1min(),
        )

    @staticmethod
    def count_cli_processes(*, proc_dir: Path = Path("/proc")) -> int | None:
        """Reads /proc directly rather than shelling out to `pgrep`: a sample
        costs no process spawn (spawn cost is a measured bottleneck in this
        workload), and `pgrep -c` prints 0 *and* exits non-zero on no match,
        a wart that has already produced a corrupt measurement in this project.

        Counts the whole machine, not this sweep -- see RunTiming's docstring.
        Returns None where there is no /proc (e.g. macOS), since "unknown" and
        "zero" are different facts. `proc_dir` is injectable only so a test can
        exercise both branches without a real process table.
        """
        if not proc_dir.is_dir():
            return None
        count = 0
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes()
            except OSError:
                continue  # The process exited mid-scan; it is no longer in flight.
            if b"hitl_pmp.cli" in cmdline:
                count += 1
        return count

    @staticmethod
    def load_average_1min() -> float | None:
        try:
            return os.getloadavg()[0]
        except (OSError, AttributeError):  # pragma: no cover - platform-dependent
            return None


class InFlightCounter(BaseModel):
    """How many of *this* sweep's runs are executing right now. Real per-sweep
    state, so it is a real instance passed explicitly to each worker rather than
    a class-level global -- see CLAUDE.md's core/ section for why this project
    stopped using mutable ClassVars for exactly this shape of state.

    Threads mutate it concurrently, hence the lock. Both methods report the count
    *including* the calling run, so entry and exit samples are on the same scale.
    """

    model_config = ConfigDict(frozen=True)

    _count: int = PrivateAttr(default=0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def enter(self) -> int:
        with self._lock:
            self._count += 1
            return self._count

    def leave(self) -> int:
        with self._lock:
            count = self._count
            self._count -= 1
            return count


class SweepRunner:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def default_seeds(*, num_seeds: int) -> list[int]:
        """Seeds 0..num_seeds-1 -- fixed, never randomly drawn. See this module's
        own docstring for why."""
        return list(range(num_seeds))

    @staticmethod
    def parse_method_args(*, raw: list[str]) -> dict[str, list[str]]:
        """Parses repeated `--method-args "method=--flag value"` entries."""
        parsed: dict[str, list[str]] = {}
        for entry in raw:
            method, separator, args = entry.partition("=")
            if not separator or not method:
                raise ValueError(f"--method-args must look like method=args, got {entry!r}")
            parsed[method] = shlex.split(args)
        return parsed

    @staticmethod
    def plan(
        *,
        env: str,
        methods: list[str],
        seeds: list[int],
        results_root: Path,
        shared_args: list[str],
        method_args: dict[str, list[str]],
    ) -> list[SweepRun]:
        runs: list[SweepRun] = []
        for method in methods:
            for seed in seeds:
                output_dir = results_root / method / str(seed)
                runs.append(
                    SweepRun(
                        method=method,
                        seed=seed,
                        output_dir=output_dir,
                        command=[
                            sys.executable,
                            "-m",
                            "hitl_pmp.cli",
                            "--env",
                            env,
                            "--method",
                            method,
                            "--seed",
                            str(seed),
                            "--output-dir",
                            str(output_dir),
                            *shared_args,
                            *method_args.get(method, []),
                        ],
                    )
                )
        return runs

    @staticmethod
    def new_sweep_id() -> str:
        """Identifies one `run_sweep` invocation, so an analysis can group runs by
        the sweep they came from (runs of one sweep shared a `--max-workers`;
        runs of two overlapping sweeps did not).

        Timestamp for sortability and so an id is readable on its own, pid for
        the launching process, and a random suffix because neither of those is
        actually unique: two sweeps started by one process inside the same second
        collide on both, which a test caught.
        """
        return f"{datetime.now().astimezone():%Y%m%dT%H%M%S%z}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def execute(
        *, runs: list[SweepRun], max_workers: int, sweep_id: str | None = None
    ) -> list[SweepOutcome]:
        """Runs every command concurrently. Threads (not processes) because each
        worker only waits on a subprocess, so the GIL is never the bottleneck."""
        resolved_sweep_id = sweep_id if sweep_id is not None else SweepRunner.new_sweep_id()
        in_flight = InFlightCounter()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda run: SweepRunner._execute_one(
                        run=run,
                        sweep_id=resolved_sweep_id,
                        max_workers=max_workers,
                        in_flight=in_flight,
                    ),
                    runs,
                )
            )

    @staticmethod
    def _execute_one(
        *, run: SweepRun, sweep_id: str, max_workers: int, in_flight: InFlightCounter
    ) -> SweepOutcome:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        # Pin each child to one math thread: workers already run concurrently, so
        # letting every child grab all cores oversubscribes and slows the sweep
        # down. Results are unaffected -- determinism is thread-count independent
        # (pinned by tests/scripts/test_reproducibility.py).
        child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        in_flight_at_start = in_flight.enter()
        machine_at_start = MachineSampler.sample()
        start_time = datetime.now().astimezone()
        start_monotonic = time.monotonic()
        # -1 stands for "the child never reported a status", i.e. subprocess.run
        # itself raised -- the spawn failed, so there was never an exit code to
        # collect. `check=False` means a child that runs and *fails* comes back
        # normally with its own non-zero code; only failing to start lands here.
        # That is not hypothetical on this box: memory pressure makes fork()
        # raise OSError, and the runs most likely to hit it are the ones started
        # while every other worker is already resident.
        #
        # It is caught rather than allowed to propagate because propagating
        # breaks this module's central promise -- `list(executor.map(...))`
        # re-raises the first exception any worker raised, so one failed spawn
        # would abort the other 29 runs mid-sweep. See SweepOutcome's docstring:
        # a failure is reported, not raised. The traceback becomes the outcome's
        # `output`, so the original exception is *reported*, never swallowed:
        # it is printed in the failure summary and written to that run's
        # log.txt, exactly like a child's own stderr would be.
        #
        # The timing is written either way by the finally below: a failed run is
        # data, and losing its timing would silently bias any later wall-clock
        # analysis toward runs that happened to succeed.
        returncode = -1
        output = ""
        try:
            completed = subprocess.run(  # noqa: S603
                run.command, capture_output=True, text=True, env=child_env, check=False
            )
            returncode = completed.returncode
            output = completed.stdout + completed.stderr
        except Exception:  # noqa: BLE001 -- deliberately broad; see above
            output = traceback.format_exc()
        finally:
            elapsed_seconds = time.monotonic() - start_monotonic
            end_time = datetime.now().astimezone()
            machine_at_end = MachineSampler.sample()
            in_flight_at_end = in_flight.leave()
            SweepRunner._write_timing(
                run=run,
                timing=RunTiming(
                    sweep_id=sweep_id,
                    method=run.method,
                    seed=run.seed,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    start_epoch_seconds=start_time.timestamp(),
                    end_epoch_seconds=end_time.timestamp(),
                    elapsed_seconds=elapsed_seconds,
                    returncode=returncode,
                    succeeded=returncode == 0,
                    max_workers=max_workers,
                    cpu_count=os.cpu_count(),
                    sweep_runs_in_flight_at_start=in_flight_at_start,
                    sweep_runs_in_flight_at_end=in_flight_at_end,
                    machine_at_start=machine_at_start,
                    machine_at_end=machine_at_end,
                ),
            )
        (run.output_dir / "log.txt").write_text(output)
        status = "ok" if returncode == 0 else f"FAILED rc={returncode}"
        print(f"[{status}] {run.method} seed={run.seed}", flush=True)
        return SweepOutcome(run=run, returncode=returncode, output=output)

    @staticmethod
    def _write_timing(*, run: SweepRun, timing: RunTiming) -> None:
        """Written per run rather than once per sweep, and immediately rather than
        at the end: a sweep that is interrupted (or a box that OOMs) still leaves
        the timings of everything that already finished. Re-running a sweep
        overwrites it, same as `log.txt`."""
        (run.output_dir / "timing.json").write_text(timing.model_dump_json(indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--shared-args", default="", help="Flags applied to every run, as one string."
    )
    parser.add_argument(
        "--method-args",
        action="append",
        default=[],
        help='Repeatable, "method=--flag value". Flags for one method only.',
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Concurrent runs. Defaults to the CPU count.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs = SweepRunner.plan(
        env=args.env,
        methods=args.methods,
        seeds=SweepRunner.default_seeds(num_seeds=args.num_seeds),
        results_root=args.results_root,
        shared_args=shlex.split(args.shared_args),
        method_args=SweepRunner.parse_method_args(raw=args.method_args),
    )
    sweep_id = SweepRunner.new_sweep_id()
    print(
        f"Running {len(runs)} runs with {args.max_workers} workers (sweep {sweep_id})...",
        flush=True,
    )
    outcomes = SweepRunner.execute(runs=runs, max_workers=args.max_workers, sweep_id=sweep_id)

    failures = [outcome for outcome in outcomes if not outcome.succeeded]
    print(f"\n{len(outcomes) - len(failures)}/{len(outcomes)} runs succeeded.")
    for failure in failures:
        print(f"  FAILED {failure.run.method} seed={failure.run.seed}: {failure.output[-400:]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
