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
"""

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict


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
    more to recover from than a single missing datapoint."""

    model_config = ConfigDict(frozen=True)

    run: SweepRun
    returncode: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


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
    def execute(*, runs: list[SweepRun], max_workers: int) -> list[SweepOutcome]:
        """Runs every command concurrently. Threads (not processes) because each
        worker only waits on a subprocess, so the GIL is never the bottleneck."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda run: SweepRunner._execute_one(run=run), runs))

    @staticmethod
    def _execute_one(*, run: SweepRun) -> SweepOutcome:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        # Pin each child to one math thread: workers already run concurrently, so
        # letting every child grab all cores oversubscribes and slows the sweep
        # down. Results are unaffected -- determinism is thread-count independent
        # (pinned by tests/scripts/test_reproducibility.py).
        child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        completed = subprocess.run(  # noqa: S603
            run.command, capture_output=True, text=True, env=child_env, check=False
        )
        output = completed.stdout + completed.stderr
        (run.output_dir / "log.txt").write_text(output)
        status = "ok" if completed.returncode == 0 else f"FAILED rc={completed.returncode}"
        print(f"[{status}] {run.method} seed={run.seed}", flush=True)
        return SweepOutcome(run=run, returncode=completed.returncode, output=output)


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
    print(f"Running {len(runs)} runs with {args.max_workers} workers...", flush=True)
    outcomes = SweepRunner.execute(runs=runs, max_workers=args.max_workers)

    failures = [outcome for outcome in outcomes if not outcome.succeeded]
    print(f"\n{len(outcomes) - len(failures)}/{len(outcomes)} runs succeeded.")
    for failure in failures:
        print(f"  FAILED {failure.run.method} seed={failure.run.seed}: {failure.output[-400:]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
