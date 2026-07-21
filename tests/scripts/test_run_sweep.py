import json
import sys
from pathlib import Path

import pytest

from scripts.run_sweep import SweepRun, SweepRunner


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
