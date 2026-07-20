import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.metrics import LightSwitchMetrics
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.cli import RandomSkillsCli


@pytest.fixture(autouse=True)
def _restore_state() -> Iterator[None]:
    """RandomSkillsCli.run() mutates shared ClassVar state (grid_size, seed,
    LightSwitchMetrics.evaluations, Problem.env/tasks, ...) as a side effect --
    snapshot/restore around every test in this file, same reason
    tests/environments/lightswitch/test_cli.py does."""
    original_grid_size = LightSwitchEnvironment.grid_size
    original_seed = LightSwitchTasks.seed
    original_problem_env = getattr(Problem, "env", None)
    original_problem_tasks = getattr(Problem, "tasks", None)
    try:
        yield
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size
        LightSwitchTasks.set_seed(seed=original_seed)
        if original_problem_env is not None:
            Problem.env = original_problem_env
        if original_problem_tasks is not None:
            Problem.tasks = original_problem_tasks
        LightSwitchMetrics.reset()


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags (added by Cli.add_global_arguments
    there, not by RandomSkillsCli.add_arguments) plus a minimal slice of
    LightSwitchCli's own --grid-size, so RandomSkillsCli.run can be exercised in
    isolation without going through the full Cli dispatcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--grid-size", type=int, default=LightSwitchEnvironment.grid_size)
    parser.add_argument(
        "--light-on-tolerance", type=float, default=LightSwitchEnvironment.light_on_tolerance
    )
    parser.add_argument(
        "--same-position-tolerance",
        type=float,
        default=LightSwitchEnvironment.same_position_tolerance,
    )
    parser.add_argument(
        "--canonical-light-target",
        type=float,
        default=LightSwitchEnvironment.canonical_light_target,
    )
    parser.add_argument(
        "--test-env-seed-offset", type=int, default=LightSwitchTasks.test_env_seed_offset
    )
    parser.add_argument("--target-low", type=float, default=LightSwitchTasks.target_low)
    parser.add_argument("--target-high", type=float, default=LightSwitchTasks.target_high)
    RandomSkillsCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.num_cycles == 10
    assert args.steps_per_cycle == 20


def test_run_records_one_evaluation_per_cycle_plus_the_initial_one() -> None:
    args = _build_parser().parse_args([
        "--grid-size",
        "5",
        "--num-cycles",
        "3",
        "--steps-per-cycle",
        "2",
        "--num-test-tasks",
        "2",
    ])
    RandomSkillsCli.run(args=args)
    curve = LightSwitchMetrics.task_training_curve()
    assert len(curve) == 4  # 1 initial + 1 per cycle
    assert [transitions for transitions, _ in curve] == [0, 2, 4, 6]


def test_run_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--grid-size", "5", "--num-cycles", "0"])
    RandomSkillsCli.run(args=args)
    assert list(tmp_path.iterdir()) == []


def test_run_with_output_dir_writes_stats_and_video(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args([
        "--grid-size",
        "5",
        "--num-cycles",
        "1",
        "--steps-per-cycle",
        "2",
        "--num-test-tasks",
        "2",
        "--output-dir",
        str(tmp_path),
    ])
    RandomSkillsCli.run(args=args)

    stats_path = tmp_path / "stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert len(stats["task_training_curve"]) == 2  # 1 initial + 1 cycle

    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert not (tmp_path / "episode.gif").exists()


def test_run_with_output_dir_and_gif_also_writes_a_gif(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args([
        "--grid-size",
        "5",
        "--num-cycles",
        "0",
        "--num-test-tasks",
        "2",
        "--output-dir",
        str(tmp_path),
        "--gif",
    ])
    RandomSkillsCli.run(args=args)
    gif_path = tmp_path / "episode.gif"
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_run_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--grid-size", "5", "--num-cycles", "0", "--seed", "99"])
    RandomSkillsCli.run(args=args)
    first_curve = LightSwitchMetrics.task_training_curve()

    args = _build_parser().parse_args(["--grid-size", "5", "--num-cycles", "0", "--seed", "99"])
    RandomSkillsCli.run(args=args)
    second_curve = LightSwitchMetrics.task_training_curve()

    assert first_curve == second_curve
