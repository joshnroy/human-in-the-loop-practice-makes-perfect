import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli, SkillOracleCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """LightSwitchCli.run_method() mutates shared ClassVar state (grid_size, seed,
    Problem.env/tasks, Metrics.evaluations, ...) as a side effect; snapshot and
    restore it around every test in this file so tests can't leak configuration
    into each other regardless of execution order."""
    original_grid_size = LightSwitchEnvironment.grid_size
    original_light_on_tolerance = LightSwitchEnvironment.light_on_tolerance
    original_same_position_tolerance = LightSwitchEnvironment.same_position_tolerance
    original_canonical_light_target = LightSwitchEnvironment.canonical_light_target
    original_seed = LightSwitchTasks.seed
    original_test_env_seed_offset = LightSwitchTasks.test_env_seed_offset
    original_target_low = LightSwitchTasks.target_low
    original_target_high = LightSwitchTasks.target_high
    original_problem_env = getattr(Problem, "env", None)
    original_problem_tasks = getattr(Problem, "tasks", None)
    try:
        yield
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size
        LightSwitchEnvironment.light_on_tolerance = original_light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = original_same_position_tolerance
        LightSwitchEnvironment.canonical_light_target = original_canonical_light_target
        LightSwitchTasks.test_env_seed_offset = original_test_env_seed_offset
        LightSwitchTasks.target_low = original_target_low
        LightSwitchTasks.target_high = original_target_high
        LightSwitchTasks.set_seed(seed=original_seed)
        if original_problem_env is not None:
            Problem.env = original_problem_env
        if original_problem_tasks is not None:
            Problem.tasks = original_problem_tasks
        Metrics.reset()


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global --seed/--num-test-tasks/--output-dir (added
    by Cli.add_global_arguments there, not by LightSwitchCli.add_arguments) plus
    this domain's own flags and SkillOracleCli's (a no-op), so LightSwitchCli/
    SkillOracleCli can be exercised in isolation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    SkillOracleCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    assert args.num_test_tasks == 20
    assert args.grid_size == LightSwitchEnvironment.grid_size
    assert args.light_on_tolerance == LightSwitchEnvironment.light_on_tolerance
    assert args.same_position_tolerance == LightSwitchEnvironment.same_position_tolerance
    assert args.canonical_light_target == LightSwitchEnvironment.canonical_light_target
    assert args.seed == LightSwitchTasks.seed
    assert args.test_env_seed_offset == LightSwitchTasks.test_env_seed_offset
    assert args.target_low == LightSwitchTasks.target_low
    assert args.target_high == LightSwitchTasks.target_high


def test_skill_oracle_cli_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5"])
    SkillOracleCli.run(args=args)
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    SkillOracleCli.run(args=args)
    first_target = LightSwitchTasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    SkillOracleCli.run(args=args)
    second_target = LightSwitchTasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    assert first_target == second_target


def test_run_method_respects_a_smaller_grid_size_override(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "4", "--grid-size", "3"])
    SkillOracleCli.run(args=args)
    assert "success rate: 4/4 (100%)" in capsys.readouterr().out
    assert LightSwitchEnvironment.grid_size == 3


def test_run_method_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2"])
    SkillOracleCli.run(args=args)
    assert list(tmp_path.iterdir()) == []


def test_run_method_with_output_dir_writes_a_video_file(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2", "--output-dir", str(tmp_path)])
    SkillOracleCli.run(args=args)
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_method_with_output_dir_creates_missing_directories(*, tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "results"
    args = _build_parser().parse_args(["--num-test-tasks", "1", "--output-dir", str(output_dir)])
    SkillOracleCli.run(args=args)
    assert (output_dir / "episode.mp4").exists()
