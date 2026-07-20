import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """LightSwitchCli.run() mutates shared ClassVar state (grid_size, seed, ...) as
    a side effect; snapshot and restore it around every test in this file so tests
    can't leak configuration into each other regardless of execution order."""
    original_grid_size = LightSwitchEnvironment.grid_size
    original_light_on_tolerance = LightSwitchEnvironment.light_on_tolerance
    original_same_position_tolerance = LightSwitchEnvironment.same_position_tolerance
    original_canonical_light_target = LightSwitchEnvironment.canonical_light_target
    original_seed = LightSwitchTasks.seed
    original_test_env_seed_offset = LightSwitchTasks.test_env_seed_offset
    original_target_low = LightSwitchTasks.target_low
    original_target_high = LightSwitchTasks.target_high
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


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global --seed/--num-test-tasks (added by
    Cli.add_global_arguments there, not by LightSwitchCli.add_arguments) plus this
    domain's own flags, so LightSwitchCli.run can be exercised in isolation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    assert args.num_test_tasks == 20
    assert args.policy == "oracle"
    assert args.grid_size == LightSwitchEnvironment.grid_size
    assert args.light_on_tolerance == LightSwitchEnvironment.light_on_tolerance
    assert args.same_position_tolerance == LightSwitchEnvironment.same_position_tolerance
    assert args.canonical_light_target == LightSwitchEnvironment.canonical_light_target
    assert args.seed == LightSwitchTasks.seed
    assert args.test_env_seed_offset == LightSwitchTasks.test_env_seed_offset
    assert args.target_low == LightSwitchTasks.target_low
    assert args.target_high == LightSwitchTasks.target_high


def test_run_with_oracle_policy_solves_every_sampled_task() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5"])
    assert LightSwitchCli.run(args=args) == (5, 5)


def test_run_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    LightSwitchCli.run(args=args)
    first_target = LightSwitchTasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    LightSwitchCli.run(args=args)
    second_target = LightSwitchTasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    assert first_target == second_target


def test_run_respects_a_smaller_grid_size_override() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "4", "--grid-size", "3"])
    assert LightSwitchCli.run(args=args) == (4, 4)
    assert LightSwitchEnvironment.grid_size == 3


def test_run_rejects_an_unknown_policy_choice() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--policy", "not-a-real-policy"])


def test_run_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2"])
    LightSwitchCli.run(args=args)
    assert list(tmp_path.iterdir()) == []


def test_run_with_output_dir_writes_a_video_file(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2", "--output-dir", str(tmp_path)])
    LightSwitchCli.run(args=args)
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_with_output_dir_writes_matching_stats_json(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--output-dir", str(tmp_path)])
    num_solved, num_test_tasks = LightSwitchCli.run(args=args)

    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["env"] == "lightswitch"
    assert stats["policy"] == "oracle"
    assert stats["num_test_tasks"] == num_test_tasks == 3
    assert stats["num_solved"] == num_solved == 3
    assert stats["success_rate"] == 1.0
    assert stats["per_task_solved"] == [True, True, True]


def test_run_with_output_dir_creates_missing_directories(*, tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "results"
    args = _build_parser().parse_args(["--num-test-tasks", "1", "--output-dir", str(output_dir)])
    LightSwitchCli.run(args=args)
    assert (output_dir / "episode.mp4").exists()
    assert (output_dir / "stats.json").exists()
