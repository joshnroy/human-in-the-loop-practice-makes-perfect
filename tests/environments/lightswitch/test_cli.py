import argparse
from pathlib import Path

import pytest

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global --seed/--num-test-tasks/--output-dir (added
    by Cli.add_global_arguments there, not by LightSwitchCli.add_arguments) plus
    this domain's own flags, so LightSwitchCli can be exercised in isolation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    assert args.num_test_tasks == 20
    assert args.grid_size == LightSwitchEnvironment.model_fields["grid_size"].default
    assert args.light_on_tolerance == LightSwitchEnvironment.light_on_tolerance
    assert args.same_position_tolerance == LightSwitchEnvironment.same_position_tolerance
    assert (
        args.canonical_light_target
        == LightSwitchEnvironment.model_fields["canonical_light_target"].default
    )
    assert args.seed == 0
    assert (
        args.test_env_seed_offset == LightSwitchTasks.model_fields["test_env_seed_offset"].default
    )
    assert args.target_low == LightSwitchTasks.model_fields["target_low"].default
    assert args.target_high == LightSwitchTasks.model_fields["target_high"].default


def test_run_method_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5"])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    tasks_a = LightSwitchTasks(env=LightSwitchEnvironment(), seed=99)
    first_target = tasks_a.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    tasks_b = LightSwitchTasks(env=LightSwitchEnvironment(), seed=99)
    second_target = tasks_b.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    assert first_target == second_target


def test_run_method_respects_a_smaller_grid_size_override(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "4", "--grid-size", "3"])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 4/4 (100%)" in capsys.readouterr().out


def test_run_method_applies_light_on_tolerance_override() -> None:
    """light_on_tolerance/same_position_tolerance are the only two values left as
    ClassVar mutation (see LightSwitchEnvironment's own docstring for why) --
    confirm run_method's apply_config step still applies them."""
    original = LightSwitchEnvironment.light_on_tolerance
    try:
        args = _build_parser().parse_args(["--num-test-tasks", "1", "--light-on-tolerance", "0.5"])
        LightSwitchCli.run_method(
            args=args,
            method=SkillOracleMethod,
            num_cycles=0,
            max_steps_per_interaction=0,
        )
        assert LightSwitchEnvironment.light_on_tolerance == 0.5
    finally:
        LightSwitchEnvironment.light_on_tolerance = original


def test_run_method_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2"])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_method_with_output_dir_writes_a_video_file(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2", "--output-dir", str(tmp_path)])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_method_with_output_dir_creates_missing_directories(*, tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "results"
    args = _build_parser().parse_args(["--num-test-tasks", "1", "--output-dir", str(output_dir)])
    LightSwitchCli.run_method(
        args=args,
        method=SkillOracleMethod,
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert (output_dir / "episode.mp4").exists()
