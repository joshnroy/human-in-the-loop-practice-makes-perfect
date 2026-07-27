import argparse
from pathlib import Path

import pytest

from hitl_pmp.environments.tossingroom.cli import TossingRoomCli
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags plus this domain's own, so TossingRoomCli
    can be exercised in isolation (mirrors Light Switch's test_cli)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    TossingRoomCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    fields = TossingRoomEnvironment.model_fields
    assert args.num_rooms == fields["num_rooms"].default
    assert args.start_room == fields["start_room"].default
    assert args.recycling_bin_room == fields["recycling_bin_room"].default
    assert args.trash_bin_room == fields["trash_bin_room"].default
    assert args.button_room == fields["button_room"].default
    assert args.blocked_right_from == fields["blocked_right_from"].default
    assert args.throw_tolerance == fields["throw_tolerance"].default
    assert args.target_low == TossingRoomTasks.model_fields["target_low"].default
    assert args.goal_type is None


def test_run_method_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "8"])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 8/8 (100%)" in capsys.readouterr().out


def test_run_method_forces_a_single_goal_type(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5", "--goal-type", "recycling"])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    a = TossingRoomTasks(env=TossingRoomEnvironment(), seed=99).sample_test_task()
    b = TossingRoomTasks(env=TossingRoomEnvironment(), seed=99).sample_test_task()
    assert a.initial_state.get(
        obj=TossingRoomEnvironment.recycling, feature_name="target_force"
    ) == b.initial_state.get(obj=TossingRoomEnvironment.recycling, feature_name="target_force")


def test_run_method_respects_a_larger_layout(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args([
        "--num-test-tasks",
        "4",
        "--num-rooms",
        "9",
        "--trash-bin-room",
        "8",
        "--button-room",
        "8",
        "--goal-type",
        "trash",
    ])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 4/4 (100%)" in capsys.readouterr().out


def test_run_method_with_output_dir_writes_a_video(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args([
        "--num-test-tasks",
        "1",
        "--goal-type",
        "recycling",
        "--output-dir",
        str(tmp_path),
    ])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    video = tmp_path / "episode.mp4"
    assert video.exists()
    assert video.stat().st_size > 0


def test_run_method_without_output_dir_writes_nothing(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2"])
    TossingRoomCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert list(tmp_path.iterdir()) == []
