import argparse
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossingroomsplitidentity import cli as tossingroomsplitidentity_cli
from hitl_pmp.environments.tossingroomsplitidentity.cli import TossingRoomSplitIdentityCli
from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.tasks import TossingRoomSplitIdentityTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags plus this domain's own, so
    TossingRoomSplitIdentityCli can be exercised in isolation (mirrors Light Switch's
    test_cli)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    TossingRoomSplitIdentityCli.add_arguments(parser=parser)
    return parser


def test_there_is_no_button_room_flag() -> None:
    """Each bin's button sits in that bin's own room, so --trash-bin-room and
    --recycling-bin-room place the buttons too. A separate --button-room would be a
    second knob that has to agree with them."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--button-room", "4"])


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    fields = TossingRoomSplitIdentityEnvironment.model_fields
    assert args.num_rooms == fields["num_rooms"].default
    assert args.start_room == fields["start_room"].default
    assert args.recycling_bin_room == fields["recycling_bin_room"].default
    assert args.trash_bin_room == fields["trash_bin_room"].default
    assert args.blocked_right_from == fields["blocked_right_from"].default
    assert args.throw_tolerance == fields["throw_tolerance"].default
    # The throw representation's whole configurable surface on the ENVIRONMENT: one
    # canonical value for hard_reset. The dynamics have no relation to configure.
    assert args.canonical_target_force == fields["canonical_target_force"].default
    task_fields = TossingRoomSplitIdentityTasks.model_fields
    # The two per-task cause ranges, identical in name and default to the causal arm's:
    # this arm draws the same causes and merely resolves them before they reach the State.
    assert args.distance_low == task_fields["distance_low"].default
    assert args.distance_high == task_fields["distance_high"].default
    assert args.weight_low == task_fields["weight_low"].default
    assert args.weight_high == task_fields["weight_high"].default
    assert args.goal_type is None


def test_the_default_cause_ranges_are_the_causal_arms() -> None:
    """The two arms draw the SAME tasks, so their draw ranges must be the same literals.
    Drawing `target_force` directly from a Uniform instead would have matched on a random
    force but not on the marginal -- the best fixed force would land 119/400 here against
    185/400 in the causal arm, so a state-blind sampler would not be comparable."""
    args = _build_parser().parse_args([])
    assert (args.distance_low, args.distance_high) == (1.0, 3.0)
    assert (args.weight_low, args.weight_high) == (0.5, 1.5)


@pytest.mark.parametrize(
    "flag",
    [
        "--reference-force",
        "--reference-distance",
        "--reference-weight",
        "--distance-coefficient",
        "--weight-coefficient",
        "--canonical-throw-distance",
        "--canonical-item-weight",
    ],
)
def test_there_are_no_relation_flags(*, flag: str) -> None:
    """Under the identity representation the DYNAMICS have no relation to configure: the
    required force IS `item.target_force`. The five constants still exist, as
    `TossingRoomSplitIdentityTasks` fields, because they shape the task DISTRIBUTION and
    are what makes this arm's marginal match the causal arm's -- but they are deliberately
    not flags, since changing one would silently break that match. A relation flag
    surviving the fork would be a knob that either does nothing or breaks the pairing."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args([flag, "0.5"])


def test_num_test_tasks_field_default_matches_the_global_flag_default() -> None:
    """Two independent literals (hitl_pmp/cli.py's --num-test-tasks and this domain's
    Tasks field) that must agree, since the field is what the fixed test-set composition
    is divided out of."""
    parser = argparse.ArgumentParser()
    Cli.add_global_arguments(parser=parser)
    args = parser.parse_args(["--env", "tossingroomsplitidentity", "--method", "skill-oracle"])
    assert (
        args.num_test_tasks == TossingRoomSplitIdentityTasks.model_fields["num_test_tasks"].default
    )


def test_run_method_passes_num_test_tasks_into_tasks(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The composition is only fixed if Tasks knows how many test tasks the harness
    will draw -- so --num-test-tasks has to reach the constructor."""
    captured: dict[str, object] = {}
    build = tossingroomsplitidentity_cli.TossingRoomSplitIdentityTasks
    # Parsed before the patch: add_arguments reads defaults off the real class.
    args = _build_parser().parse_args(["--num-test-tasks", "7"])

    def spy(**kwargs) -> TossingRoomSplitIdentityTasks:
        captured.update(kwargs)
        return build(**kwargs)

    monkeypatch.setattr(tossingroomsplitidentity_cli, "TossingRoomSplitIdentityTasks", spy)
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert captured["num_test_tasks"] == 7


def test_target_range_flags_reach_the_sampled_task(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cause-range flags shape the throw problem here exactly as they do in the causal
    arm, so they have to reach Tasks rather than being parsed and dropped. Pinned to
    degenerate single-value ranges, which makes the effect visible in one task: at the
    reference distance and reference weight the resolved target is exactly
    `reference_force`."""
    args = _build_parser().parse_args([
        "--num-test-tasks",
        "1",
        "--goal-type",
        "trash",
        "--distance-low",
        "2.0",
        "--distance-high",
        "2.0",
        "--weight-low",
        "1.0",
        "--weight-high",
        "1.0",
    ])
    captured: dict[str, object] = {}
    build = tossingroomsplitidentity_cli.TossingRoomSplitIdentityTasks

    def spy(**kwargs) -> TossingRoomSplitIdentityTasks:
        captured.update(kwargs)
        return build(**kwargs)

    monkeypatch.setattr(tossingroomsplitidentity_cli, "TossingRoomSplitIdentityTasks", spy)
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert captured["distance_low"] == 2.0
    assert captured["weight_low"] == 1.0
    # ...and the ranges are what a task actually draws from, not merely something stored.
    tasks = build(**captured)
    task = tasks.sample_test_task()
    assert task.initial_state.get(
        obj=TossingRoomSplitIdentityEnvironment.trash, feature_name="target_force"
    ) == pytest.approx(tasks.reference_force)


def test_run_method_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "8"])
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 8/8 (100%)" in capsys.readouterr().out


def test_run_method_forces_a_single_goal_type(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5", "--goal-type", "recycling"])
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    a = TossingRoomSplitIdentityTasks(
        env=TossingRoomSplitIdentityEnvironment(), seed=99
    ).sample_test_task()
    b = TossingRoomSplitIdentityTasks(
        env=TossingRoomSplitIdentityEnvironment(), seed=99
    ).sample_test_task()
    for item in (
        TossingRoomSplitIdentityEnvironment.recycling,
        TossingRoomSplitIdentityEnvironment.trash,
    ):
        assert a.initial_state.get(obj=item, feature_name="target_force") == b.initial_state.get(
            obj=item, feature_name="target_force"
        )


def test_run_method_respects_a_larger_layout(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args([
        "--num-test-tasks",
        "4",
        "--num-rooms",
        "9",
        "--trash-bin-room",
        "8",
        "--goal-type",
        "trash",
    ])
    TossingRoomSplitIdentityCli.run_method(
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
    TossingRoomSplitIdentityCli.run_method(
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
    TossingRoomSplitIdentityCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert list(tmp_path.iterdir()) == []
