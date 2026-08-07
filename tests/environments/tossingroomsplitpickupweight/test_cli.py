import argparse
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossingroomsplitpickupweight import (
    cli as tossingroomsplitpickupweight_cli,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.cli import TossingRoomSplitPickupWeightCli
from hitl_pmp.environments.tossingroomsplitpickupweight.environment import (
    TossingRoomSplitPickupWeightEnvironment,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.tasks import (
    TossingRoomSplitPickupWeightTasks,
)
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags plus this domain's own, so
    TossingRoomSplitPickupWeightCli can be exercised in isolation (mirrors Light
    Switch's test_cli)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    TossingRoomSplitPickupWeightCli.add_arguments(parser=parser)
    return parser


def test_there_is_no_button_room_flag() -> None:
    """Each bin's button sits in that bin's own room, so --trash-bin-room and
    --recycling-bin-room place the buttons too. A separate --button-room would be a
    second knob that has to agree with them."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--button-room", "4"])


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    fields = TossingRoomSplitPickupWeightEnvironment.model_fields
    assert args.num_rooms == fields["num_rooms"].default
    assert args.start_room == fields["start_room"].default
    assert args.recycling_bin_room == fields["recycling_bin_room"].default
    assert args.trash_bin_room == fields["trash_bin_room"].default
    assert args.blocked_right_from == fields["blocked_right_from"].default
    assert args.throw_tolerance == fields["throw_tolerance"].default
    # The (unobserved) required-force relation, in reference form.
    assert args.reference_force == fields["reference_force"].default
    assert args.reference_distance == fields["reference_distance"].default
    assert args.reference_weight == fields["reference_weight"].default
    assert args.distance_coefficient == fields["distance_coefficient"].default
    assert args.weight_coefficient == fields["weight_coefficient"].default
    assert args.throw_distance == fields["throw_distance"].default
    assert args.canonical_item_weight == fields["canonical_item_weight"].default
    # ...and the range every pickup draws from. There are no per-task cause ranges: the
    # weight range moved onto the Environment with the draw, and the distance is fixed.
    assert args.pickup_weight_low == fields["pickup_weight_low"].default
    assert args.pickup_weight_high == fields["pickup_weight_high"].default
    # Sized from the run's step budget rather than defaulted, so the flag itself is None.
    assert args.weight_schedule_length is None
    assert args.goal_type is None


def test_there_is_no_target_force_flag() -> None:
    """The force a throw needs is not configurable per item and not in the state: it is
    derived from the bin's throw_distance and the item's weight by a relation only the
    environment knows."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--canonical-target-force", "0.5"])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--target-low", "0.5"])


def test_num_test_tasks_field_default_matches_the_global_flag_default() -> None:
    """Two independent literals (hitl_pmp/cli.py's --num-test-tasks and this domain's
    Tasks field) that must agree, since the field is what the fixed test-set composition
    is divided out of."""
    parser = argparse.ArgumentParser()
    Cli.add_global_arguments(parser=parser)
    args = parser.parse_args(["--env", "tossingroomsplitpickupweight", "--method", "skill-oracle"])
    assert (
        args.num_test_tasks
        == TossingRoomSplitPickupWeightTasks.model_fields["num_test_tasks"].default
    )


def test_run_method_passes_num_test_tasks_into_tasks(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """The composition is only fixed if Tasks knows how many test tasks the harness
    will draw -- so --num-test-tasks has to reach the constructor."""
    captured: dict[str, object] = {}
    build = tossingroomsplitpickupweight_cli.TossingRoomSplitPickupWeightTasks
    # Parsed before the patch: add_arguments reads defaults off the real class.
    args = _build_parser().parse_args(["--num-test-tasks", "7"])

    def spy(**kwargs) -> TossingRoomSplitPickupWeightTasks:
        captured.update(kwargs)
        return build(**kwargs)

    monkeypatch.setattr(tossingroomsplitpickupweight_cli, "TossingRoomSplitPickupWeightTasks", spy)
    TossingRoomSplitPickupWeightCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert captured["num_test_tasks"] == 7


def test_run_method_hands_the_loop_a_separate_evaluation_problem(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This domain's composition root builds two independent triples, so evaluation's
    per-episode `reset_to_task` lands on an environment practice never sees. The two
    must be genuinely distinct objects all the way down -- a shared Environment or a
    shared Tasks would put the write straight back."""
    captured: dict[str, object] = {}
    real_run = tossingroomsplitpickupweight_cli.MethodRunner.run

    def spy(**kwargs):
        captured.update(kwargs)
        return real_run(**kwargs)

    monkeypatch.setattr(tossingroomsplitpickupweight_cli.MethodRunner, "run", spy)
    TossingRoomSplitPickupWeightCli.run_method(
        args=_build_parser().parse_args(["--num-test-tasks", "4"]),
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    practice = captured["problem"]
    evaluation = captured["evaluation_problem"]
    assert evaluation is not None
    assert evaluation is not practice
    assert evaluation.env is not practice.env
    assert evaluation.tasks is not practice.tasks


def test_the_two_problems_are_configured_identically(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Separate instances, same configuration -- including the seed. That is what makes
    the split a no-op on results: the evaluation Tasks' test stream is derived from the
    same seed, so it yields exactly the test tasks the practice Tasks would have."""
    captured: dict[str, object] = {}
    real_run = tossingroomsplitpickupweight_cli.MethodRunner.run

    def spy(**kwargs):
        captured.update(kwargs)
        return real_run(**kwargs)

    monkeypatch.setattr(tossingroomsplitpickupweight_cli.MethodRunner, "run", spy)
    TossingRoomSplitPickupWeightCli.run_method(
        args=_build_parser().parse_args(["--num-test-tasks", "4", "--seed", "3"]),
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    practice = captured["problem"]
    evaluation = captured["evaluation_problem"]
    # Every configured field of both models, compared by name rather than by a
    # whole-model dump (which cannot serialize the numpy-backed current_state).
    # Enumerated rather than spot-checked so a field added later is covered too --
    # test_env_seed_offset in particular is what derives the test stream.
    for field in TossingRoomSplitPickupWeightEnvironment.model_fields:
        if field == "current_state":
            continue
        assert getattr(evaluation.env, field) == getattr(practice.env, field), field
    for field in TossingRoomSplitPickupWeightTasks.model_fields:
        if field == "env":  # the two Environments are distinct objects by design
            continue
        assert getattr(evaluation.tasks, field) == getattr(practice.tasks, field), field
    assert evaluation.tasks.seed == 3
    assert evaluation.tasks.num_test_tasks == 4


def test_the_two_problems_draw_the_same_test_tasks() -> None:
    """The property that matching configuration is only a proxy for, asserted
    directly: two independently-built triples yield the SAME test set, in the same
    order. That is what makes moving the draw from the practice Tasks to the
    evaluation Tasks a no-op on results rather than a silent change of which tasks
    are measured."""
    args = _build_parser().parse_args(["--num-test-tasks", "6", "--seed", "5"])
    practice = TossingRoomSplitPickupWeightCli.build_problem(args=args)
    evaluation = TossingRoomSplitPickupWeightCli.build_problem(args=args)
    # Whole sequences, not one draw each: the goal-family schedule is a permutation
    # built once from the test stream, so a per-instance divergence would only show
    # up part-way through.
    left = [practice.tasks.sample_test_task() for _ in range(6)]
    right = [evaluation.tasks.sample_test_task() for _ in range(6)]
    for first, second in zip(left, right, strict=True):
        assert first.goal.describe() == second.goal.describe()
        # Compared feature-vector by feature-vector: State wraps numpy arrays, so
        # `==` on the model is ambiguous rather than false.
        assert set(first.initial_state.data) == set(second.initial_state.data)
        for obj, features in first.initial_state.data.items():
            assert np.array_equal(features, second.initial_state.data[obj]), obj.name


def test_run_method_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "8"])
    TossingRoomSplitPickupWeightCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 8/8 (100%)" in capsys.readouterr().out


def test_run_method_forces_a_single_goal_type(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5", "--goal-type", "recycling"])
    TossingRoomSplitPickupWeightCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3", "--seed", "99"])
    TossingRoomSplitPickupWeightCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    a = TossingRoomSplitPickupWeightTasks(
        env=TossingRoomSplitPickupWeightEnvironment(), seed=99
    ).sample_test_task()
    b = TossingRoomSplitPickupWeightTasks(
        env=TossingRoomSplitPickupWeightEnvironment(), seed=99
    ).sample_test_task()
    assert a.initial_state.get(
        obj=TossingRoomSplitPickupWeightEnvironment.recycling, feature_name="weight"
    ) == b.initial_state.get(
        obj=TossingRoomSplitPickupWeightEnvironment.recycling, feature_name="weight"
    )
    assert a.initial_state.get(
        obj=TossingRoomSplitPickupWeightEnvironment.recycling_bin, feature_name="throw_distance"
    ) == b.initial_state.get(
        obj=TossingRoomSplitPickupWeightEnvironment.recycling_bin, feature_name="throw_distance"
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
    TossingRoomSplitPickupWeightCli.run_method(
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
    TossingRoomSplitPickupWeightCli.run_method(
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
    TossingRoomSplitPickupWeightCli.run_method(
        args=args,
        method_factory=lambda ctx: SkillOracleMethod(env=ctx.env, oracle=ctx.oracle),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert list(tmp_path.iterdir()) == []


def test_two_way_ledge_defaults_off_and_is_a_store_true_flag() -> None:
    """Opt-in, so PR #122's ten banked seeds are reproduced by the default command
    line."""
    assert _build_parser().parse_args([]).two_way_ledge is False
    assert _build_parser().parse_args(["--two-way-ledge"]).two_way_ledge is True
    assert (
        _build_parser().parse_args([]).two_way_ledge
        == TossingRoomSplitPickupWeightEnvironment.model_fields["two_way_ledge"].default
    )


def test_two_way_ledge_reaches_the_built_environment() -> None:
    """A flag that parses but never reaches the Environment would run the one-way world
    while `config_snapshot.json` recorded the two-way one -- the exact failure that makes
    a banked result unattributable."""
    parser = _build_parser()
    built = TossingRoomSplitPickupWeightCli.build_problem(args=parser.parse_args([]))
    assert built.env.two_way_ledge is False
    two_way = TossingRoomSplitPickupWeightCli.build_problem(
        args=parser.parse_args(["--two-way-ledge"])
    )
    assert two_way.env.two_way_ledge is True
    # ...and the world it built really is two-way, not merely labelled so.
    assert two_way.rooms_to_walk_between(from_room=1, to_room=3) == 2
