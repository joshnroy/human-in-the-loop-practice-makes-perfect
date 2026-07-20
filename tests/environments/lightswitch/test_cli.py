import argparse

import pytest

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global --seed/--num-test-tasks (added by
    _add_global_arguments there, not by LightSwitchCli.add_arguments) plus this
    domain's own flags, so LightSwitchCli.run can be exercised in isolation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
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
    original_grid_size = LightSwitchEnvironment.grid_size
    try:
        args = _build_parser().parse_args(["--num-test-tasks", "4", "--grid-size", "3"])
        assert LightSwitchCli.run(args=args) == (4, 4)
        assert LightSwitchEnvironment.grid_size == 3
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size


def test_run_rejects_an_unknown_policy_choice() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--policy", "not-a-real-policy"])
