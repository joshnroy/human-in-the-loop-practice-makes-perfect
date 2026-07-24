import argparse
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.cli import ENVIRONMENTS, Cli
from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Policy, Rollout, SetupCommand
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.ballring.cli import BallRingCli
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.tasks import BallRingTasks

E = BallRingEnvironment


class _NoOpMethod(Method):
    """A stand-in Method for exercising the composition root end to end. No
    Ball-Ring Method exists yet (PR 2 ships skills; the method CLIs are still
    hardcoded to Light Switch), so this only needs get_task_policy to be real --
    with num_cycles=0 the other hooks are never reached."""

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        return lambda state: LabeledAction(action=np.array([0.0, 0.0, 0.0, 0.5, 0.5]), label="noop")

    def reset_environment(self, *, start_state: State) -> bool:
        raise NotImplementedError

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    BallRingCli.add_arguments(parser=parser)
    return parser


def test_ballring_is_registered_in_the_global_environments() -> None:
    assert ENVIRONMENTS["ballring"] is BallRingCli


def test_global_cli_resolves_env_ballring_and_registers_its_flags() -> None:
    args = Cli.parse_args(
        argv=["--env", "ballring", "--method", "skill-oracle", "--num-tables", "4"]
    )
    assert args.env == "ballring"
    assert args.num_tables == 4


def test_add_arguments_defaults_match_live_class_values() -> None:
    args = _build_parser().parse_args([])
    assert args.num_tables == E.model_fields["num_tables"].default
    assert args.num_sticky_tables == E.model_fields["num_sticky_tables"].default
    assert args.pick_success_prob == E.model_fields["pick_success_prob"].default
    assert args.place_sticky_fall_prob == E.model_fields["place_sticky_fall_prob"].default
    assert args.place_ball_fall_prob == E.model_fields["place_ball_fall_prob"].default
    assert args.place_smooth_fall_prob == E.model_fields["place_smooth_fall_prob"].default
    assert args.noise_seed == E.model_fields["noise_seed"].default
    assert args.test_env_seed_offset == BallRingTasks.model_fields["test_env_seed_offset"].default


def test_run_method_drives_the_composition_root(*, capsys: Any) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "3"])
    BallRingCli.run_method(
        args=args,
        method_factory=lambda env: _NoOpMethod(env=env),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    # The no-op policy never solves, but the run completes and reports a rate.
    assert "success rate: 0/3" in capsys.readouterr().out


def test_run_method_applies_seed_deterministically() -> None:
    for _ in range(2):
        args = _build_parser().parse_args(["--num-test-tasks", "2", "--seed", "99"])
        BallRingCli.run_method(
            args=args,
            method_factory=lambda env: _NoOpMethod(env=env),
            num_cycles=0,
            max_steps_per_interaction=0,
        )
    tasks = BallRingTasks(env=E(), seed=99)
    first = tasks.sample_test_task().initial_state.get(obj=E.ball, feature_name="x")
    tasks.set_seed(seed=99)
    assert tasks.sample_test_task().initial_state.get(obj=E.ball, feature_name="x") == first


def test_run_method_with_output_dir_writes_stats_but_no_video(*, tmp_path: Path) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "2", "--output-dir", str(tmp_path)])
    BallRingCli.run_method(
        args=args,
        method_factory=lambda env: _NoOpMethod(env=env),
        num_cycles=0,
        max_steps_per_interaction=0,
    )
    assert (tmp_path / "stats.json").exists()
    # No Ball-Ring renderer yet, so no episode.mp4 is written (see BallRingCli).
    assert not (tmp_path / "episode.mp4").exists()
