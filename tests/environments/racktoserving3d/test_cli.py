"""CLI composition tests: no simulator creation or experiment launch."""

import argparse
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.problem import EPISODE_HORIZON
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.method_runner import MethodRunner
from tests.practice_cost_fixtures import CostFixtures


def _parse(*, argv=()):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    RackServingCli.add_arguments(parser=parser)
    return parser.parse_args(list(argv))


def _initial_state(*, seed=0):
    data = {obj: np.zeros(obj.type.dim) for obj in Objects.all_objects()}
    data[Objects.scene] = np.array([seed, 0, 0, 0], dtype=float)
    return State(data=data)


def test_cli_defaults_match_config_with_step_protocol_registered():
    args = _parse()
    defaults = RackServingConfig()
    for field in RackServingConfig.model_fields:
        if field != "mode":
            assert getattr(args, field) == getattr(defaults, field)
    assert args.surface_kind == "wire_solid"
    assert not hasattr(args, "cooling_seconds")
    assert args.human_interruption_cost == 5
    assert args.skill_step_limit == 12000
    assert not args.scene_bg
    assert args.human_reset
    assert ENVIRONMENTS["racktoserving3d"] is RackServingCli
    assert "racktoserving3d" in STEP_PROTOCOL_ENVIRONMENTS


def test_global_cli_accepts_variant_and_physical_options():
    args = Cli.parse_args(
        argv=[
            "--env",
            "racktoserving3d",
            "--method",
            "random-skills",
            "--surface-kind",
            "concept",
            "--gap",
            "0.02",
            "--rack-height",
            "0.03",
            "--rack-mass",
            "0.8",
            "--dumpling-mass",
            "0.03",
            "--dumpling-friction",
            "0.3",
            "0.005",
            "0.0001",
            "--no-human-reset",
            *CostFixtures.step_argv(env="racktoserving3d"),
        ]
    )
    problem = RackServingCli.build_problem(args=args, mode="training")
    assert problem.env.config.surface_kind == "concept"
    assert problem.env.config.gap == 0.02
    assert problem.env.config.rack_mass == 0.8
    assert problem.env.config.dumpling_mass == 0.03
    assert problem.env.config.dumpling_friction == (0.3, 0.005, 0.0001)
    assert not args.human_reset


@pytest.mark.parametrize("option", ["--human-interruption-cost", "--rack-mass"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_physical_and_cost_config_rejected(*, option, value):
    args = _parse(argv=(option, value))
    with pytest.raises(ValidationError):
        RackServingCli.build_problem(args=args, mode="training")


def test_runner_uses_distinct_training_and_deployment_environments(*, monkeypatch):
    captured = {}
    closed = []

    factory = Mock(return_value="test-method")

    def run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(MethodRunner, "run", run)
    monkeypatch.setattr(RackServingEnvironment, "close", lambda env: closed.append(env))
    RackServingCli.run_method(
        args=_parse(), method_factory=factory, num_cycles=0, max_steps_per_interaction=1
    )
    practice = captured["problem"]
    evaluation = captured["evaluation_problem"]
    context = factory.call_args.args[0]
    assert practice.env is not evaluation.env
    assert practice.env.config.mode == "training"
    assert evaluation.env.config.mode == "deployment"
    assert practice.human is not None
    assert evaluation.human is None
    assert context.skill_provider.env is practice.env
    assert len(context.skill_provider.human_cube_bin_reset_skills()) == 2
    assert RackServingSkillProvider(env=evaluation.env).human_cube_bin_reset_skills() == ()
    assert len(closed) == 2
    assert 1 + 2 * len(Objects.food) <= EPISODE_HORIZON


def test_runner_closes_both_environments_if_method_construction_fails(*, monkeypatch):
    closed_modes = []

    factory = Mock(side_effect=RuntimeError("test factory failure"))

    monkeypatch.setattr(
        RackServingEnvironment, "close", lambda env: closed_modes.append(env.config.mode)
    )
    with pytest.raises(RuntimeError, match="factory failure"):
        RackServingCli.run_method(
            args=_parse(), method_factory=factory, num_cycles=0, max_steps_per_interaction=1
        )
    assert closed_modes == ["deployment", "training"]


def test_deployment_rejects_human_task_reset_before_scene_rebuild(*, monkeypatch):
    problem = RackServingCli.build_problem(args=_parse(), mode="deployment")
    writes = []

    def reset(*, seed):
        writes.append(seed)
        return _initial_state(seed=seed)

    monkeypatch.setattr(RackServingEnvironment, "reset_to_seed", staticmethod(reset))
    with pytest.raises(PermissionError, match="training-only"):
        problem.tasks.sample_reset_start_task()
    assert not writes


def test_in_place_task_does_not_reset_scene_or_clock(*, monkeypatch):
    problem = RackServingCli.build_problem(args=_parse(), mode="training")
    state = _initial_state()
    state.set(obj=Objects.scene, feature_name="elapsed_seconds", feature_val=312.0)
    problem.env.current_state = state

    def reset(*, seed):
        pytest.fail("in-place sampling must not reset the environment")

    monkeypatch.setattr(RackServingEnvironment, "reset_to_seed", staticmethod(reset))
    task = problem.tasks.sample_train_task_in_place()
    assert task.initial_state is state
    assert state.get(obj=Objects.scene, feature_name="elapsed_seconds") == 312.0


def test_set_state_only_restores_episode_initial_states(*, monkeypatch):
    problem = RackServingCli.build_problem(args=_parse(), mode="training")
    reset_seeds = []

    def reset(*, seed):
        reset_seeds.append(seed)
        return _initial_state(seed=seed)

    monkeypatch.setattr(RackServingEnvironment, "reset_to_seed", staticmethod(reset))
    state = _initial_state(seed=11)
    problem.env.set_state(state=state)
    assert reset_seeds == [11]
    state.set(obj=Objects.scene, feature_name="steps_taken", feature_val=1)
    with pytest.raises(ValueError, match="initial"):
        problem.env.set_state(state=state)
    assert reset_seeds == [11]


def test_train_test_and_human_reset_rng_streams_are_independent(*, monkeypatch):
    def reset(*, seed):
        return _initial_state(seed=seed)

    monkeypatch.setattr(RackServingEnvironment, "reset_to_seed", staticmethod(reset))
    first = RackServingCli.build_problem(args=_parse(), mode="training").tasks
    second = RackServingCli.build_problem(args=_parse(), mode="training").tasks
    first.sample_reset_start_task()
    first.sample_test_task()
    train_one = first.sample_train_task().initial_state.get(obj=Objects.scene, feature_name="seed")
    train_two = second.sample_train_task().initial_state.get(obj=Objects.scene, feature_name="seed")
    assert train_one == train_two


@pytest.mark.parametrize("target", ["round_plate", "slate", "lipped_platter", "corelle_plate"])
def test_cli_selects_the_serving_target(*, target):
    args = _parse(argv=("--serving-target", target, "--plate-radius", "0.12"))
    problem = RackServingCli.build_problem(args=args, mode="training")
    assert problem.env.config.serving_target == target
    assert problem.env.config.plate_radius == 0.12


def test_cli_rejects_unknown_serving_target():
    with pytest.raises(SystemExit):
        _parse(argv=("--serving-target", "bowl"))


def test_cli_carries_the_opt_in_plate_tuck_sweep_speed_and_custom_layout(*, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "hitl_pmp.environments.racktoserving3d.cli.RackServingEnvironment",
        lambda **kwargs: seen.append(kwargs["config"]) or Mock(),
    )
    monkeypatch.setattr(
        "hitl_pmp.environments.racktoserving3d.cli.RackServingTasks", lambda **kwargs: Mock()
    )
    monkeypatch.setattr(
        "hitl_pmp.environments.racktoserving3d.cli.RackServingProblem", lambda **kwargs: Mock()
    )
    offsets = [str(v) for v in (-0.08, 0, 0, -0.03, 0, 0, 0, 0, 0, 0.03, 0, 0, 0.08, 0.02, 0.1)]
    args = _parse(
        argv=["--plate-tuck", "0.035", "--final-sweep-speed", "0.1", "--start-layout", "custom"]
        + ["--custom-layout", *offsets]
    )
    RackServingCli.build_problem(args=args, mode="training")
    (config,) = seen
    assert config.plate_tuck == 0.035
    assert config.final_sweep_speed == 0.1
    assert config.custom_layout[4] == (0.08, 0.02, 0.1)
    assert len(config.custom_layout) == 5
