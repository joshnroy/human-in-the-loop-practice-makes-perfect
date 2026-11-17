"""Opt-in `gather_delta_high`: a wider gather delta range, everything else as is."""

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.skills import GATHER_AND_SWEEP_BOUNDS, SWEEP_BOUNDS
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.environments.racktoserving3d.toy_interface import GATHER, SWEEP3, TOY_SKILL_IDS
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER is not installed"
)


def _provider(*, config):
    return RackServingSkillProvider(env=RackServingToyEnvironment(config=config))


def test_the_declared_bounds_are_unchanged_by_default():
    config = RackServingConfig()
    assert config.gather_delta_high is None
    assert config.gather_and_sweep_bounds() == GATHER_AND_SWEEP_BOUNDS


def test_a_wider_delta_replaces_only_deltas_upper_bound():
    config = RackServingConfig(gather_delta_high=0.2)
    assert config.gather_and_sweep_bounds() == ((0.03, 0.14), (0.005, 0.2), (0.0, 0.12))


@pytest.mark.parametrize("high", [0.005, 0.004, 0.21])
def test_a_delta_range_must_stay_nonempty_and_below_the_gather_start(*, high):
    with pytest.raises(ValidationError):
        RackServingConfig(gather_delta_high=high)


def test_the_sampler_draws_delta_over_the_wider_range_and_sweep_is_untouched():
    provider = _provider(config=RackServingConfig(gather_delta_high=0.2))
    rng = np.random.default_rng(3)
    ground = provider.ground(name=GATHER)
    samples = np.array([provider.sample_params(ground_skill=ground, rng=rng) for _ in range(400)])
    assert samples[:, 1].min() >= 0.005 and samples[:, 1].max() <= 0.2
    assert samples[:, 1].max() > 0.18
    sweep = provider.ground(name=SWEEP3)
    depths = np.array([provider.sample_params(ground_skill=sweep, rng=rng)[0] for _ in range(200)])
    assert depths.min() >= SWEEP_BOUNDS[0][0] and depths.max() <= SWEEP_BOUNDS[0][1]


def test_an_action_inside_the_wider_range_encodes_only_when_it_is_configured():
    params = np.array([0.07, 0.15])
    wide = _provider(config=RackServingConfig(gather_delta_high=0.2))
    ground = wide.ground(name=GATHER)
    action = wide.compute_action(ground_skill=ground, params=params, state=State(data={}))
    np.testing.assert_array_equal(action, [TOY_SKILL_IDS[GATHER], 0, 0.07, 0.15, 0.0])
    with pytest.raises(ValueError, match="outside the declared bounds"):
        _provider(config=RackServingConfig()).compute_action(
            ground_skill=ground, params=params, state=State(data={})
        )


def test_the_controller_checks_its_own_configured_bounds():
    controller = RackControllers.__new__(RackControllers)
    controller.gather_and_sweep_bounds = RackServingConfig(
        gather_delta_high=0.2
    ).gather_and_sweep_bounds()
    with pytest.raises(SkillFailed, match=r"delta \[0.005,0.2\]"):
        controller.gather_and_sweep(s=0.07, delta=0.21, d=0.1)


@needs_kinder
def test_the_environments_controllers_take_the_configured_bounds():
    env = RackServingEnvironment(config=RackServingConfig(gather_delta_high=0.2))
    try:
        env.reset_to_seed(seed=0)
        assert env.rack_controllers().gather_and_sweep_bounds[1] == (0.005, 0.2)
    finally:
        env.close()
