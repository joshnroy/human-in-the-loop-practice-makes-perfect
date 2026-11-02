"""Provider offerings and encoding are pure and deployment-safe."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.skill_provider import (
    RackServingActions,
    RackServingSkillProvider,
)
from hitl_pmp.environments.racktoserving3d.skills import SWEEP_SPEED_M_S
from hitl_pmp.environments.racktoserving3d.skills import RackServing3DSkills as Skills


def _provider(*, mode="training", offer=True):
    return RackServingSkillProvider(
        env=RackServingEnvironment(
            config=RackServingConfig(mode=mode, human_interruption_cost=8.0)
        ),
        offer_human_reset=offer,
    )


def _sweep():
    return GroundSkill(
        skill=Skills.SWEEP_RACK,
        objects=(Objects.robot, Objects.scraper, Objects.rack, Objects.tray, *Objects.food),
    )


def test_sweep_samples_lateral_offset_and_plow_angle_in_bounds():
    provider = _provider()
    rng = np.random.default_rng(10)
    samples = np.array([provider.sample_params(ground_skill=_sweep(), rng=rng) for _ in range(200)])
    assert samples.shape == (200, 2)
    assert np.all(np.abs(samples[:, 0]) <= 0.1875)
    assert np.all(np.abs(samples[:, 1]) <= 40)
    assert samples[:, 0].min() < -0.1 and samples[:, 0].max() > 0.1
    assert samples[:, 1].min() < -20 and samples[:, 1].max() > 20
    action = provider.compute_action(
        ground_skill=_sweep(), params=np.array([-0.04, 15.0]), state=State(data={})
    )
    np.testing.assert_array_equal(action, [1, 0, -0.04, 15.0])


def test_sweep_bounds_are_lateral_offset_then_plow_angle_with_fixed_speed():
    assert Skills.SWEEP_BOUNDS == ((-0.1875, 0.1875), (-40.0, 40.0))
    assert SWEEP_SPEED_M_S == 0.06
    assert Skills.SWEEP_RACK.param_dim == 2


def test_nonparameterized_actions_return_empty_parameters_and_correct_food_index():
    provider = _provider()
    rng = np.random.default_rng(0)
    ground = GroundSkill(
        skill=Skills.PICK_FOOD, objects=(Objects.robot, Objects.food[3], Objects.rack, Objects.tray)
    )
    params = provider.sample_params(ground_skill=ground, rng=rng)
    assert params.shape == (0,)
    np.testing.assert_array_equal(
        provider.compute_action(ground_skill=ground, params=params, state=State(data={})),
        [3, 3, 0, 0],
    )
    wait = GroundSkill(skill=Skills.WAIT, objects=())
    assert provider.sample_params(ground_skill=wait, rng=rng).shape == (0,)
    np.testing.assert_array_equal(
        RackServingActions.encode(ground_skill=wait, params=np.zeros(0)), [6, 0, 0, 0]
    )


@pytest.mark.parametrize(
    "params",
    [
        [],
        [0],
        [0, 0, 0.06],
        [0.19, 0],
        [-0.19, 0],
        [0, 41],
        [0, -41],
        [0, np.nan],
        [np.inf, 0],
        [[0, 0]],
    ],
)
def test_invalid_sweep_parameters_rejected(*, params):
    with pytest.raises(ValueError):
        RackServingActions.encode(ground_skill=_sweep(), params=np.array(params))


@pytest.mark.parametrize("params", [[-0.1875, -40], [0.1875, 40]])
def test_sweep_parameter_boundaries_are_inclusive(*, params):
    assert RackServingActions.encode(ground_skill=_sweep(), params=np.array(params)).shape == (4,)


def test_human_skills_are_priced_but_never_encoded_as_robot_actions():
    provider = _provider()
    human = provider.human_cube_bin_reset_skills()
    assert len(human) == 2
    for ground in human:
        assert ground.evaluate_practice_cost() == 8.0
        assert provider.movables_reset_destination(ground_skill=ground) == ground.skill.name
        assert ground.skill not in provider.skills()
        with pytest.raises(PermissionError, match="not robot actions"):
            provider.compute_action(ground_skill=ground, params=np.zeros(0), state=State(data={}))


def test_deployment_cannot_offer_or_decode_training_human_groundings():
    deployment = _provider(mode="deployment")
    assert deployment.human_cube_bin_reset_skills() == ()
    assert deployment.human_cube_bin_reset_skill() is None
    assert deployment.movables_reset_skills() == ()
    for ground in _provider().human_cube_bin_reset_skills():
        with pytest.raises(PermissionError, match="training-only"):
            deployment.movables_reset_destination(ground_skill=ground)
        with pytest.raises(PermissionError, match="not robot actions"):
            deployment.compute_action(ground_skill=ground, params=np.zeros(0), state=State(data={}))


def test_no_human_flag_removes_only_human_offerings():
    disabled = _provider(offer=False)
    assert disabled.human_cube_bin_reset_skills() == ()
    assert disabled.skills() == Skills.robot_skills()
