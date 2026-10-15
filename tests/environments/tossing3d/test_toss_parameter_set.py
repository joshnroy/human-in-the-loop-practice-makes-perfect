"""The toss's parameter bounds are a parameter set the environment carries.

The simulator's set is exactly the constants every sim experiment used, so nothing a
sim run draws changes; a real-robot environment carries the real swing's set instead,
and every draw, band and refusal on it follows that set.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.toss_direction import TossStandoffBand
from hitl_pmp.environments.tossing3d.types import TossFeasibilityGeometry, TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    REAL_TOSS_PARAMETERS,
    SIM_TOSS_PARAMETERS,
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.robot),
    )


def test_the_sim_set_is_the_constants_every_sim_run_used() -> None:
    assert SIM_TOSS_PARAMETERS.standoff_bounds == WIDE_TOSS_STANDOFF_BOUNDS
    assert SIM_TOSS_PARAMETERS.speed_bounds == WIDE_TOSS_SPEED_BOUNDS
    assert SIM_TOSS_PARAMETERS.release_ms_bounds == WIDE_TOSS_RELEASE_MS_BOUNDS


def test_the_real_set_is_the_real_swings() -> None:
    assert REAL_TOSS_PARAMETERS.standoff_bounds == (0.9, 1.45)
    assert REAL_TOSS_PARAMETERS.speed_bounds == (100.0, 140.0)
    assert REAL_TOSS_PARAMETERS.release_ms_bounds == (500.0, 850.0)


def test_an_environment_carries_the_sim_set_unless_told_otherwise() -> None:
    assert Tossing3DToss.parameters_of(env=Tossing3DEnvironment()) == SIM_TOSS_PARAMETERS
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    assert Tossing3DToss.parameters_of(env=env) == REAL_TOSS_PARAMETERS


def test_the_sim_draw_is_unchanged_draw_for_draw() -> None:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    ours = np.random.default_rng(3)
    reference = np.random.default_rng(3)
    for _ in range(50):
        np.testing.assert_array_equal(
            provider.sample_params(ground_skill=_toss(env=env), rng=ours),
            WideLongRangeTossProposal.sample(rng=reference),
        )


def test_a_real_environment_draws_only_inside_the_real_set() -> None:
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    provider = Tossing3DSkillProvider(env=env)
    rng = np.random.default_rng(0)
    draws = np.array([
        provider.sample_params(ground_skill=_toss(env=env), rng=rng) for _ in range(500)
    ])
    for column, (low, high) in enumerate((
        REAL_TOSS_PARAMETERS.standoff_bounds,
        REAL_TOSS_PARAMETERS.speed_bounds,
        REAL_TOSS_PARAMETERS.release_ms_bounds,
    )):
        assert draws[:, column].min() >= low
        assert draws[:, column].max() <= high
        # And it spans the set rather than a corner of it.
        assert draws[:, column].max() - draws[:, column].min() > 0.9 * (high - low)


def test_violation_names_the_parameter_and_the_bound() -> None:
    assert REAL_TOSS_PARAMETERS.violation(params=(1.0, 120.0, 600.0)) is None
    assert "speed" in str(REAL_TOSS_PARAMETERS.violation(params=(1.0, 150.0, 600.0)))
    assert "standoff" in str(REAL_TOSS_PARAMETERS.violation(params=(2.0, 120.0, 600.0)))
    assert "release" in str(REAL_TOSS_PARAMETERS.violation(params=(1.0, 120.0, 900.0)))


def test_a_parameter_set_refuses_an_empty_or_inverted_range() -> None:
    with pytest.raises(ValueError):
        TossParameterSet(
            standoff_bounds=(1.0, 0.9), speed_bounds=(100, 140), release_ms_bounds=(500, 850)
        )


def test_the_robot_side_band_follows_the_parameter_set() -> None:
    TossStandoffBand.clear_cache()
    geometry = TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.55, 0.5),
        bin_pose=(2.0, 0.0, 0.0),
        obstacles=(),
        sampling_x_bounds=(-5.0, 5.0),
        sampling_y_bounds=(-5.0, 5.0),
    )
    assert TossStandoffBand.robot_side_bounds(geometry=geometry) == WIDE_TOSS_STANDOFF_BOUNDS
    assert (
        TossStandoffBand.robot_side_bounds(geometry=geometry, parameters=REAL_TOSS_PARAMETERS)
        == REAL_TOSS_PARAMETERS.standoff_bounds
    )
