"""The toss's parameter bounds are a parameter set the environment carries.

The simulator's set is exactly the constants every sim experiment used, so nothing a
sim run draws changes; a real-robot environment carries the real swing's set instead,
and every draw, band and refusal on it follows that set.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.toss_direction import (
    TossDirectionChoice,
    TossDirectionSelector,
    TossStandoffBand,
)
from hitl_pmp.environments.tossing3d.types import TossFeasibilityGeometry, TossParameterSet
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    REAL_TOSS_PARAMETERS,
    SIM_TOSS_PARAMETERS,
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

from .observations import state


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


def test_the_sim_set_is_the_constants_every_sim_run_used() -> None:
    assert SIM_TOSS_PARAMETERS.standoff_bounds == WIDE_TOSS_STANDOFF_BOUNDS
    assert SIM_TOSS_PARAMETERS.speed_bounds == WIDE_TOSS_SPEED_BOUNDS
    assert SIM_TOSS_PARAMETERS.release_ms_bounds == WIDE_TOSS_RELEASE_MS_BOUNDS


def test_the_real_set_is_the_real_swings() -> None:
    # One 1.20 m standoff for both bins (Josh, 2026-11-05), +-5 mm because a set's bounds
    # must be an interval; the robot side's own bounds are 0.9-1.45 m.
    assert REAL_TOSS_PARAMETERS.standoff_bounds == (1.295, 1.305)
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
    assert REAL_TOSS_PARAMETERS.violation(params=(1.3, 120.0, 600.0)) is None
    assert "speed" in str(REAL_TOSS_PARAMETERS.violation(params=(1.3, 150.0, 600.0)))
    assert "standoff" in str(REAL_TOSS_PARAMETERS.violation(params=(2.0, 120.0, 600.0)))
    assert "standoff" in str(REAL_TOSS_PARAMETERS.violation(params=(1.0, 120.0, 600.0)))
    assert "release" in str(REAL_TOSS_PARAMETERS.violation(params=(1.3, 120.0, 900.0)))


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


def _toss_row(*, env: Tossing3DEnvironment, params: np.ndarray) -> list[float]:
    row = Tossing3DSkillProvider(env=env).hand_selected_feature_transform(
        ground_skill=_toss(env=env), state=state(env=env), params=params
    )
    assert row is not None
    return row


def test_the_real_set_learns_speed_and_release_only() -> None:
    # The real standoff is fixed at 1.20 m; its +-5 mm band is a placeholder interval,
    # and min/max normalisation would stretch it to the span of the other two inputs.
    assert REAL_TOSS_PARAMETERS.classifier_inputs == ("speed", "release_ms")


def test_the_sim_set_learns_all_three() -> None:
    assert SIM_TOSS_PARAMETERS.classifier_inputs == ("standoff", "speed", "release_ms")


def test_a_real_classifier_row_ignores_the_standoff() -> None:
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    rows = [
        _toss_row(env=env, params=np.array([standoff, 114.72, 729.37]))
        for standoff in (1.195, 1.200, 1.202, 1.205)
    ]
    assert rows == [[1.0, 114.72, 729.37]] * 4


def test_two_real_candidates_differing_only_in_standoff_score_identically() -> None:
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    sampler = LearnedSkillSampler(
        skill_name="MoveToTossLocationAndToss", param_dim=3, seed=0, max_train_iters=200
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = WideLongRangeTossProposal.sample(rng=rng, parameters=REAL_TOSS_PARAMETERS)
        sampler.observe(sampler_input=_toss_row(env=env, params=params), success=params[1] < 120)
    sampler.fit()
    assert sampler.is_fitted
    rows = [_toss_row(env=env, params=np.array([s, 114.72, 729.37])) for s in (1.295, 1.305)]
    low, high = sampler.score_inputs(sampler_inputs=rows)
    assert low == high


def test_the_executed_real_action_still_carries_the_drawn_standoff(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    def select_for_state(*, state, standoff: float) -> TossDirectionChoice:
        del state, standoff
        return TossDirectionChoice(
            direction_deg=0, rotation=0.0, stand_xy=(0.0, 0.0), clearance_m=1.0
        )

    monkeypatch.setattr(TossDirectionSelector, "select_for_state", select_for_state)
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    candidates = [np.array([1.196, 114.72, 729.37]), np.array([1.204, 114.72, 729.37])]
    rows = [_toss_row(env=env, params=candidate) for candidate in candidates]
    assert rows[0] == rows[1]
    sampler = LearnedSkillSampler(skill_name="MoveToTossLocationAndToss", param_dim=3, seed=0)
    chosen = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False).params
    assert any(chosen is candidate for candidate in candidates)
    action = Tossing3DSkillProvider(env=env).compute_action(
        ground_skill=_toss(env=env), params=chosen, state=state(env=env)
    )
    assert action[1] == pytest.approx(chosen[0])


def test_the_sim_classifier_row_is_unchanged_byte_for_byte() -> None:
    env = Tossing3DEnvironment()
    rng = np.random.default_rng(11)
    for _ in range(50):
        params = WideLongRangeTossProposal.sample(rng=rng)
        row = _toss_row(env=env, params=params)
        # What `Tossing3DToss.feature_row` built before classifier inputs were declared.
        expected = [1.0, *(float(value) for value in params)]
        assert [type(value) for value in row] == [float] * 4
        assert np.array(row).tobytes() == np.array(expected).tobytes()


def test_classifier_inputs_must_be_a_nonempty_ordered_subset() -> None:
    for bad in ((), ("speed", "speed"), ("release_ms", "speed"), ("yaw",)):
        with pytest.raises(ValueError):
            TossParameterSet(
                standoff_bounds=(1.0, 1.4),
                speed_bounds=(100.0, 140.0),
                release_ms_bounds=(500.0, 850.0),
                classifier_inputs=bad,
            )
