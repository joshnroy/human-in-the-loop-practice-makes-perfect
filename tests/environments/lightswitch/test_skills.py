import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills


def test_move_robot_declares_two_cell_params_and_no_continuous_params() -> None:
    skill = LightSwitchSkills.MOVE_ROBOT
    assert skill.name == "MoveRobot"
    assert skill.types == (
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
    )
    assert skill.param_dim == 0


def test_turn_on_light_declares_one_continuous_param() -> None:
    skill = LightSwitchSkills.TURN_ON_LIGHT
    assert skill.name == "TurnOnLight"
    assert skill.types == (
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.light_type,
    )
    assert skill.param_dim == 1


def test_turn_off_light_declares_one_continuous_param() -> None:
    skill = LightSwitchSkills.TURN_OFF_LIGHT
    assert skill.name == "TurnOffLight"
    assert skill.param_dim == 1


def test_jump_to_light_declares_all_three_cells_and_the_light() -> None:
    skill = LightSwitchSkills.JUMP_TO_LIGHT
    assert skill.name == "JumpToLight"
    assert skill.types == (
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.light_type,
    )


def test_sample_params_returns_empty_array_for_zero_dim_skill() -> None:
    ground_skill = GroundSkill(skill=LightSwitchSkills.MOVE_ROBOT, objects=())
    params = LightSwitchSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    assert params.shape == (0,)


def test_sample_params_returns_values_within_unit_interval() -> None:
    ground_skill = GroundSkill(skill=LightSwitchSkills.TURN_ON_LIGHT, objects=())
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = LightSwitchSkills.sample_params(ground_skill=ground_skill, rng=rng)
        assert params.shape == (1,)
        assert -1.0 <= params[0] <= 1.0


def test_compute_action_for_move_robot_ignores_params() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT,
        objects=(LightSwitchEnvironment.robot, cells[0], cells[3]),
    )
    action = LightSwitchSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    target_x = state.get(obj=cells[3], feature_name="x")
    assert action.tolist() == [target_x - robot_x, 0.0]


def test_compute_action_for_turn_on_light_uses_the_sampled_dlight() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(
            LightSwitchEnvironment.robot,
            LightSwitchEnvironment.get_cells()[-1],
            LightSwitchEnvironment.light,
        ),
    )
    action = LightSwitchSkills.compute_action(
        ground_skill=ground_skill, params=np.array([0.42]), state=state
    )
    assert action.tolist() == [0.0, 0.42]


def test_compute_action_for_turn_off_light_uses_the_same_toggle_logic() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.TURN_OFF_LIGHT,
        objects=(
            LightSwitchEnvironment.robot,
            LightSwitchEnvironment.get_cells()[-1],
            LightSwitchEnvironment.light,
        ),
    )
    action = LightSwitchSkills.compute_action(
        ground_skill=ground_skill, params=np.array([-0.3]), state=state
    )
    assert action.tolist() == [0.0, -0.3]


def test_compute_action_for_jump_to_light_is_always_a_no_op() -> None:
    """The "impossible" skill: predicators' actual policy ignores state and params
    entirely and always returns a no-op -- confirmed here regardless of what params
    get sampled, matching test_integration.py's raw-action-level version of the same
    claim."""
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.JUMP_TO_LIGHT,
        objects=(
            LightSwitchEnvironment.robot,
            cells[0],
            cells[1],
            cells[-1],
            LightSwitchEnvironment.light,
        ),
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        params = LightSwitchSkills.sample_params(ground_skill=ground_skill, rng=rng)
        action = LightSwitchSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        assert action.tolist() == [0.0, 0.0]


def test_move_robot_end_to_end_reaches_the_target_cell() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT,
        objects=(LightSwitchEnvironment.robot, cells[0], cells[5]),
    )
    state = LightSwitchEnvironment.get_current_state()
    params = LightSwitchSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    action = LightSwitchSkills.compute_action(ground_skill=ground_skill, params=params, state=state)
    next_state = LightSwitchEnvironment.take_action(action=action)

    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == state.get(
        obj=cells[5], feature_name="x"
    )


def test_compute_action_dispatches_by_value_not_identity() -> None:
    """compute_action must recognize a Skill by field values, not by being the exact
    LightSwitchSkills.MOVE_ROBOT object -- a Method could easily reconstruct an
    equal-content Skill independently (e.g. after a round trip through a planner),
    and frozen pydantic models are value-equal/hashable for exactly this reason."""
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    cells = LightSwitchEnvironment.get_cells()
    reconstructed_skill = Skill(
        name="MoveRobot",
        types=(
            LightSwitchEnvironment.robot_type,
            LightSwitchEnvironment.cell_type,
            LightSwitchEnvironment.cell_type,
        ),
        param_dim=0,
    )
    assert reconstructed_skill is not LightSwitchSkills.MOVE_ROBOT
    ground_skill = GroundSkill(
        skill=reconstructed_skill, objects=(LightSwitchEnvironment.robot, cells[0], cells[3])
    )
    action = LightSwitchSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    target_x = state.get(obj=cells[3], feature_name="x")
    assert action.tolist() == [target_x - robot_x, 0.0]


def test_compute_action_rejects_a_skill_outside_the_known_set() -> None:
    unknown_skill = Skill(name="Unknown", types=(), param_dim=0)
    ground_skill = GroundSkill(skill=unknown_skill, objects=())
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    with pytest.raises(ValueError, match="Unknown skill"):
        LightSwitchSkills.compute_action(ground_skill=ground_skill, params=np.zeros(0), state=state)


def test_turn_on_light_end_to_end_changes_the_light_level_when_co_located() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    LightSwitchEnvironment.set_state(state=state)

    ground_skill = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(
            LightSwitchEnvironment.robot,
            LightSwitchEnvironment.get_cells()[-1],
            LightSwitchEnvironment.light,
        ),
    )
    action = LightSwitchSkills.compute_action(
        ground_skill=ground_skill, params=np.array([0.5]), state=state
    )
    next_state = LightSwitchEnvironment.take_action(action=action)
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 0.5
