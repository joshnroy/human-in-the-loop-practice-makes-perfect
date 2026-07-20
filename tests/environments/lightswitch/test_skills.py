import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills


def test_move_robot_declares_two_cell_parameters_and_no_continuous_params() -> None:
    skill = LightSwitchSkills.MOVE_ROBOT
    assert skill.name == "MoveRobot"
    assert [p.type for p in skill.parameters] == [
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
    ]
    assert skill.param_dim == 0


def test_move_robot_preconditions_require_adjacency_and_current_position() -> None:
    skill = LightSwitchSkills.MOVE_ROBOT
    robot, current_cell, target_cell = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ADJACENT, variables=(current_cell, target_cell)),
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, current_cell)),
    })
    assert skill.add_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, target_cell))
    })
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, current_cell))
    })


def test_turn_on_light_declares_one_continuous_param() -> None:
    skill = LightSwitchSkills.TURN_ON_LIGHT
    assert skill.name == "TurnOnLight"
    assert [p.type for p in skill.parameters] == [
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.light_type,
    ]
    assert skill.param_dim == 1


def test_turn_on_light_preconditions_require_light_off() -> None:
    skill = LightSwitchSkills.TURN_ON_LIGHT
    robot, current_cell, light = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=LIGHT_IN_CELL, variables=(light, current_cell)),
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, current_cell)),
        LiftedAtom(predicate=LIGHT_OFF, variables=(light,)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=LIGHT_ON, variables=(light,))})
    assert skill.delete_effects == frozenset({LiftedAtom(predicate=LIGHT_OFF, variables=(light,))})


def test_turn_off_light_declares_one_continuous_param() -> None:
    skill = LightSwitchSkills.TURN_OFF_LIGHT
    assert skill.name == "TurnOffLight"
    assert skill.param_dim == 1


def test_turn_off_light_preconditions_require_light_on() -> None:
    skill = LightSwitchSkills.TURN_OFF_LIGHT
    robot, current_cell, light = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=LIGHT_IN_CELL, variables=(light, current_cell)),
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, current_cell)),
        LiftedAtom(predicate=LIGHT_ON, variables=(light,)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=LIGHT_OFF, variables=(light,))})
    assert skill.delete_effects == frozenset({LiftedAtom(predicate=LIGHT_ON, variables=(light,))})


def test_jump_to_light_declares_all_three_cells_and_the_light() -> None:
    skill = LightSwitchSkills.JUMP_TO_LIGHT
    assert skill.name == "JumpToLight"
    assert [p.type for p in skill.parameters] == [
        LightSwitchEnvironment.robot_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.cell_type,
        LightSwitchEnvironment.light_type,
    ]


def test_jump_to_light_claims_an_effect_its_option_never_actually_achieves() -> None:
    """The "impossible skill": the NSRT's symbolic add_effect promises the robot
    reaches cell3, but compute_action (below) always emits a no-op -- this
    mismatch is deliberate, matching predicators' own JumpToLight."""
    skill = LightSwitchSkills.JUMP_TO_LIGHT
    robot, cell1, cell2, cell3, light = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, cell1)),
        LiftedAtom(predicate=ADJACENT, variables=(cell1, cell2)),
        LiftedAtom(predicate=ADJACENT, variables=(cell2, cell3)),
        LiftedAtom(predicate=LIGHT_IN_CELL, variables=(light, cell3)),
    })
    assert skill.add_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, cell3))
    })
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_CELL, variables=(robot, cell1))
    })


def test_move_robot_ground_skill_grounds_preconditions_for_real_light_switch_cells() -> None:
    cells = LightSwitchEnvironment.get_cells()
    robot = LightSwitchEnvironment.robot
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT, objects=(robot, cells[0], cells[1])
    )
    assert ground_skill.preconditions == frozenset({
        GroundAtom(predicate=ADJACENT, objects=(cells[0], cells[1])),
        GroundAtom(predicate=ROBOT_IN_CELL, objects=(robot, cells[0])),
    })
    assert ground_skill.add_effects == frozenset({
        GroundAtom(predicate=ROBOT_IN_CELL, objects=(robot, cells[1]))
    })


def test_sample_params_returns_empty_array_for_zero_dim_skill() -> None:
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.MOVE_ROBOT,
        objects=(LightSwitchEnvironment.robot, cells[0], cells[1]),
    )
    params = LightSwitchSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    assert params.shape == (0,)


def test_sample_params_returns_values_within_unit_interval() -> None:
    cells = LightSwitchEnvironment.get_cells()
    ground_skill = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(LightSwitchEnvironment.robot, cells[-1], LightSwitchEnvironment.light),
    )
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
    move_robot = LightSwitchSkills.MOVE_ROBOT
    reconstructed_skill = Skill(
        name=move_robot.name,
        parameters=move_robot.parameters,
        preconditions=move_robot.preconditions,
        add_effects=move_robot.add_effects,
        delete_effects=move_robot.delete_effects,
        param_dim=move_robot.param_dim,
    )
    assert reconstructed_skill is not LightSwitchSkills.MOVE_ROBOT
    assert reconstructed_skill == LightSwitchSkills.MOVE_ROBOT
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
    unknown_skill = Skill(
        name="Unknown",
        parameters=(),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        param_dim=0,
    )
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
