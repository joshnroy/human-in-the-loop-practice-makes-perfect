import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    Rollout,
    SetupCommand,
    SetupCommandTarget,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)
_OBJ2 = Object(name="block2", type=_BLOCK)
_OBJ3 = Object(name="block3", type=_BLOCK)

_AT = Predicate(name="At", types=(_BLOCK, _BLOCK), holds=lambda state, objects: True)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


def _move_skill() -> tuple[Skill, Variable, Variable, Variable]:
    """A generic "Move" skill: At(robot, current) -> At(robot, target), matching
    the shape LightSwitchSkills.MOVE_ROBOT will use, but domain-agnostic here."""
    robot = Variable(name="robot", type=_BLOCK)
    current = Variable(name="current", type=_BLOCK)
    target = Variable(name="target", type=_BLOCK)
    skill = Skill(
        name="Move",
        parameters=(robot, current, target),
        preconditions=frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))}),
        add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, target))}),
        delete_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))}),
        param_dim=0,
    )
    return skill, robot, current, target


def test_rollout_accepts_one_fewer_action_than_states() -> None:
    rollout = Rollout(states=[_state(x=0.0), _state(x=1.0)], actions=[np.array([1.0])])
    assert len(rollout.actions) == len(rollout.states) - 1


def test_rollout_accepts_single_state_and_no_actions() -> None:
    rollout = Rollout(states=[_state(x=0.0)], actions=[])
    assert rollout.actions == []


def test_rollout_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        Rollout(states=[_state(x=0.0)], actions=[np.array([1.0])])


def test_setup_command_for_robot_target() -> None:
    command = SetupCommand(target=SetupCommandTarget.ROBOT, goal=Goal(atoms=frozenset()))
    assert command.target is SetupCommandTarget.ROBOT


def test_setup_command_for_human_target() -> None:
    command = SetupCommand(target=SetupCommandTarget.HUMAN, goal=Goal(atoms=frozenset()))
    assert command.target is SetupCommandTarget.HUMAN


def test_variable_declares_a_name_and_type() -> None:
    var = Variable(name="robot", type=_BLOCK)
    assert var.name == "robot"
    assert var.type == _BLOCK


def test_variables_with_equal_content_are_equal_and_hashable() -> None:
    a = Variable(name="robot", type=_BLOCK)
    b = Variable(name="robot", type=_BLOCK)
    assert a == b
    assert hash(a) == hash(b)


def test_lifted_atom_grounds_into_a_ground_atom_via_substitution() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    lifted = LiftedAtom(predicate=_AT, variables=(robot, robot))
    ground = lifted.ground(substitution={robot: _OBJ})
    assert ground == GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ))


def test_lifted_atoms_with_equal_content_are_equal_and_hashable() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    a = LiftedAtom(predicate=_AT, variables=(robot,))
    b = LiftedAtom(predicate=_AT, variables=(robot,))
    assert a == b
    assert hash(a) == hash(b)


def test_skill_declares_parameters_preconditions_effects_and_param_dim() -> None:
    skill, robot, current, target = _move_skill()
    assert skill.name == "Move"
    assert skill.parameters == (robot, current, target)
    assert skill.preconditions == frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))})
    assert skill.add_effects == frozenset({LiftedAtom(predicate=_AT, variables=(robot, target))})
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=_AT, variables=(robot, current))
    })
    assert skill.param_dim == 0


def test_skill_rejects_a_precondition_variable_not_in_parameters() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    stray = Variable(name="stray", type=_BLOCK)
    with pytest.raises(ValidationError):
        Skill(
            name="Move",
            parameters=(robot,),
            preconditions=frozenset({LiftedAtom(predicate=_AT, variables=(robot, stray))}),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            param_dim=0,
        )


def test_skill_rejects_an_effect_variable_not_in_parameters() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    stray = Variable(name="stray", type=_BLOCK)
    with pytest.raises(ValidationError):
        Skill(
            name="Move",
            parameters=(robot,),
            preconditions=frozenset(),
            add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, stray))}),
            delete_effects=frozenset(),
            param_dim=0,
        )


def test_skills_with_equal_content_are_equal_and_hashable() -> None:
    a, *_ = _move_skill()
    b, *_ = _move_skill()
    assert a == b
    assert hash(a) == hash(b)


def test_ground_skill_binds_a_skill_to_concrete_objects() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.skill == skill
    assert ground_skill.objects == (_OBJ, _OBJ2, _OBJ3)


def test_ground_skill_rejects_wrong_number_of_objects() -> None:
    skill, *_ = _move_skill()
    with pytest.raises(ValidationError):
        GroundSkill(skill=skill, objects=(_OBJ, _OBJ2))


def test_ground_skill_rejects_an_object_whose_type_does_not_match_its_parameter() -> None:
    skill, *_ = _move_skill()
    wrong_type = Type(name="not_a_block", feature_names=())
    wrong_obj = Object(name="thing", type=wrong_type)
    with pytest.raises(ValidationError):
        GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, wrong_obj))


def test_ground_skill_grounds_preconditions_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.preconditions == frozenset({
        GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ2))
    })


def test_ground_skill_grounds_add_effects_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.add_effects == frozenset({GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ3))})


def test_ground_skill_grounds_delete_effects_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.delete_effects == frozenset({
        GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ2))
    })


def test_ground_skills_with_equal_content_are_equal_and_hashable() -> None:
    skill, *_ = _move_skill()
    a = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    b = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert a == b
    assert hash(a) == hash(b)
