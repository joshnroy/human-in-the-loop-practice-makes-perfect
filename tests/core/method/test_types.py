import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill, Rollout, SetupCommand, SetupCommandTarget, Skill
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


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


def test_skill_declares_types_and_param_dim() -> None:
    skill = Skill(name="Move", types=(_BLOCK, _BLOCK), param_dim=1)
    assert skill.name == "Move"
    assert skill.types == (_BLOCK, _BLOCK)
    assert skill.param_dim == 1


def test_skills_with_equal_content_are_equal_and_hashable() -> None:
    a = Skill(name="Move", types=(_BLOCK,), param_dim=0)
    b = Skill(name="Move", types=(_BLOCK,), param_dim=0)
    assert a == b
    assert hash(a) == hash(b)


def test_ground_skill_binds_a_skill_to_concrete_objects() -> None:
    skill = Skill(name="Move", types=(_BLOCK,), param_dim=0)
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ,))
    assert ground_skill.skill == skill
    assert ground_skill.objects == (_OBJ,)


def test_ground_skills_with_equal_content_are_equal_and_hashable() -> None:
    skill = Skill(name="Move", types=(_BLOCK,), param_dim=0)
    a = GroundSkill(skill=skill, objects=(_OBJ,))
    b = GroundSkill(skill=skill, objects=(_OBJ,))
    assert a == b
    assert hash(a) == hash(b)
