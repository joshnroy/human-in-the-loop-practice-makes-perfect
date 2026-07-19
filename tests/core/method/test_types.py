import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import Rollout, SetupCommand, SetupCommandTarget
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
