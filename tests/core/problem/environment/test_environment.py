import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _DummyEnvironment(Environment):
    @staticmethod
    def take_action(*, action: Action) -> State:
        Environment.current_state = _state(x=float(action[0]))
        return Environment.current_state

    @staticmethod
    def get_valid_actions() -> list[Action]:
        return [np.array([1.0]), np.array([-1.0])]

    @staticmethod
    def hard_reset() -> None:
        Environment.set_state(state=_state(x=0.0))


def test_environment_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Environment()  # type: ignore[abstract]


def test_set_state_is_visible_through_base_class_and_subclass() -> None:
    new_state = _state(x=9.0)
    Environment.set_state(state=new_state)
    assert Environment.get_current_state() is new_state
    assert _DummyEnvironment.get_current_state() is new_state


def test_take_action_advances_current_state() -> None:
    Environment.set_state(state=_state(x=3.0))
    result = _DummyEnvironment.take_action(action=np.array([5.0]))
    assert result[_OBJ].tolist() == [5.0]
    assert Environment.get_current_state()[_OBJ].tolist() == [5.0]


def test_get_valid_actions_returns_domain_actions() -> None:
    actions = _DummyEnvironment.get_valid_actions()
    assert len(actions) == 2
    assert actions[0].tolist() == [1.0]
    assert actions[1].tolist() == [-1.0]


def test_hard_reset_sets_current_state() -> None:
    Environment.set_state(state=_state(x=99.0))
    _DummyEnvironment.hard_reset()
    assert Environment.get_current_state()[_OBJ].tolist() == [0.0]
