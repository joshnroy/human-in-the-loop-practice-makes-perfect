import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _DummyEnvironment(Environment):
    def take_action(self, *, action: Action) -> State:
        self.set_state(state=_state(x=float(action[0])))
        return self.get_current_state()

    def get_valid_actions(self) -> list[Action]:
        return [np.array([1.0]), np.array([-1.0])]

    def hard_reset(self) -> None:
        self.set_state(state=_state(x=0.0))


def test_environment_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Environment()  # type: ignore[abstract]


def test_get_current_state_raises_before_any_state_is_ever_set() -> None:
    env = _DummyEnvironment()
    with pytest.raises(AssertionError):
        env.get_current_state()


def test_set_state_and_get_current_state_round_trip() -> None:
    env = _DummyEnvironment()
    new_state = _state(x=9.0)
    env.set_state(state=new_state)
    assert env.get_current_state() is new_state


def test_take_action_advances_current_state() -> None:
    env = _DummyEnvironment()
    env.set_state(state=_state(x=3.0))
    result = env.take_action(action=np.array([5.0]))
    assert result[_OBJ].tolist() == [5.0]
    assert env.get_current_state()[_OBJ].tolist() == [5.0]


def test_get_valid_actions_returns_domain_actions() -> None:
    env = _DummyEnvironment()
    actions = env.get_valid_actions()
    assert len(actions) == 2
    assert actions[0].tolist() == [1.0]
    assert actions[1].tolist() == [-1.0]


def test_hard_reset_sets_current_state() -> None:
    env = _DummyEnvironment()
    env.set_state(state=_state(x=99.0))
    env.hard_reset()
    assert env.get_current_state()[_OBJ].tolist() == [0.0]


def test_two_instances_carry_independent_state() -> None:
    """The whole point of this refactor: no shared/global current_state anymore."""
    first = _DummyEnvironment()
    second = _DummyEnvironment()
    first.set_state(state=_state(x=1.0))
    second.set_state(state=_state(x=2.0))
    assert first.get_current_state()[_OBJ].tolist() == [1.0]
    assert second.get_current_state()[_OBJ].tolist() == [2.0]
