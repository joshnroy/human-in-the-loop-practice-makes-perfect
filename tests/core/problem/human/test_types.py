import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.human.types import CommandGoalDescription, CommandStartStateDescription
from hitl_pmp.core.problem.tasks.types import Goal

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def test_command_start_state_description_wraps_state() -> None:
    state = State(data={_OBJ: np.array([1.0])})
    description = CommandStartStateDescription(state=state)
    assert description.state is state


def test_command_goal_description_wraps_goal() -> None:
    goal = Goal(atoms=frozenset())
    description = CommandGoalDescription(goal=goal)
    assert description.goal is goal


def test_command_goal_description_carries_no_target_state_by_default() -> None:
    """The original shape -- "bring this goal about, however you like" -- is untouched,
    so every caller that predates the reset command keeps its behaviour."""
    assert CommandGoalDescription(goal=Goal(atoms=frozenset())).target_state is None


def test_command_goal_description_can_carry_a_target_state() -> None:
    """A reset command: "put the world back into exactly this configuration". See the
    field's own comment for why that is a second kind of command rather than a
    weakening of the symbolic goal."""
    state = State(data={_OBJ: np.array([1.0])})
    description = CommandGoalDescription(goal=Goal(atoms=frozenset()), target_state=state)
    assert description.target_state is state
