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
