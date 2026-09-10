"""Types used by determinized belief-space search."""

from typing import Generic

from pydantic import BaseModel, ConfigDict

from .protocol import ActionT, BeliefStateT, EnvironmentStateT
from .stop_action import StopAction


class DeterminizedSearchQueueEntry(
    BaseModel,
    Generic[EnvironmentStateT, BeliefStateT, ActionT],
):
    """One entry in Algorithm 3's ``Open`` priority queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    priority: float
    sequence: int
    path_cost: float
    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    key: object
    stop_value: float
    first_action: ActionT | StopAction

    def __lt__(self, other: object, /) -> bool:  # noqa: PLR0917
        if not isinstance(other, DeterminizedSearchQueueEntry):
            return NotImplemented
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class DeterminizedSearchNode(BaseModel, Generic[EnvironmentStateT, BeliefStateT]):
    """Generic node information passed to the search heuristic."""

    model_config = ConfigDict(frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    stop_value: float
    path_cost: float
