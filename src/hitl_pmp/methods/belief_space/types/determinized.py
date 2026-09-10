"""Types used by determinized belief-space search."""

from typing import Generic

from pydantic import BaseModel, ConfigDict

from .protocol import ActionT, BeliefStateT, EnvironmentStateT
from .stop_action import StopAction


class DeterminizedSearchQueueEntry(
    BaseModel,
    Generic[EnvironmentStateT, BeliefStateT, ActionT],
):
    """Algorithm 3's ``(b, g)`` plus path-recovery and cache metadata.

    In this implementation, ``b`` is represented by ``environment_state``,
    ``belief_state``, and ``summed_cost``. The remaining fields are bookkeeping:
    ``key`` supports Closed/Cost lookup, ``stop_value`` caches J(C, b), ``depth``
    supports diagnostics/heuristics, and ``first_action`` recovers Algorithm 3's
    line-26 answer without retaining complete paths.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    g: float
    sequence: int
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
        return (self.g, self.sequence) < (other.g, other.sequence)


class DeterminizedSearchNode(BaseModel, Generic[EnvironmentStateT, BeliefStateT]):
    """Generic node information passed to the search heuristic."""

    model_config = ConfigDict(frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    stop_value: float
    g: float


class DeterminizedSearchDiagnostics(BaseModel):
    """Record-keeping counters; none affect Algorithm 3's search decisions."""

    expanded_nodes: int = 0
    generated_successors: int = 0
    merged_nodes: int = 0
    action_transitions_evaluated: int = 0
    chance_outcomes_enumerated: int = 0
    max_frontier_size: int = 0
    max_depth_reached: int = 0
    termination_reason: str = "frontier_exhausted"

    def observe_frontier(self, *, size: int) -> None:
        self.max_frontier_size = max(self.max_frontier_size, size)

    def observe_expansion(self, *, depth: int) -> None:
        self.expanded_nodes += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
