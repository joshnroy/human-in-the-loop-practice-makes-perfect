"""Types used by determinized belief-space search."""

from typing import Generic

from pydantic import BaseModel, ConfigDict

from .protocol import ActionT, BeliefStateT, EnvironmentStateT
from .stop_action import StopAction


class DeterminizedSearchBelief(
    BaseModel,
    Generic[EnvironmentStateT, BeliefStateT],
):
    """Our factored representation of Algorithm 3's belief ``b``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float


class DeterminizedNodeDiagnosticInfo(BaseModel):
    """Per-node information used only for diagnostics."""

    model_config = ConfigDict(frozen=True)

    depth: int


class DeterminizedNodeCacheInfo(BaseModel):
    """Bookkeeping used for Closed/Cost lookup and cached J(C, b)."""

    model_config = ConfigDict(frozen=True)

    key: object
    stop_value: float
    queue_sequence: int


class DeterminizedPathRecoveryInfo(BaseModel, Generic[ActionT]):
    """Minimal path data needed to return Algorithm 3's first action."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    first_action: ActionT | StopAction


class DeterminizedSearchQueueEntry(
    BaseModel,
    Generic[EnvironmentStateT, BeliefStateT, ActionT],
):
    """Recursive representation of Algorithm 3's ``(b, g)`` and bookkeeping."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    belief: DeterminizedSearchBelief[EnvironmentStateT, BeliefStateT]
    g: float
    diagnostic_info: DeterminizedNodeDiagnosticInfo
    cache_info: DeterminizedNodeCacheInfo
    path_recovery_info: DeterminizedPathRecoveryInfo[ActionT]

    def __lt__(self, other: object, /) -> bool:  # noqa: PLR0917
        if not isinstance(other, DeterminizedSearchQueueEntry):
            return NotImplemented
        return (self.g, self.cache_info.queue_sequence) < (
            other.g,
            other.cache_info.queue_sequence,
        )


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
