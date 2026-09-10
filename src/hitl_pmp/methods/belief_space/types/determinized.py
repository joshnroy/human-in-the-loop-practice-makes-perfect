"""Types used by determinized belief-space search."""

from typing import Generic, Protocol

from pydantic import BaseModel, ConfigDict

from .protocol import BeliefStateT, EnvironmentStateT


class DeterminizedSearchNode(BaseModel, Generic[EnvironmentStateT, BeliefStateT]):
    """Generic node information exposed to injected search policies."""

    model_config = ConfigDict(frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    stop_value: float
    path_cost: float


class DeterminizedHeuristic(Protocol[EnvironmentStateT, BeliefStateT]):
    """Additional estimated cost-to-go used to order the search frontier."""

    def __call__(
        self, *, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]
    ) -> float: ...
