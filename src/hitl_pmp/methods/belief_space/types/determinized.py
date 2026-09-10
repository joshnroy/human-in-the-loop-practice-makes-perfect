"""Types used by determinized belief-space search."""

from typing import Generic

from pydantic import BaseModel, ConfigDict

from .protocol import BeliefStateT, EnvironmentStateT


class DeterminizedSearchNode(BaseModel, Generic[EnvironmentStateT, BeliefStateT]):
    """Generic node information passed to the search heuristic."""

    model_config = ConfigDict(frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    stop_value: float
    path_cost: float
