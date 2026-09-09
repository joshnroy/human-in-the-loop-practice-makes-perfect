"""Common interface for interchangeable belief-space planners."""

import abc
from typing import Generic

from .types.protocol import (
    ActionT,
    BeliefSpaceModel,
    BeliefStateT,
    EnvironmentStateT,
    ThetaT,
)
from .types.search_trace import SearchTrace
from .types.stop_action import StopAction


class BeliefSpacePlanner(Generic[EnvironmentStateT, BeliefStateT, ThetaT, ActionT], abc.ABC):
    """A planner that can be injected without changing a domain model."""

    name: str

    @abc.abstractmethod
    def solve(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        num_samples: int,
        trace: SearchTrace | None = None,
    ) -> tuple[float, ActionT | StopAction]:
        """Return the selected action and its planner-specific value."""
        raise NotImplementedError
