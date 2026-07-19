import abc

from hitl_pmp.core.environment.types import State

from .types import Cost


class HumanOracle(abc.ABC):
    """The v0-v3 human-cost-model axis from the design doc; a static-method container,
    never instantiated, swappable independent of Environment."""

    @staticmethod
    @abc.abstractmethod
    def send_command(*, start_state: State, goal_state: State) -> Cost:
        """Move the environment from start_state to goal_state; cost is inf if infeasible."""
        raise NotImplementedError
