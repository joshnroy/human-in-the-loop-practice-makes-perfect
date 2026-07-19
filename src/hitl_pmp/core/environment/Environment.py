import abc
from typing import ClassVar

from gymnasium.spaces import Space

from .types import Action, State


class Environment(abc.ABC):
    """Pure dynamics, as a static-method container — never instantiated. Concrete
    subclasses set action_space and any of their own internal state as class
    attributes; all methods are static and keyword-only. The most external/
    foundational module in core/ — nothing else here is imported into it."""

    action_space: ClassVar[Space]

    @staticmethod
    @abc.abstractmethod
    def get_current_state() -> State:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def simulate(*, state: State, action: Action) -> State:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_valid_actions(*, state: State) -> set[Action]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def set_state(*, state: State) -> None:
        """Privileged setter used by Problem/HumanOracle to force a state; NOT a semantic reset."""
        raise NotImplementedError
