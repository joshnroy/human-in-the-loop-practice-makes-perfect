"""Data types and model callbacks for belief-space search."""

from collections.abc import Hashable, Iterable
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")
StateContraT = TypeVar("StateContraT", bound=Hashable, contravariant=True)
ActionContraT = TypeVar("ActionContraT", contravariant=True)
ActionCoT = TypeVar("ActionCoT", covariant=True)


class ChanceOutcome(BaseModel, Generic[StateT]):
    """One exactly enumerated successor of an action."""

    model_config = ConfigDict(frozen=True)

    probability: float
    next_state: StateT


class ExpectimaxResult(BaseModel, Generic[ActionT]):
    """Optimal value and first action; None means stop now."""

    model_config = ConfigDict(frozen=True)

    value: float
    action: ActionT | None


class StopValue(Protocol[StateContraT]):
    def __call__(self, *, state: StateContraT) -> float: ...


class AvailableActions(Protocol[StateContraT, ActionCoT]):
    def __call__(self, *, state: StateContraT) -> Iterable[ActionCoT]: ...


class ChanceOutcomes(Protocol[StateT, ActionContraT]):
    def __call__(
        self, *, state: StateT, action: ActionContraT
    ) -> Iterable[ChanceOutcome[StateT]]: ...
