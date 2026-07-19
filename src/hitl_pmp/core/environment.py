from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np
from gymnasium.spaces import Space
from pydantic import BaseModel, ConfigDict


class Environment(abc.ABC):
    """Pure dynamics, as a static-method container — never instantiated. Concrete
    subclasses set action_space and any of their own internal state as class
    attributes; all methods are static and keyword-only. The most external/
    foundational file in core/ — nothing else here is imported into it."""

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


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: dict[Object, np.ndarray]

    def __getitem__(self, obj: Object) -> np.ndarray:  # noqa: PLR0917 (dunder: no `*` possible)
        return self.data[obj]


class Object(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: Type


class Type(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    parent: Type | None = None


Action = np.ndarray

for _model in (State, Object, Type):
    _model.model_rebuild()
