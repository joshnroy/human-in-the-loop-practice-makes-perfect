from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from .environment.types import Object, State, Type


class Task(BaseModel):
    """The top-level unit of work: an initial state and a goal to reach from it."""

    initial_state: State
    goal: Goal


class Goal(BaseModel):
    atoms: frozenset[GroundAtom]

    def is_satisfied(self, *, state: State) -> bool:
        return all(atom.predicate.holds(state, atom.objects) for atom in self.atoms)


class Predicate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    types: tuple[Type, ...]
    holds: Callable[[State, tuple[Object, ...]], bool]

    def __call__(self, *, state: State, objects: tuple[Object, ...]) -> GroundAtom:
        return GroundAtom(predicate=self, objects=objects)


class GroundAtom(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: Predicate
    objects: tuple[Object, ...]
