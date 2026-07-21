from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.types import Object, State, Type


class Task(BaseModel):
    """The top-level unit of work: an initial state and a goal to reach from it."""

    initial_state: State
    goal: Goal


class Goal(BaseModel):
    """The set of GroundAtoms that must all hold in a State for a Task built
    around this Goal to count as solved."""

    atoms: frozenset[GroundAtom]

    def is_satisfied(self, *, state: State) -> bool:
        return all(atom.predicate.holds(state, atom.objects) for atom in self.atoms)


class Predicate(BaseModel):
    """A named, typed relation with a state-dependent truth classifier (holds),
    not yet applied to any particular objects -- the lifted half of GroundAtom,
    the same way Skill is the lifted half of GroundSkill (method/types.py).
    Applying it to concrete Objects (via __call__ here, or LiftedAtom.ground for
    the Skill-variable case in method/types.py) produces a GroundAtom."""

    model_config = ConfigDict(frozen=True)

    name: str
    types: tuple[Type, ...]
    holds: Callable[[State, tuple[Object, ...]], bool]

    def __call__(self, *, state: State, objects: tuple[Object, ...]) -> GroundAtom:
        return GroundAtom(predicate=self, objects=objects)


class GroundAtom(BaseModel):
    """A Predicate applied to concrete Objects: one specific fact that either
    holds or doesn't in a given State (predicate.holds(state, objects)). The
    ground half of Predicate -- and, via LiftedAtom.ground (method/types.py), of
    LiftedAtom too. This is what a Skill's (grounded) preconditions/effects and
    planning.SkillGrounder's true_atoms are made of once everything is grounded,
    and what a Goal's own atoms are."""

    model_config = ConfigDict(frozen=True)

    predicate: Predicate
    objects: tuple[Object, ...]
