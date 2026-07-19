from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from .environment import Environment, Object, State, Type
from .human_oracle import Cost, HumanOracle

if TYPE_CHECKING:
    from .method import Policy


class Problem(abc.ABC):
    """Composition root: one Environment + one HumanOracle + a task distribution.

    A static-method container, never instantiated — env/human are class attributes
    (references to the Environment/HumanOracle *classes*, themselves also never
    instantiated). Unlike Environment, a Problem is NOT reusable across research
    questions.
    """

    env: ClassVar[type[Environment]]
    human: ClassVar[type[HumanOracle]]

    @staticmethod
    @abc.abstractmethod
    def get_train_tasks() -> set[Task]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_test_task() -> Task:
        """A randomly sampled test task, not known to the agent ahead of time."""
        raise NotImplementedError

    @staticmethod
    def request_human_reset(*, goal_state: State) -> Cost:
        """The only sanctioned reset: pay the human's cost, let them move the state."""
        cost = Problem.human.send_command(
            start_state=Problem.env.get_current_state(), goal_state=goal_state
        )
        if cost != float("inf"):
            Problem.env.set_state(state=goal_state)
        return cost

    @staticmethod
    @abc.abstractmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        """Run policy on task until goal reached or timeout; returns whether it succeeded."""
        raise NotImplementedError


class Task(BaseModel):
    """The top-level unit of work: an initial state and a goal to reach from it."""

    model_config = ConfigDict(frozen=True)

    initial_state: State
    goal: Goal


class Goal(BaseModel):
    model_config = ConfigDict(frozen=True)

    atoms: frozenset[GroundAtom]

    def is_satisfied(self, *, state: State) -> bool:
        return all(atom.predicate.holds(state, atom.objects) for atom in self.atoms)


class GroundAtom(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: Predicate
    objects: tuple[Object, ...]


class Predicate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    types: tuple[Type, ...]
    holds: Callable[[State, tuple[Object, ...]], bool]

    def __call__(self, *, state: State, objects: tuple[Object, ...]) -> GroundAtom:
        return GroundAtom(predicate=self, objects=objects)


for _model in (Task, Goal, GroundAtom, Predicate):
    _model.model_rebuild()
