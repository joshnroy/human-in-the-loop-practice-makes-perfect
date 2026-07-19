import abc
from typing import ClassVar

from hitl_pmp.core.method.types import Policy

from .environment.environment import Environment
from .environment.types import State
from .human_oracle.human_oracle import HumanOracle
from .human_oracle.types import Cost
from .types import Task


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
    def get_train_tasks() -> list[Task]:
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
