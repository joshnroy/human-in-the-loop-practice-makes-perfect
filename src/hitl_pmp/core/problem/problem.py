import abc
from typing import ClassVar

from hitl_pmp.core.method.types import Policy

from .environment.environment import Environment
from .environment.types import Action, State
from .human_oracle.human_oracle import HumanOracle
from .human_oracle.types import Cost
from .tasks.tasks import Tasks
from .tasks.types import Task


class Problem(abc.ABC):
    """Composition root: Environment + HumanOracle + Tasks. A static-method
    container, never instantiated — env/human/tasks are class attributes
    (references to the Environment/HumanOracle/Tasks *classes*, themselves also
    never instantiated). Mirrors the design doc's flat Problem(ABC): every method
    here is a thin passthrough to the relevant part, except run_task_episode,
    which is genuine orchestration logic each concrete Problem must implement.
    """

    env: ClassVar[type[Environment]]
    human: ClassVar[type[HumanOracle]]
    tasks: ClassVar[type[Tasks]]

    @staticmethod
    def get_current_state() -> State:
        return Problem.env.get_current_state()

    @staticmethod
    def take_action(*, action: Action) -> State:
        return Problem.env.take_action(action=action)

    @staticmethod
    def get_valid_actions() -> set[Action]:
        return Problem.env.get_valid_actions()

    @staticmethod
    def hard_reset() -> None:
        Problem.env.hard_reset()

    @staticmethod
    def send_human_command(*, goal_state: State) -> Cost:
        """The only sanctioned reset: pay the human's cost, let them move the state."""
        cost = Problem.human.send_command(
            start_state=Problem.get_current_state(), goal_state=goal_state
        )
        if cost != float("inf"):
            Problem.env.set_state(state=goal_state)
        return cost

    @staticmethod
    def get_train_tasks() -> list[Task]:
        return Problem.tasks.get_train_tasks()

    @staticmethod
    def get_test_task() -> Task:
        return Problem.tasks.get_test_task()

    @staticmethod
    @abc.abstractmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        """Run policy on task until goal reached or timeout; returns whether it succeeded."""
        raise NotImplementedError
