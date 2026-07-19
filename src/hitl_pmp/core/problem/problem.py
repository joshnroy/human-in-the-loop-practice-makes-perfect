import abc
from typing import ClassVar

from hitl_pmp.core.method.types import Policy

from .environment.environment import Environment
from .environment.types import Action, State
from .human.human import HumanOracle
from .human.types import CommandGoalDescription, CommandStartStateDescription, Cost
from .tasks.tasks import Tasks
from .tasks.types import Goal, Task


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
    def _describe_command(
        *, goal: Goal
    ) -> tuple[CommandStartStateDescription, CommandGoalDescription]:
        return (
            CommandStartStateDescription(state=Problem.get_current_state()),
            CommandGoalDescription(goal=goal),
        )

    @staticmethod
    def calculate_cost_for_human_command(*, goal: Goal) -> Cost:
        """Query what asking the human for this would cost, without actually asking."""
        start, end = Problem._describe_command(goal=goal)
        return Problem.human.calculate_cost_for_human_command(
            command_start_state_description=start, command_goal_description=end
        )

    @staticmethod
    def execute_human_command(*, goal: Goal) -> None:
        """The only sanctioned reset: let the human work toward goal. No return value —
        query calculate_cost_for_human_command beforehand if the cost is needed; this
        method's only job is to make it happen. Problem.human is responsible for
        updating Problem.env (it was handed env directly) to reflect whatever actually
        happened, since only it knows what that was."""
        start, end = Problem._describe_command(goal=goal)
        Problem.human.execute_human_command(
            command_start_state_description=start, command_goal_description=end, env=Problem.env
        )

    @staticmethod
    def sample_train_task() -> Task:
        return Problem.tasks.sample_train_task()

    @staticmethod
    def sample_test_task() -> Task:
        return Problem.tasks.sample_test_task()

    @staticmethod
    @abc.abstractmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        """Run policy on task until goal reached or timeout; returns whether it succeeded."""
        raise NotImplementedError
