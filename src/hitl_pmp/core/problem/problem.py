import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.renderer.renderer import Renderer

from .environment.environment import Environment
from .environment.types import Action, State
from .human.human import HumanOracle
from .human.types import CommandGoalDescription, CommandStartStateDescription, Cost
from .tasks.tasks import Tasks
from .tasks.types import Goal, Task


class Problem(BaseModel, abc.ABC):
    """Composition root: Environment + HumanOracle + Tasks. A real,
    constructor-injected instance now (not a static-method container): env/tasks
    are required fields (references to the actual Environment/Tasks *instances*
    this Problem drives), human is optional since not every domain has one
    (LightSwitchProblem never sets it -- no irreversible action exists there).
    Mirrors the design doc's flat Problem(ABC): every method here is a thin
    passthrough to the relevant part, except run_task_episode, which is genuine
    orchestration logic each concrete Problem must implement.

    human stays type[HumanOracle] rather than an instance, unlike env/tasks: it has
    no state of its own to hold (see human.py's own docstring), so there's nothing
    for an instance to carry that the class itself doesn't already provide.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    tasks: Tasks
    human: type[HumanOracle] | None = None

    def get_current_state(self) -> State:
        return self.env.get_current_state()

    def take_action(self, *, action: Action) -> State:
        return self.env.take_action(action=action)

    def get_valid_actions(self) -> list[Action]:
        return self.env.get_valid_actions()

    def hard_reset(self) -> None:
        self.env.hard_reset()

    def reset_to_task(self, *, task: Task) -> State:
        """Install a task's initial state as the environment's current state, and
        return it.

        Harness-only, like hard_reset -- an agent never calls this. It exists so
        both places that start an episode from a task (an evaluation episode, and
        each of PracticeLoop's free periods) go through one named operation rather
        than reaching into self.env.set_state, whose documented role is the
        HumanOracle's privileged override."""
        self.env.set_state(state=task.initial_state)
        return self.env.get_current_state()

    def _describe_command(
        self, *, goal: Goal
    ) -> tuple[CommandStartStateDescription, CommandGoalDescription]:
        return (
            CommandStartStateDescription(state=self.get_current_state()),
            CommandGoalDescription(goal=goal),
        )

    def calculate_cost_for_human_command(self, *, goal: Goal) -> Cost:
        """Query what asking the human for this would cost, without actually asking."""
        assert self.human is not None, "calculate_cost_for_human_command needs self.human set."
        start, end = self._describe_command(goal=goal)
        return self.human.calculate_cost_for_human_command(
            command_start_state_description=start, command_goal_description=end
        )

    def execute_human_command(self, *, goal: Goal) -> None:
        """The only sanctioned reset: let the human work toward goal. No return value —
        query calculate_cost_for_human_command beforehand if the cost is needed; this
        method's only job is to make it happen. self.human is responsible for
        updating self.env (it was handed env directly) to reflect whatever actually
        happened, since only it knows what that was."""
        assert self.human is not None, "execute_human_command needs self.human set."
        start, end = self._describe_command(goal=goal)
        self.human.execute_human_command(
            command_start_state_description=start, command_goal_description=end, env=self.env
        )

    def sample_train_task(self) -> Task:
        return self.tasks.sample_train_task()

    def sample_test_task(self) -> Task:
        return self.tasks.sample_test_task()

    @abc.abstractmethod
    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        """Run policy on task until goal reached or timeout; returns (succeeded, frames).
        frames is empty unless renderer is given, in which case every run is
        optionally recordable through this one path -- one frame per step (including
        the initial state) via renderer.render_frame, no separate rendering-only
        codepath needed."""
        raise NotImplementedError
