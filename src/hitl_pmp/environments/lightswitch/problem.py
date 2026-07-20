from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import LightSwitchEnvironment
from .tasks import LightSwitchTasks


class LightSwitchProblem(Problem):
    """No HumanOracle is ever set: this domain has no irreversible action, so
    run_task_episode never needs Problem.execute_human_command. Unlike Problem's own
    facade methods (which always read Problem.env/Problem.tasks off the base class,
    since Problem is a generic container meant to be repointed at whichever domain is
    running), LightSwitchProblem is permanently tied to its own domain -- env/tasks
    are set right here in the class body, so no external wiring step is needed to
    use it."""

    env: ClassVar[type[Environment]] = LightSwitchEnvironment
    tasks: ClassVar[type[Tasks]] = LightSwitchTasks

    @staticmethod
    def max_episode_steps() -> int:
        """Matches the paper's own Light-Switch task horizon (Appendix F):
        H_eval = number of grid cells + 2. Computed fresh each call (not cached) so
        an overridden LightSwitchEnvironment.grid_size is respected."""
        return LightSwitchEnvironment.grid_size + 2

    @staticmethod
    def run_task_episode(
        *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        env = LightSwitchProblem.env
        env.set_state(state=task.initial_state)
        state = env.get_current_state()
        frames = [renderer.render_frame(state=state)] if renderer is not None else []
        for _ in range(LightSwitchProblem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                return True, frames
            state = env.take_action(action=policy(state))
            if renderer is not None:
                frames.append(renderer.render_frame(state=state))
        return task.goal.is_satisfied(state=state), frames
