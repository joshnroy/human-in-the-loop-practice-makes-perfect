"""Standard skill-step episodes; no step-protocol or algorithm modifications."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import RackServingEnvironment
from .objects import NUM_FOOD
from .tasks import RackServingTasks

EPISODE_HORIZON = 2 * NUM_FOOD + 2


class RackServingProblem(Problem):
    env: RackServingEnvironment
    tasks: RackServingTasks

    def max_episode_steps(self) -> int:
        """Five independent pick/place deliveries and two spare skills."""
        return EPISODE_HORIZON

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        del renderer
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        for _ in range(self.max_episode_steps()):
            if self.env.backend().task_succeeded():
                break
            labeled = policy(state)
            state = self.env.take_action(action=labeled.action)
            actions.append(labeled)
            states.append(state)
        return self.env.backend().task_succeeded(), [], EpisodeTrace(states=states, actions=actions)
