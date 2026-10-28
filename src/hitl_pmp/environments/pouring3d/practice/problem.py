"""Standard task episode facade for native pouring."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import FrozenPourEnvironment
from .tasks import FrozenPourTasks


class FrozenPourProblem(Problem):
    env: FrozenPourEnvironment
    tasks: FrozenPourTasks

    def max_episode_steps(self) -> int:
        return 2 * self.env.payload_count + 6

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        del renderer
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        for _ in range(self.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            labeled = policy(state)
            state = self.env.take_action(action=labeled.action)
            actions.append(labeled)
            states.append(state)
        return task.goal.is_satisfied(state=state), [], EpisodeTrace(states=states, actions=actions)
