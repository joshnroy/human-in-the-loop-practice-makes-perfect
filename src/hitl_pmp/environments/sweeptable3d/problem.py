"""The table domain's facade and failure-aware legacy evaluation loop."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import SweepTable3DEnvironment
from .objects import NUM_CUBES
from .tasks import SweepTable3DTasks

EPISODE_HORIZON = 2 * NUM_CUBES + 2


class SweepTable3DProblem(Problem):
    env: SweepTable3DEnvironment
    tasks: SweepTable3DTasks

    def max_episode_steps(self) -> int:
        return EPISODE_HORIZON

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        """Stop after one failed sweep or any floor contact, before checking the goal.

        The goal includes SweepSucceeded, not just five occupied tray positions.
        Full-motion videos are replayed from recordings after the run, as for the shelf.
        """
        del renderer
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        failed = self.env.backend().task_failed()
        for _ in range(self.max_episode_steps()):
            if failed or task.goal.is_satisfied(state=state):
                break
            labeled_action = policy(state)
            state = self.env.take_action(action=labeled_action.action)
            actions.append(labeled_action)
            states.append(state)
            failed = failed or self.env.backend().task_failed()
        return (
            not failed and task.goal.is_satisfied(state=state),
            [],
            EpisodeTrace(states=states, actions=actions),
        )
