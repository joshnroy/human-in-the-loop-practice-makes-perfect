import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import TwoRoutesEnvironment
from .tasks import TwoRoutesTasks


class TwoRoutesProblem(Problem):
    env: TwoRoutesEnvironment
    tasks: TwoRoutesTasks

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        """One attempt: the task is to finish by whichever route the policy takes."""
        del renderer
        state = self.reset_to_task(task=task)
        labeled = policy(state)
        after = self.env.take_action(action=labeled.action)
        return (
            task.goal.is_satisfied(state=after),
            [],
            EpisodeTrace(states=[state, after], actions=[labeled]),
        )
