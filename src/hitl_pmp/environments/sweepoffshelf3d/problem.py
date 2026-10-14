"""`SweepOffShelf3DProblem` -- the `core.Problem` facade over this domain."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import SweepOffShelf3DEnvironment
from .tasks import SweepOffShelf3DTasks

# Skills per evaluation episode: the shortest solve (`PickWiper`, `SweepShelf`) plus this
# repo's usual two spare, enough for one stray cube's pick and place. Carrying all five
# cubes by hand takes ten skills and so cannot finish an episode. The belief-space
# method's deployment value uses this same horizon.
EPISODE_HORIZON = 2 + 2


class SweepOffShelf3DProblem(Problem):
    env: SweepOffShelf3DEnvironment
    tasks: SweepOffShelf3DTasks

    def max_episode_steps(self) -> int:
        return EPISODE_HORIZON

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        """The standard loop. Nothing is rendered in-run: one skill is thousands of
        physics steps, so videos are made afterwards from the run's logs."""
        del renderer
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        for _ in range(self.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            labeled_action = policy(state)
            state = self.env.take_action(action=labeled_action.action)
            actions.append(labeled_action)
            states.append(state)
        return (
            task.goal.is_satisfied(state=state),
            [],
            EpisodeTrace(states=states, actions=actions),
        )
