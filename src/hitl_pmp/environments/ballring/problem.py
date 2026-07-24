from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import BallRingEnvironment
from .tasks import BallRingTasks


class BallRingProblem(Problem):
    """No HumanOracle is set yet: PR 1 ports only the environment/tasks/facade, and
    ``run_task_episode`` here never calls ``Problem.execute_human_command``.
    (Ball-Ring *does* have genuinely irreversible actions -- a bare ball placed on a
    table falls to the floor -- so a HumanOracle is a natural later addition; it just
    is not part of this env-only PR.) env/tasks are narrowed to this domain's own
    concrete types, matching ``LightSwitchProblem``.
    """

    env: BallRingEnvironment
    tasks: BallRingTasks

    # Paper's H_eval for the simulated Ball-Ring: horizon = 8
    # (active_sampler_learning.yaml's ball_and_cup_sticky_table block). Unlike Light
    # Switch's grid-size-derived horizon, this is a fixed constant of the domain.
    horizon: ClassVar[int] = 8

    def max_episode_steps(self) -> int:
        return self.horizon

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        state = self.reset_to_task(task=task)
        frames = [renderer.render_frame(state=state, env=self.env)] if renderer is not None else []
        for _ in range(self.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                return True, frames
            labeled_action = policy(state)
            state = self.env.take_action(action=labeled_action.action)
            if renderer is not None:
                frames.append(
                    renderer.render_frame(state=state, env=self.env, label=labeled_action.label)
                )
        return task.goal.is_satisfied(state=state), frames
