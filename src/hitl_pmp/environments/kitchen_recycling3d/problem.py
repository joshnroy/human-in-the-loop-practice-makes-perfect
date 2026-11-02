"""Core Problem facade; human interventions use the existing request protocol."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import KitchenRecyclingEnvironment
from .tasks import KitchenRecyclingTasks

EPISODE_HORIZON = 6


class KitchenRecyclingProblem(Problem):
    env: KitchenRecyclingEnvironment
    tasks: KitchenRecyclingTasks
    episode_horizon: int = Field(default=EPISODE_HORIZON, ge=1)

    def max_episode_steps(self) -> int:
        return self.episode_horizon

    def reset_to_task(self, *, task: Task) -> State:
        return self.env.reset_to_state(state=task.initial_state)

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        frames = [] if renderer is None else [renderer.render_frame(state=state, env=self.env)]
        for _ in range(self.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            labeled = policy(state)
            state = self.env.take_action(action=labeled.action)
            states.append(state)
            actions.append(labeled)
            if renderer is not None:
                frames.append(renderer.render_frame(state=state, env=self.env, label=labeled.label))
        return (
            task.goal.is_satisfied(state=state),
            frames,
            EpisodeTrace(states=states, actions=actions),
        )
