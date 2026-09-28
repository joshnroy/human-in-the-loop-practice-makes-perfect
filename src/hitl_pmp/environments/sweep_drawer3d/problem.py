"""Separate evaluation episodes with exact all-five goal and per-cube diagnostics."""

import numpy as np

from hitl_pmp.core.method.types import EpisodeTrace, LabeledAction, Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import SweepDrawerEnvironment
from .symbolic import SweepSymbols
from .tasks import SweepDrawerTasks


class SweepDrawerProblem(Problem):
    env: SweepDrawerEnvironment
    tasks: SweepDrawerTasks
    deployment_horizon: int = 5

    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        if not self.env.evaluation:
            raise RuntimeError("Scored episodes must use the separate Sweep evaluation simulator")
        state = self.reset_to_task(task=task)
        states = [state]
        actions: list[LabeledAction] = []
        frames: list[np.ndarray] = []
        for _ in range(self.deployment_horizon):
            if task.goal.is_satisfied(state=state):
                break
            labeled = policy(state)
            state = self.env.take_action(action=labeled.action)
            actions.append(labeled)
            states.append(state)
            if renderer is not None:
                frames.append(renderer.render_frame(state=state, env=self.env))
        success = task.goal.is_satisfied(state=state)
        self.env._write_event(
            event={
                "kind": "evaluation",
                "seed": self.env.session().seed,
                "solved": success,
                "cubes_solved": sum(
                    int(state.get(obj=SweepSymbols.SCENE, feature_name=f"InDrawer{i}"))
                    for i in range(5)
                ),
                "cubes_total": 5,
                "actions": len(actions),
                "horizon": self.deployment_horizon,
            }
        )
        return success, frames, EpisodeTrace(states=states, actions=actions)
