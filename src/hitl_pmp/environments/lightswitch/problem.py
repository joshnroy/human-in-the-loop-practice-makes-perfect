import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import LightSwitchEnvironment
from .tasks import LightSwitchTasks


class LightSwitchProblem(Problem):
    """No HumanOracle is ever set: this domain has no irreversible action, so
    run_task_episode never needs Problem.execute_human_command. env/tasks are
    still required constructor fields (inherited from Problem) -- unlike the old
    ClassVar-singleton design, there's no domain-fixed default to bake in here,
    since a caller might legitimately want two independently-configured
    LightSwitchEnvironment instances (e.g. different grid_size) each wrapped in
    their own LightSwitchProblem."""

    env: LightSwitchEnvironment
    tasks: LightSwitchTasks

    def max_episode_steps(self) -> int:
        """Matches the paper's own Light-Switch task horizon (Appendix F):
        H_eval = number of grid cells + 2. Computed fresh each call (not cached) so
        an overridden self.env.grid_size is respected."""
        return self.env.grid_size + 2

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
