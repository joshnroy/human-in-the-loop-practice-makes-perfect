import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import TossingRoomEnvironment
from .tasks import TossingRoomTasks


class TossingRoomProblem(Problem):
    """No HumanOracle is set (Problem.human stays None): the irreversible ledge exists
    in the dynamics, but the oracle solves every task forward-only and never needs to
    be lifted back up it, so run_task_episode never needs Problem.execute_human_command
    -- mirrors LightSwitchProblem. env/tasks are required constructor fields, narrowed
    to this domain's own concrete types."""

    env: TossingRoomEnvironment
    tasks: TossingRoomTasks

    def max_episode_steps(self) -> int:
        """A generous bound on any goal's forward-only solve: Pickup + at most
        num_rooms moves + Throw (or a walk to the button + Press). 2 * num_rooms + 2
        comfortably covers all three families for any layout. Computed fresh each call
        so an overridden self.env.num_rooms is respected."""
        return 2 * self.env.num_rooms + 2

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
