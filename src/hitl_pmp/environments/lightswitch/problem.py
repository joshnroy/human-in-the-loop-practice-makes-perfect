from typing import ClassVar

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task

from .environment import LightSwitchEnvironment
from .tasks import LightSwitchTasks


class LightSwitchProblem(Problem):
    """No HumanOracle is ever set: this domain has no irreversible action, so
    run_task_episode never needs Problem.execute_human_command."""

    # Matches the paper's own Light-Switch task horizon (Appendix F):
    # H_eval = number of grid cells + 2.
    max_episode_steps: ClassVar[int] = LightSwitchEnvironment.grid_size + 2

    @staticmethod
    def wire() -> None:
        """Point the shared Problem singleton at this domain's env/tasks. Call once
        before using any Problem.* method (get_current_state, take_action,
        run_task_episode, ...) -- matches the pattern set in
        tests/core/problem/test_problem.py's _wire_problem()."""
        Problem.env = LightSwitchEnvironment
        Problem.tasks = LightSwitchTasks

    @staticmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        Problem.env.set_state(state=task.initial_state)
        state = Problem.env.get_current_state()
        for _ in range(LightSwitchProblem.max_episode_steps):
            if task.goal.is_satisfied(state=state):
                return True
            state = Problem.env.take_action(action=policy(state))
        return task.goal.is_satisfied(state=state)
