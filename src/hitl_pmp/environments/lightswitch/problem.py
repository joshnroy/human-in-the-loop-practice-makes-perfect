from typing import ClassVar

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Task

from .environment import LightSwitchEnvironment
from .tasks import LightSwitchTasks


class LightSwitchProblem(Problem):
    """No HumanOracle is ever set: this domain has no irreversible action, so
    run_task_episode never needs Problem.execute_human_command. Unlike Problem's own
    facade methods (which always read Problem.env/Problem.tasks off the base class,
    since Problem is a generic container meant to be repointed at whichever domain is
    running), LightSwitchProblem is permanently tied to its own domain -- env/tasks
    are set right here in the class body, so no external wiring step is needed to
    use it."""

    env: ClassVar[type[Environment]] = LightSwitchEnvironment
    tasks: ClassVar[type[Tasks]] = LightSwitchTasks

    @staticmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        env = LightSwitchProblem.env
        env.set_state(state=task.initial_state)
        state = env.get_current_state()
        # Matches the paper's own Light-Switch task horizon (Appendix F):
        # H_eval = number of grid cells + 2. Computed here (not cached) so an
        # overridden LightSwitchEnvironment.grid_size is respected at call time.
        max_steps = LightSwitchEnvironment.grid_size + 2
        for _ in range(max_steps):
            if task.goal.is_satisfied(state=state):
                return True
            state = env.take_action(action=policy(state))
        return task.goal.is_satisfied(state=state)
