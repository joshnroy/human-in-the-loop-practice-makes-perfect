"""Independent deployment episodes and persistent, cold in-place practice."""

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.pouring3d.practice.problem import FrozenPourProblem
from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks

from .environment import KitchenEnvironment
from .types import KitchenObjects


class KitchenTasks(FrozenPourTasks):
    env: KitchenEnvironment

    def build_task(self, *, seed: int, requested: int | None = None) -> Task:
        requested = self.env.requested_count if requested is None else requested
        # Task generation is an explicit harness operation, not a robot action.
        if not self.env.deployment and self.env._started:
            raise ValueError("Active practice must sample_train_task_in_place; no free reset")
        state = self.env.initialize_harness_episode(seed=seed, requested=requested)
        return Task(initial_state=state, goal=self.goal())


class KitchenProblem(FrozenPourProblem):
    env: KitchenEnvironment
    tasks: KitchenTasks

    def reset_to_task(self, *, task: Task) -> State:
        if not self.env.deployment and self.env._started:
            raise ValueError("Active kitchen practice cannot reset to a task")
        state = task.initial_state
        return self.env.initialize_harness_episode(
            seed=int(state.get(obj=KitchenObjects.scene, feature_name="seed")),
            requested=int(state.get(obj=KitchenObjects.scene, feature_name="requested_count")),
        )
