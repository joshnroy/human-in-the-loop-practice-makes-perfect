"""Fixed, prevalidated held-out seeds and side-effect-free continuous practice."""

from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import SweepSimpleEnvironment
from .skill_provider import SweepSimpleSkillProvider


class SweepSimpleTasks(Tasks):
    env: SweepSimpleEnvironment
    test_seeds: tuple[int, ...]
    _test_index: int = PrivateAttr(default=0)

    def sample_train_task(self) -> Task:
        return self.sample_train_task_in_place()

    def sample_train_task_in_place(self) -> Task:
        return Task(
            initial_state=self.env.get_current_state(),
            goal=Goal(atoms=SweepSimpleSkillProvider.deployment_goal_atoms()),
        )

    def sample_test_task(self) -> Task:
        if not self.env.evaluation:
            raise RuntimeError(
                "Held-out task construction requires the isolated evaluation simulator"
            )
        if self._test_index >= len(self.test_seeds):
            raise ValueError("Requested more test tasks than the frozen valid-seed manifest")
        seed = self.test_seeds[self._test_index]
        self._test_index += 1
        state = self.env.reset_to_seed(seed=seed)
        return Task(
            initial_state=state, goal=Goal(atoms=SweepSimpleSkillProvider.deployment_goal_atoms())
        )
