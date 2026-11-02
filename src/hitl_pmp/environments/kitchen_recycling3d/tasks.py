"""Independent train/test scene-seed streams and a strictly reset-free task path."""

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import KitchenRecyclingEnvironment
from .predicates import KitchenRecyclingPredicates

SCENE_SEED_LIMIT = 2**31 - 1


class KitchenRecyclingTasks(Tasks):
    env: KitchenRecyclingEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000

    _train_rng: np.random.Generator = PrivateAttr()
    _test_rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self.set_seed(seed=self.seed)

    def set_seed(self, *, seed: int) -> None:
        self.seed = seed
        self._train_rng = np.random.default_rng(seed)
        self._test_rng = np.random.default_rng(seed + self.test_env_seed_offset)

    def goal(self) -> Goal:
        return Goal(atoms=KitchenRecyclingPredicates.goal_atoms(config=self.env.config))

    def sample_train_task(self) -> Task:
        return self.build_task(scene_seed=int(self._train_rng.integers(SCENE_SEED_LIMIT)))

    def sample_test_task(self) -> Task:
        return self.build_task(scene_seed=int(self._test_rng.integers(SCENE_SEED_LIMIT)))

    def sample_train_task_in_place(self) -> Task:
        """Used under NEVER: neither a reset nor an RNG draw is permitted here."""
        return Task(initial_state=self.env.get_current_state(), goal=self.goal())

    def build_task(self, *, scene_seed: int) -> Task:
        """Explicit scene construction, only for callers authorized to reset."""
        return Task(initial_state=self.env.reset_to_seed(seed=scene_seed), goal=self.goal())
