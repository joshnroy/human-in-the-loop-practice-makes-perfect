"""Seeded table tasks with independent train, test and human full-reset streams."""

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task

from .environment import SweepTable3DEnvironment
from .predicates import SweepTable3DPredicates

SCENE_SEED_LIMIT = 2**31 - 1


class SweepTable3DTasks(Tasks):
    env: SweepTable3DEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    reset_start_seed_offset: int = 20000

    _train_rng: np.random.Generator = PrivateAttr()
    _test_rng: np.random.Generator = PrivateAttr()
    _reset_start_rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._train_rng = np.random.default_rng(self.seed)
        self._test_rng = np.random.default_rng(self.seed + self.test_env_seed_offset)
        self._reset_start_rng = np.random.default_rng(self.seed + self.reset_start_seed_offset)

    @staticmethod
    def goal() -> Goal:
        return Goal(atoms=SweepTable3DPredicates.goal_atoms())

    def sample_train_task(self) -> Task:
        return self.build_task(scene_seed=int(self._train_rng.integers(0, SCENE_SEED_LIMIT)))

    def sample_test_task(self) -> Task:
        return self.build_task(scene_seed=int(self._test_rng.integers(0, SCENE_SEED_LIMIT)))

    def sample_reset_start_task(self) -> Task:
        return self.build_task(scene_seed=int(self._reset_start_rng.integers(0, SCENE_SEED_LIMIT)))

    def sample_train_task_in_place(self) -> Task:
        return Task(initial_state=self.env.get_current_state(), goal=self.goal())

    def build_task(self, *, scene_seed: int) -> Task:
        return Task(initial_state=self.env.reset_to_seed(seed=scene_seed), goal=self.goal())
