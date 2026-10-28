"""Seed-separated task streams; in-place practice never rebuilds its world."""

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import FrozenPourEnvironment
from .predicates import PourPredicates


class FrozenPourTasks(Tasks):
    env: FrozenPourEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    _train: np.random.Generator = PrivateAttr()
    _test: np.random.Generator = PrivateAttr()
    _reset: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._train = np.random.default_rng(self.seed)
        self._test = np.random.default_rng(self.seed + self.test_env_seed_offset)
        self._reset = np.random.default_rng(self.seed + 20000)

    def goal(self) -> Goal:
        return Goal(
            atoms=frozenset({
                GroundAtom(
                    predicate=PourPredicates.get(name="Goal"),
                    objects=(self.env.objects().requests()[self.env.requested_count - 1],),
                )
            })
        )

    def build_task(self, *, seed: int) -> Task:
        state = self.env.reset_to_seed(seed=seed, requested=self.env.requested_count)
        return Task(initial_state=state, goal=self.goal())

    def sample_train_task(self) -> Task:
        return self.build_task(seed=int(self._train.integers(0, 2**31 - 1)))

    def sample_test_task(self) -> Task:
        return self.build_task(seed=int(self._test.integers(0, 2**31 - 1)))

    def sample_reset_start_task(self) -> Task:
        return self.build_task(seed=int(self._reset.integers(0, 2**31 - 1)))

    def sample_train_task_in_place(self) -> Task:
        return Task(initial_state=self.env.get_current_state(), goal=self.goal())
