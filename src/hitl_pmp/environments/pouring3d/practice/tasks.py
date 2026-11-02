"""Seed-separated task streams; in-place practice never rebuilds its world."""

import numpy as np
from pydantic import PrivateAttr, StrictInt, model_validator

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import FrozenPourEnvironment
from .predicates import PourPredicates


class FrozenPourTasks(Tasks):
    env: FrozenPourEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    requested_counts: tuple[StrictInt, ...] | None = None
    _test_index: int = PrivateAttr(default=0)
    _train: np.random.Generator = PrivateAttr()
    _test: np.random.Generator = PrivateAttr()
    _reset: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def validate_schedule(self) -> "FrozenPourTasks":
        if self.requested_counts is None:
            self.requested_counts = (self.env.requested_count,)
        counts = self.requested_counts
        if (
            not counts
            or len(set(counts)) != len(counts)
            or any(not 1 <= k <= self.env.payload_count for k in counts)
        ):
            raise ValueError("Requested counts must be unique, nonempty, and within payload count")
        return self

    def next_test_request(self) -> int:
        assert self.requested_counts is not None
        requested = self.requested_counts[self._test_index % len(self.requested_counts)]
        self._test_index += 1
        return requested

    def model_post_init(self, __context: object) -> None:
        self._train = np.random.default_rng(self.seed)
        self._test = np.random.default_rng(self.seed + self.test_env_seed_offset)
        self._reset = np.random.default_rng(self.seed + 20000)

    def goal(self, *, requested: int | None = None) -> Goal:
        requested = self.env.requested_count if requested is None else requested
        return Goal(
            atoms=frozenset({
                GroundAtom(
                    predicate=PourPredicates.get(name="Goal"),
                    objects=(self.env.objects().requests()[requested - 1],),
                )
            })
        )

    def build_task(self, *, seed: int, requested: int | None = None) -> Task:
        requested = self.env.requested_count if requested is None else requested
        state = self.env.reset_to_seed(seed=seed, requested=requested)
        return Task(initial_state=state, goal=self.goal())

    def sample_train_task(self) -> Task:
        return self.build_task(seed=int(self._train.integers(0, 2**31 - 1)))

    def sample_test_task(self) -> Task:
        return self.build_task(
            seed=int(self._test.integers(0, 2**31 - 1)),
            requested=self.next_test_request(),
        )

    def sample_reset_start_task(self) -> Task:
        return self.build_task(seed=int(self._reset.integers(0, 2**31 - 1)))

    def sample_train_task_in_place(self) -> Task:
        return Task(
            initial_state=self.env.get_current_state(),
            goal=self.goal(),
        )
