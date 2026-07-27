from enum import Enum

import numpy as np
from pydantic import PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import TossingRoomEnvironment
from .predicates import BIN_EMPTY, ITEM_IN_BIN


class TossingRoomGoalType(str, Enum):
    """The three goal families this domain supports. A str-Enum so the CLI's
    --goal-type choices map straight onto it. Kept here (not a bucket types.py) since
    only Tasks generates over it -- the same call it lives inside of."""

    RECYCLING = "recycling"  # throw the recycling item into the recycling bin
    TRASH = "trash"  # throw the trash item into the trash bin
    EMPTY = "empty"  # empty both (initially non-empty) bins via the button


class TossingRoomTasks(Tasks):
    """Task/goal generation for Tossing Room. Each task draws per-item throw targets
    (Uniform[target_low, target_high)) -- the continuous per-episode value a future
    learner could specialize the throw skill on, the way Light Switch's light target
    varies. Train and test share the sampling distribution, differing only in RNG
    stream (predicators' test_env_seed_offset convention, mirrored from Light Switch).

    A goal type is normally sampled from goal_weights, but build_task takes it
    explicitly, so a caller can pin one deterministically -- forced_goal_type (the
    CLI's --goal-type) fixes every sampled task to a single family, which is what
    makes the demo GIF a throw task on any seed and lets each oracle branch be tested
    without sampling until lucky.

    env must be the same TossingRoomEnvironment instance the surrounding Problem
    drives: build_task calls self.env.build_initial_state, which needs that instance's
    layout (bin/button rooms, start room) to place everything correctly."""

    env: TossingRoomEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    target_low: float = 0.5
    target_high: float = 1.0
    # Bins start with this many items for the EMPTY goal (both bins non-empty, so the
    # goal is never already/half satisfied). Inclusive integer range.
    initial_count_low: int = 1
    initial_count_high: int = 3
    # Default sampling weights over (RECYCLING, TRASH, EMPTY): biased toward the throw
    # families, since they are the interesting pick-traverse-throw tasks.
    goal_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    forced_goal_type: TossingRoomGoalType | None = None

    _train_rng: np.random.Generator = PrivateAttr()
    _test_rng: np.random.Generator = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self.set_seed(seed=self.seed)

    @property
    def train_rng(self) -> np.random.Generator:
        return self._train_rng

    @property
    def test_rng(self) -> np.random.Generator:
        return self._test_rng

    def set_seed(self, *, seed: int) -> None:
        """Reset seed and rederive both RNG streams together (train_rng from seed,
        test_rng from seed + test_env_seed_offset) -- the single entry point for
        reseeding, mirroring Light Switch's Tasks."""
        self.seed = seed
        self._train_rng = self._make_rng(offset=0)
        self._test_rng = self._make_rng(offset=self.test_env_seed_offset)

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def sample_train_task(self) -> Task:
        return self._sample_task(rng=self.train_rng)

    def sample_test_task(self) -> Task:
        return self._sample_task(rng=self.test_rng)

    def _sample_task(self, *, rng: np.random.Generator) -> Task:
        goal_type = self._sample_goal_type(rng=rng)
        return self.build_task(goal_type=goal_type, rng=rng)

    def _sample_goal_type(self, *, rng: np.random.Generator) -> TossingRoomGoalType:
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        types = (
            TossingRoomGoalType.RECYCLING,
            TossingRoomGoalType.TRASH,
            TossingRoomGoalType.EMPTY,
        )
        return types[int(rng.choice(len(types), p=list(self.goal_weights)))]

    def build_task(self, *, goal_type: TossingRoomGoalType, rng: np.random.Generator) -> Task:
        # Always draw both per-item targets (regardless of goal type) so the
        # continuous per-task value stays comparable across streams/seeds.
        trash_target = float(rng.uniform(self.target_low, self.target_high))
        recycling_target = float(rng.uniform(self.target_low, self.target_high))

        if goal_type is TossingRoomGoalType.EMPTY:
            recycling_count = int(
                rng.integers(self.initial_count_low, self.initial_count_high, endpoint=True)
            )
            trash_count = int(
                rng.integers(self.initial_count_low, self.initial_count_high, endpoint=True)
            )
            initial_state = self.env.build_initial_state(
                trash_target_force=trash_target,
                recycling_target_force=recycling_target,
                recycling_count=recycling_count,
                trash_count=trash_count,
            )
            atoms = frozenset({
                BIN_EMPTY(state=initial_state, objects=(self.env.recycling_bin,)),
                BIN_EMPTY(state=initial_state, objects=(self.env.trash_bin,)),
            })
            return Task(initial_state=initial_state, goal=Goal(atoms=atoms))

        # A throw goal: bins start empty; the goal is the single ItemInBin atom.
        initial_state = self.env.build_initial_state(
            trash_target_force=trash_target, recycling_target_force=recycling_target
        )
        if goal_type is TossingRoomGoalType.RECYCLING:
            atom: GroundAtom = ITEM_IN_BIN(
                state=initial_state, objects=(self.env.recycling, self.env.recycling_bin)
            )
        else:
            atom = ITEM_IN_BIN(state=initial_state, objects=(self.env.trash, self.env.trash_bin))
        return Task(initial_state=initial_state, goal=Goal(atoms=frozenset({atom})))
