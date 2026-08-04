from enum import Enum

import numpy as np
from pydantic import Field, PrivateAttr

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
    varies. The two streams differ in RNG seed (predicators' test_env_seed_offset
    convention, mirrored from Light Switch) *and*, since the fixed-composition change
    below, in how each task's goal family is chosen.

    **Train stream: sampled.** sample_train_task draws its goal family from
    goal_weights, exactly as before -- the training task distribution is deliberately
    untouched by the test-set change.

    **Test stream: a fixed composition, not a sampled one.** sample_test_task draws
    from a deterministic schedule of exactly test_goal_type_counts() families,
    shuffled with test_rng, rather than sampling each family independently. Two
    reasons, both about the analyses this domain feeds (which are per goal family,
    with the within-seed TRASH - RECYCLING success gap as the headline):
      * EMPTY is deterministic (MoveRoom x k + Press, no Throw) -- it was 100% in
        every seed of every arm measured so far, sd exactly 0. It is worth keeping as
        a live sanity check but not worth ~20% of the evaluation budget, so it gets a
        small fixed allocation and the throw families get the rest.
      * Sampling made the per-family *counts* random too (16/10/4 at seed 0 vs
        11/12/7 at seed 1 for 30 test tasks), adding a variance source on top of the
        binomial noise already in each family's success rate. A fixed composition
        removes it at zero extra compute.
    At the 30 test tasks this domain's experiments use, the composition is 14 TRASH /
    14 RECYCLING / 2 EMPTY, for every seed. **Consequence: every Tossing Room number
    produced after this change is measured on a different evaluation set than every
    number produced before it** -- results are not comparable across that boundary
    (this includes the archived stats.json files under docs/handoff/raw-results/).

    A goal type is normally chosen by whichever of those two rules applies, but
    build_task takes it explicitly, so a caller can pin one deterministically --
    forced_goal_type (the CLI's --goal-type) fixes every sampled task to a single
    family and takes precedence over both rules (train sampling and the test
    schedule alike), which is what makes the demo GIF a throw task on any seed and
    lets each oracle branch be tested without sampling until lucky.

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
    # Sampling weights over (RECYCLING, TRASH, EMPTY) for the *training* stream only:
    # biased toward the throw families, since they are the interesting
    # pick-traverse-throw tasks. The test stream ignores these -- see the class
    # docstring and test_goal_type_counts.
    goal_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    forced_goal_type: TossingRoomGoalType | None = None
    # Size of the fixed test set, i.e. how many test tasks the harness will draw
    # before reusing them. Wired from the global --num-test-tasks flag (whose own
    # default this mirrors); it is what test_goal_type_counts divides up. This has to
    # move in step with PracticeLoop.run's own num_test_tasks: the two are separate
    # values, and a harness that draws more than this says just starts another block
    # (30 draws against a field of 10 realises 12/12/6, not 14/14/2), silently.
    num_test_tasks: int = Field(default=10, ge=1)
    # EMPTY's allocation in that fixed composition: at most max_empty_test_tasks, and
    # never more than one per min_test_tasks_per_empty test tasks, so a small test set
    # can't be crowded out by the one family that carries no information.
    max_empty_test_tasks: int = 2
    min_test_tasks_per_empty: int = 5

    _train_rng: np.random.Generator = PrivateAttr()
    _test_rng: np.random.Generator = PrivateAttr()
    # The remaining goal families of the current test block, in draw order. Rebuilt
    # (reshuffled) whenever it empties, so drawing more than num_test_tasks test tasks
    # stays well defined and keeps the same composition block by block.
    _test_schedule: list[TossingRoomGoalType] = PrivateAttr()

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
        reseeding, mirroring Light Switch's Tasks. The test schedule is rebuilt here
        too: it is derived from test_rng, so leaving a half-consumed one behind would
        make a reseeded instance disagree with a freshly constructed one."""
        self.seed = seed
        self._train_rng = self._make_rng(offset=0)
        self._test_rng = self._make_rng(offset=self.test_env_seed_offset)
        self._test_schedule = self._build_test_schedule()

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def test_goal_type_counts(self) -> dict[TossingRoomGoalType, int]:
        """The exact goal-family composition of this instance's fixed test set --
        public so an analysis script can assert the composition of a run it is reading
        back (TossingRoomTasks(env=TossingRoomEnvironment(), num_test_tasks=30) is
        enough; nothing here depends on the layout or the seed).

        The rule: EMPTY gets min(max_empty_test_tasks, num_test_tasks //
        min_test_tasks_per_empty) tasks -- 2 at both 30 and 10 test tasks, 0 below 5 --
        and the remainder splits as evenly as possible between the two throw families,
        with an odd leftover task going to TRASH (an arbitrary but fixed tie-break, so
        the composition is a function of num_test_tasks alone). At 30: 14 TRASH / 14
        RECYCLING / 2 EMPTY."""
        num_empty = min(
            self.max_empty_test_tasks, self.num_test_tasks // self.min_test_tasks_per_empty
        )
        remaining = self.num_test_tasks - num_empty
        num_recycling = remaining // 2
        return {
            TossingRoomGoalType.TRASH: remaining - num_recycling,
            TossingRoomGoalType.RECYCLING: num_recycling,
            TossingRoomGoalType.EMPTY: num_empty,
        }

    def sample_train_task(self) -> Task:
        return self.build_task(goal_type=self._sample_goal_type(), rng=self.train_rng)

    def sample_test_task(self) -> Task:
        return self.build_task(goal_type=self._next_test_goal_type(), rng=self.test_rng)

    def _sample_goal_type(self) -> TossingRoomGoalType:
        """The *training* stream's goal family: an independent draw from goal_weights."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        types = (
            TossingRoomGoalType.RECYCLING,
            TossingRoomGoalType.TRASH,
            TossingRoomGoalType.EMPTY,
        )
        return types[int(self.train_rng.choice(len(types), p=list(self.goal_weights)))]

    def _next_test_goal_type(self) -> TossingRoomGoalType:
        """The *test* stream's goal family: the next entry of the fixed schedule (so
        the realised composition is exactly test_goal_type_counts(), every seed), or
        forced_goal_type when one is pinned -- which leaves the schedule untouched
        rather than consuming it."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        if not self._test_schedule:
            self._test_schedule = self._build_test_schedule()
        return self._test_schedule.pop(0)

    def _build_test_schedule(self) -> list[TossingRoomGoalType]:
        """One block of test_goal_type_counts() goal families in a test_rng-determined
        order. Shuffled rather than grouped by family: some analyses key on a task's
        index within the sweep, which a family-sorted test set would make misleading."""
        block = [
            goal_type
            for goal_type, count in self.test_goal_type_counts().items()
            for _ in range(count)
        ]
        return [block[int(index)] for index in self._test_rng.permutation(len(block))]

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
