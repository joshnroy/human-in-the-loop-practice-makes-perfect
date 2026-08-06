from enum import Enum

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import TossingRoomSplitEnvironment
from .predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)


class TossingRoomSplitGoalType(str, Enum):
    """The three goal families this domain supports. A str-Enum so the CLI's
    --goal-type choices map straight onto it. Kept here (not a bucket types.py) since
    only Tasks generates over it -- the same call it lives inside of."""

    RECYCLING = "recycling"  # throw the recycling item into the recycling bin
    TRASH = "trash"  # throw the trash item into the trash bin
    # Empty both prefilled bins, each via its OWN button beside it. Since the recycling
    # button sits behind the one-way ledge, this is an ordering task: trash first, then
    # drop across for recycling. The reverse order is unsolvable.
    EMPTY = "empty"


class TossingRoomSplitTasks(Tasks):
    """Task/goal generation for Tossing Room (split throws) -- `TossingRoomTasks`
    unchanged except that a throw goal now names the per-kind in-bin predicate
    (`TrashInBin`/`RecyclingInBin`) instead of the shared `ItemInBin`, and the empty
    goal names `TrashBinEmpty`/`RecyclingBinEmpty` instead of the shared `BinEmpty`.
    Everything about the *distribution* is deliberately identical, so a result here is
    comparable with Tossing Room's.

    Each task draws the two CAUSES of the throw force -- a per-bin `throw_distance`
    (Uniform[distance_low, distance_high)) and a per-item `weight`
    (Uniform[weight_low, weight_high)) -- and never the force itself, which
    `TossingRoomSplitEnvironment.required_force` derives from them with coefficients the
    agent cannot see. Those are the continuous per-episode values the two throw samplers
    specialize on; before this they drew a per-item `target_force` that sat in each
    sampler's own input row (see the environment's class docstring). Identical to
    `TossingRoomTasks` on purpose -- the split-throw experiment compares against Tossing
    Room's baseline, so the task distribution has to be the same one. The two streams
    differ in RNG seed (predicators' test_env_seed_offset convention) *and* in how each
    task's goal family is chosen.

    **Train stream: sampled** from `goal_weights`.

    **Test stream: a fixed composition, not a sampled one.** `sample_test_task` draws
    from a deterministic schedule of exactly `test_goal_type_counts()` families,
    shuffled with `test_rng`. Two reasons, both about the per-family analyses this
    domain feeds:
      * EMPTY carries no throw and so no stochasticity, so it is worth keeping as a live
        sanity check but not worth ~20% of the evaluation budget. (It is no longer a
        *trivial* walk-and-press: with one button per bin it now requires both presses in
        the one order the ledge permits, so it tests ordering rather than nothing.)
      * Sampling made the per-family *counts* random too, adding a variance source on
        top of the binomial noise already in each family's success rate.
    At the 30 test tasks this domain's experiments use, the composition is 14 TRASH /
    14 RECYCLING / 2 EMPTY, for every seed.

    `forced_goal_type` (the CLI's --goal-type) pins every sampled task to a single
    family and takes precedence over both rules, which is what makes the demo clip a
    throw task on any seed and lets each oracle branch be tested without sampling until
    lucky.

    env must be the same TossingRoomSplitEnvironment instance the surrounding Problem
    drives: build_task calls self.env.build_initial_state, which needs that instance's
    layout (bin/button rooms, start room) to place everything correctly."""

    env: TossingRoomSplitEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    # The two per-task cause ranges, identical to TossingRoomTasks'. Together with the
    # environment's relation they make the required force span exactly [0.1, 0.9], so
    # every winning window sits wholly inside the U(0, 1) band `sample_params` draws
    # from: every task reachable, none clipped, and a uniformly random force landing with
    # probability exactly 0.2 on every task.
    distance_low: float = 1.0
    distance_high: float = 3.0
    weight_low: float = 0.5
    weight_high: float = 1.5
    # Sampling weights over (RECYCLING, TRASH, EMPTY) for the *training* stream only.
    goal_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    forced_goal_type: TossingRoomSplitGoalType | None = None
    # Size of the fixed test set, i.e. how many test tasks the harness will draw before
    # reusing them. Wired from the global --num-test-tasks flag; it is what
    # test_goal_type_counts divides up, and it has to move in step with
    # PracticeLoop.run's own num_test_tasks or the realised composition silently differs.
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
    _test_schedule: list[TossingRoomSplitGoalType] = PrivateAttr()

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
        reseeding. The test schedule is rebuilt here too: it is derived from test_rng,
        so leaving a half-consumed one behind would make a reseeded instance disagree
        with a freshly constructed one."""
        self.seed = seed
        self._train_rng = self._make_rng(offset=0)
        self._test_rng = self._make_rng(offset=self.test_env_seed_offset)
        self._test_schedule = self._build_test_schedule()

    def _make_rng(self, *, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def test_goal_type_counts(self) -> dict[TossingRoomSplitGoalType, int]:
        """The exact goal-family composition of this instance's fixed test set --
        public so an analysis script can assert the composition of a run it is reading
        back (nothing here depends on the layout or the seed).

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
            TossingRoomSplitGoalType.TRASH: remaining - num_recycling,
            TossingRoomSplitGoalType.RECYCLING: num_recycling,
            TossingRoomSplitGoalType.EMPTY: num_empty,
        }

    def sample_train_task(self) -> Task:
        return self.build_task(goal_type=self._sample_goal_type(), rng=self.train_rng)

    def sample_test_task(self) -> Task:
        return self.build_task(goal_type=self._next_test_goal_type(), rng=self.test_rng)

    def _sample_goal_type(self) -> TossingRoomSplitGoalType:
        """The *training* stream's goal family: an independent draw from goal_weights."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        types = (
            TossingRoomSplitGoalType.RECYCLING,
            TossingRoomSplitGoalType.TRASH,
            TossingRoomSplitGoalType.EMPTY,
        )
        return types[int(self.train_rng.choice(len(types), p=list(self.goal_weights)))]

    def _next_test_goal_type(self) -> TossingRoomSplitGoalType:
        """The *test* stream's goal family: the next entry of the fixed schedule (so
        the realised composition is exactly test_goal_type_counts(), every seed), or
        forced_goal_type when one is pinned -- which leaves the schedule untouched
        rather than consuming it."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        if not self._test_schedule:
            self._test_schedule = self._build_test_schedule()
        return self._test_schedule.pop(0)

    def _build_test_schedule(self) -> list[TossingRoomSplitGoalType]:
        """One block of test_goal_type_counts() goal families in a test_rng-determined
        order. Shuffled rather than grouped by family: some analyses key on a task's
        index within the sweep, which a family-sorted test set would make misleading."""
        block = [
            goal_type
            for goal_type, count in self.test_goal_type_counts().items()
            for _ in range(count)
        ]
        return [block[int(index)] for index in self._test_rng.permutation(len(block))]

    def build_task(self, *, goal_type: TossingRoomSplitGoalType, rng: np.random.Generator) -> Task:
        # Always draw all four per-task causes (regardless of goal type) so the
        # continuous per-task values stay comparable across streams/seeds. Order is fixed
        # -- weights then distances -- because a run is fully determined by --seed, and it
        # matches TossingRoomTasks so the two domains draw the same tasks at a given seed.
        trash_weight = float(rng.uniform(self.weight_low, self.weight_high))
        recycling_weight = float(rng.uniform(self.weight_low, self.weight_high))
        trash_bin_distance = float(rng.uniform(self.distance_low, self.distance_high))
        recycling_bin_distance = float(rng.uniform(self.distance_low, self.distance_high))

        if goal_type is TossingRoomSplitGoalType.EMPTY:
            # One item per bin: a bin holds at most one, so the old Uniform{1, 2, 3}
            # prefill (and its initial_count_low/high knobs) has a single legal value
            # left. Both bins are filled, so the goal is never already or half satisfied
            # -- and since each bin now has its own button, emptying both is what makes
            # this an ordering task rather than one walk and one press.
            initial_state = self.env.build_initial_state(
                trash_weight=trash_weight,
                recycling_weight=recycling_weight,
                trash_bin_distance=trash_bin_distance,
                recycling_bin_distance=recycling_bin_distance,
                recycling_count=self.env.BIN_CAPACITY,
                trash_count=self.env.BIN_CAPACITY,
            )
            atoms = frozenset({
                RECYCLING_BIN_EMPTY(state=initial_state, objects=(self.env.recycling_bin,)),
                TRASH_BIN_EMPTY(state=initial_state, objects=(self.env.trash_bin,)),
            })
            return Task(initial_state=initial_state, goal=Goal(atoms=atoms))

        # A throw goal: bins start empty; the goal is the single in-bin atom for that
        # kind (TrashInBin or RecyclingInBin -- split, unlike Tossing Room's shared
        # ItemInBin, because the item and bin types are split).
        initial_state = self.env.build_initial_state(
            trash_weight=trash_weight,
            recycling_weight=recycling_weight,
            trash_bin_distance=trash_bin_distance,
            recycling_bin_distance=recycling_bin_distance,
        )
        if goal_type is TossingRoomSplitGoalType.RECYCLING:
            atom: GroundAtom = RECYCLING_IN_BIN(
                state=initial_state, objects=(self.env.recycling, self.env.recycling_bin)
            )
        else:
            atom = TRASH_IN_BIN(state=initial_state, objects=(self.env.trash, self.env.trash_bin))
        return Task(initial_state=initial_state, goal=Goal(atoms=frozenset({atom})))
