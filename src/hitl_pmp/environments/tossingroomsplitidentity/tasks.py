from enum import Enum

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Task

from .environment import TossingRoomSplitIdentityEnvironment
from .predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)


class TossingRoomSplitIdentityGoalType(str, Enum):
    """The three goal families this domain supports. A str-Enum so the CLI's
    --goal-type choices map straight onto it. Kept here (not a bucket types.py) since
    only Tasks generates over it -- the same call it lives inside of."""

    RECYCLING = "recycling"  # throw the recycling item into the recycling bin
    TRASH = "trash"  # throw the trash item into the trash bin
    # Empty both prefilled bins, each via its OWN button beside it. Since the recycling
    # button sits behind the one-way ledge, this is an ordering task: trash first, then
    # drop across for recycling. The reverse order is unsolvable.
    EMPTY = "empty"


class TossingRoomSplitIdentityTasks(Tasks):
    """Task/goal generation for Tossing Room (split throws, identity representation) --
    `TossingRoomSplitTasks` with **one** change, and it is not in what each task *is*.

    **THIS CLASS DRAWS EXACTLY THE TASKS `TossingRoomSplitTasks` DRAWS.** Same two causes
    per throw -- a per-bin `throw_distance` (Uniform[distance_low, distance_high)) and a
    per-item `weight` (Uniform[weight_low, weight_high)) -- drawn in the same order, from
    the same ranges, and combined by the same affine relation with the same five
    constants. So at any given seed the two arms present the identical sequence of goal
    families with the identical required force for every throw, task for task.

    **The only difference is what the State exposes about that task.** The causal arm
    puts the two causes in the State and keeps the relation's constants out of it, so a
    sampler must learn a relation. This arm puts the *result* in the State as
    `item.target_force` and drops the causes, so the answer is a literal column -- index
    4 of each throw sampler's own classifier row, making the optimal policy
    `force* = x_4`. That is the **degenerate identity representation** this domain exists
    to restore; see the environment's class docstring for the full row layout.

    **Why the relation lives here rather than being replaced by a direct draw.** Drawing
    `target_force` from a plain Uniform would have matched the causal arm on a random
    force (0.20 per task) but NOT on its marginal: the causal arm's required force is a
    sum of two uniforms and so is *triangular* on [0.1, 0.9], concentrating mass near
    0.5. Measured over 400 groundings per family, the best single FIXED force lands
    119/400 under a Uniform[0.1, 0.9) target against 185/400 in the causal arm -- so a
    state-blind sampler would have scored far better in one arm than the other, and a
    cross-arm reading of "how much did conditioning on the state buy" would have been
    confounded by that alone. Drawing the causes and combining them reproduces the causal
    arm's marginal exactly, which is the only way "exactly one delta" is true of the
    *distribution* as well as of the schema.

    A consequence worth having: because the two arms consume their RNG identically, a
    given seed yields the same practice tasks and the same test tasks in both, so the
    arms are paired rather than merely comparable.
    `tests/environments/tossingroomsplitidentity/test_fork_equivalence.py` asserts that
    task-for-task agreement rather than leaving it to intent.

    The two streams differ in RNG seed (predicators' test_env_seed_offset convention)
    *and* in how each task's goal family is chosen.

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

    env must be the same TossingRoomSplitIdentityEnvironment instance the surrounding Problem
    drives: build_task calls self.env.build_initial_state, which needs that instance's
    layout (bin/button rooms, start room) to place everything correctly."""

    env: TossingRoomSplitIdentityEnvironment
    seed: int = 0
    test_env_seed_offset: int = 10000
    # The two per-task cause ranges, IDENTICAL to TossingRoomSplitTasks' -- see the class
    # docstring. Together with the relation below the required force spans exactly
    # [0.1, 0.9], so every winning window sits wholly inside the U(0, 1) band
    # `sample_params` draws from: every task reachable, none clipped, and a uniformly
    # random force landing with probability exactly 0.20 on every task, in both arms.
    #
    # The pre-#80 design instead drew `target_force` directly from U[0.5, 1.0). That does
    # NOT have the property: targets above 0.9 have their window truncated by the top of
    # the force band, so it lands with probability 0.20 on only 8/10 of its range,
    # falling to 0.10 at target 1.0 and averaging 0.19 (PR #80's body records the
    # measured 16/80). Restoring that range verbatim would have made this arm harder
    # under a random draw than the causal arm, confounding "representation" with
    # "difficulty" in exactly the comparison this domain exists to support.
    distance_low: float = 1.0
    distance_high: float = 3.0
    weight_low: float = 0.5
    weight_high: float = 1.5
    # The relation that turns the two drawn causes into the target this arm then puts IN
    # the State. Same five constants as the causal arm's environment, because matching its
    # marginal is the whole point (class docstring). They live on Tasks rather than on the
    # Environment because here they are purely task-DISTRIBUTION configuration: the
    # dynamics never consult them -- `Environment.required_force` is the identity on
    # `item.target_force` -- so putting them on the environment would imply the agent has
    # something left to infer, which under this representation it does not.
    reference_force: float = 0.5
    reference_distance: float = 2.0
    reference_weight: float = 1.0
    distance_coefficient: float = 0.2
    weight_coefficient: float = 0.4
    # Sampling weights over (RECYCLING, TRASH, EMPTY) for the *training* stream only.
    goal_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    forced_goal_type: TossingRoomSplitIdentityGoalType | None = None
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
    _test_schedule: list[TossingRoomSplitIdentityGoalType] = PrivateAttr()

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

    def test_goal_type_counts(self) -> dict[TossingRoomSplitIdentityGoalType, int]:
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
            TossingRoomSplitIdentityGoalType.TRASH: remaining - num_recycling,
            TossingRoomSplitIdentityGoalType.RECYCLING: num_recycling,
            TossingRoomSplitIdentityGoalType.EMPTY: num_empty,
        }

    def sample_train_task(self) -> Task:
        return self.build_task(goal_type=self._sample_goal_type(), rng=self.train_rng)

    def sample_test_task(self) -> Task:
        return self.build_task(goal_type=self._next_test_goal_type(), rng=self.test_rng)

    def _sample_goal_type(self) -> TossingRoomSplitIdentityGoalType:
        """The *training* stream's goal family: an independent draw from goal_weights."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        types = (
            TossingRoomSplitIdentityGoalType.RECYCLING,
            TossingRoomSplitIdentityGoalType.TRASH,
            TossingRoomSplitIdentityGoalType.EMPTY,
        )
        return types[int(self.train_rng.choice(len(types), p=list(self.goal_weights)))]

    def _next_test_goal_type(self) -> TossingRoomSplitIdentityGoalType:
        """The *test* stream's goal family: the next entry of the fixed schedule (so
        the realised composition is exactly test_goal_type_counts(), every seed), or
        forced_goal_type when one is pinned -- which leaves the schedule untouched
        rather than consuming it."""
        if self.forced_goal_type is not None:
            return self.forced_goal_type
        if not self._test_schedule:
            self._test_schedule = self._build_test_schedule()
        return self._test_schedule.pop(0)

    def _build_test_schedule(self) -> list[TossingRoomSplitIdentityGoalType]:
        """One block of test_goal_type_counts() goal families in a test_rng-determined
        order. Shuffled rather than grouped by family: some analyses key on a task's
        index within the sweep, which a family-sorted test set would make misleading."""
        block = [
            goal_type
            for goal_type, count in self.test_goal_type_counts().items()
            for _ in range(count)
        ]
        return [block[int(index)] for index in self._test_rng.permutation(len(block))]

    def target_force(self, *, throw_distance: float, item_weight: float) -> float:
        """The target force a throw of `item_weight` into a bin `throw_distance` away
        gets, written exactly as `TossingRoomSplitEnvironment.required_force` writes it so
        the two arms' task distributions are the same distribution rather than two
        approximations of one.

        The difference between the arms is *where this value ends up*, not what it is: in
        the causal arm it stays out of the State and the two causes go in, so it must be
        inferred; here it goes into the State as `item.target_force` and the causes are
        discarded, so it is read. `Environment.required_force` is then the identity on it.
        """
        return (
            self.reference_force
            + self.distance_coefficient * (throw_distance - self.reference_distance)
            + self.weight_coefficient * (item_weight - self.reference_weight)
        )

    def build_task(
        self, *, goal_type: TossingRoomSplitIdentityGoalType, rng: np.random.Generator
    ) -> Task:
        # Always draw all four per-task causes (regardless of goal type) so the continuous
        # per-task values stay comparable across streams/seeds. Order is fixed -- weights
        # then distances -- because a run is fully determined by --seed, and it matches
        # `TossingRoomSplitTasks` EXACTLY: same ranges, same order, same number of draws,
        # so at a given seed the two arms consume their RNG in lockstep and present the
        # identical task with the identical required force. See the class docstring.
        trash_weight = float(rng.uniform(self.weight_low, self.weight_high))
        recycling_weight = float(rng.uniform(self.weight_low, self.weight_high))
        trash_bin_distance = float(rng.uniform(self.distance_low, self.distance_high))
        recycling_bin_distance = float(rng.uniform(self.distance_low, self.distance_high))
        # ...and then the ONE thing this arm does differently: it resolves the two causes
        # into the answer here, at task-construction time, so that the State can carry the
        # answer itself instead of the causes.
        trash_target = self.target_force(
            throw_distance=trash_bin_distance, item_weight=trash_weight
        )
        recycling_target = self.target_force(
            throw_distance=recycling_bin_distance, item_weight=recycling_weight
        )

        if goal_type is TossingRoomSplitIdentityGoalType.EMPTY:
            # One item per bin: a bin holds at most one, so the old Uniform{1, 2, 3}
            # prefill (and its initial_count_low/high knobs) has a single legal value
            # left. Both bins are filled, so the goal is never already or half satisfied
            # -- and since each bin now has its own button, emptying both is what makes
            # this an ordering task rather than one walk and one press.
            initial_state = self.env.build_initial_state(
                trash_target_force=trash_target,
                recycling_target_force=recycling_target,
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
            trash_target_force=trash_target,
            recycling_target_force=recycling_target,
        )
        if goal_type is TossingRoomSplitIdentityGoalType.RECYCLING:
            atom: GroundAtom = RECYCLING_IN_BIN(
                state=initial_state, objects=(self.env.recycling, self.env.recycling_bin)
            )
        else:
            atom = TRASH_IN_BIN(state=initial_state, objects=(self.env.trash, self.env.trash_bin))
        return Task(initial_state=initial_state, goal=Goal(atoms=frozenset({atom})))
