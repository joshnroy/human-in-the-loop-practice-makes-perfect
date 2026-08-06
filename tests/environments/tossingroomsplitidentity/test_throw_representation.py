"""The properties each throw's representation has here, asserted rather than argued.

**This file is the deliberate INVERSE of its counterpart in the causal arm**
(`tests/environments/tossingroomsplit/test_throw_representation.py`). That one asserts
that the answer is *not* in the state, that two causes are both load-bearing, and that no
single column predicts the required force. Every one of those assertions is false by
construction here, and porting them across would have been a test suite that fails on
correct code. What is asserted instead is what this arm actually claims:

1. **The answer IS in the state**, at input index 4, exactly. `required_force` is the
   identity on `item.target_force`, so the optimal policy is the literal transformation
   `force* = x_4` -- copy one column -- and a single column predicts the required force
   with zero residual. That is the degenerate identity representation this domain exists
   to restore, not a defect to be caught.
2. **Nothing else in the row carries any signal.** `EesMethod.sampler` keys classifiers by
   `skill_name`, so `ThrowTrash` and `ThrowRecycling` each see only their own skill's
   rows, and within one skill the bound robot, bin and room never vary. Every other state
   column is therefore a flat constant: 9 constant / 0 redundant / 2 free out of 11.
3. **It varies per task**, so a sampler still cannot memorise one force.
4. **Difficulty is matched to the causal arm exactly, on every axis**, because the two
   arms draw the SAME TASKS. `TossingRoomSplitIdentityTasks` draws the causal arm's two
   causes, from the same ranges in the same order, and resolves them with the same five
   constants -- then puts the result in the State as `target_force` and discards the
   causes. So a force from U(0, 1) lands with probability exactly 0.20 on every task in
   both arms, AND the marginal distribution of the required force is the same triangular
   distribution on [0.1, 0.9], so the best fixed a state-blind sampler could do is the
   same too. `test_fork_equivalence.py` asserts both, task for task.

   This is why `target_force` is NOT drawn from a plain Uniform. That would have matched
   the per-task probability but not the marginal -- a uniform target scores 120/400 for
   the best fixed force against the causal arm's 185/400 -- and a cross-arm reading of
   "what did conditioning on the state buy" would have been confounded by the difference.
"""

import numpy as np
import pytest

from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.tasks import (
    TossingRoomSplitIdentityGoalType,
    TossingRoomSplitIdentityTasks,
)

_TOLERANCE = TossingRoomSplitIdentityEnvironment.model_fields["throw_tolerance"].default

# The row layout, by role, in each throw's own parameter order (robot, item, bin, room).
# Identical for both throws by design -- see the environment's docstring on why the split
# types keep identical feature schemas. ELEVEN columns, one fewer than the causal arm's
# twelve: the bin drops `throw_distance`, which contributes nothing to the required force
# under this representation.
_COLUMN_NAMES = (
    "bias",
    "robot.room",
    "robot.holding",
    "item.kind",
    "item.target_force",  # <-- THE ANSWER, at index 4
    "bin.count",
    "bin.room",
    "bin.kind",
    "room.index",
    "room.blocks_right",
    "force",
)
_ANSWER_INDEX = _COLUMN_NAMES.index("item.target_force")
_FAMILIES = (
    TossingRoomSplitIdentityGoalType.TRASH,
    TossingRoomSplitIdentityGoalType.RECYCLING,
)


def _env() -> TossingRoomSplitIdentityEnvironment:
    return TossingRoomSplitIdentityEnvironment()


def _kind_for(
    *, env: TossingRoomSplitIdentityEnvironment, family: TossingRoomSplitIdentityGoalType
) -> int:
    return (
        env.TRASH_KIND if family is TossingRoomSplitIdentityGoalType.TRASH else env.RECYCLING_KIND
    )


def _applicable_throws(
    *,
    env: TossingRoomSplitIdentityEnvironment,
    family: TossingRoomSplitIdentityGoalType,
    num_tasks: int,
    seed: int,
):
    """Yield `(state, row, required)` for one applicable grounding of **one** throw skill
    per sampled task -- the state the throw would be taken in, that grounding's classifier
    input row, and the force it needs.

    "Applicable" means the preconditions hold: holding that kind, standing in its bin's
    room, that bin empty. Per skill, not pooled, because that is what each
    `LearnedSkillSampler` actually sees. Yielding the STATE alongside the row is what lets
    a test apply a force read out of the row to the very task the row came from, rather
    than to a state rebuilt out of that same number."""
    tasks = TossingRoomSplitIdentityTasks(env=env, seed=seed, num_test_tasks=30)
    force_rng = np.random.default_rng(seed)
    kind = _kind_for(env=env, family=family)
    bin_room = env.bin_room_for_kind(kind=kind)
    item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)
    for _ in range(num_tasks):
        task = tasks.build_task(goal_type=family, rng=tasks.train_rng)
        state = task.initial_state.model_copy(deep=True)
        state.set(obj=env.robot, feature_name="holding", feature_val=float(kind))
        state.set(obj=env.robot, feature_name="room", feature_val=float(bin_room))

        row = [1.0]
        for obj in (env.robot, item, bin_obj, env.get_rooms()[bin_room]):
            row.extend(
                float(state.get(obj=obj, feature_name=feature_name))
                for feature_name in obj.type.feature_names
            )
        # The dial the sampler picks, drawn the way the base sampler draws it.
        row.append(float(force_rng.uniform(0.0, 1.0)))
        required = env.required_force(
            item_target_force=float(state.get(obj=item, feature_name="target_force"))
        )
        yield state, row, required


def _throw_rows(
    *, family: TossingRoomSplitIdentityGoalType, num_tasks: int = 80, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """`(rows, required force per row)` over `num_tasks` applicable groundings."""
    triples = list(_applicable_throws(env=_env(), family=family, num_tasks=num_tasks, seed=seed))
    return (
        np.array([row for _state, row, _required in triples]),
        np.array([required for _state, _row, required in triples]),
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_the_row_layout_is_the_one_these_tests_assume(
    *, family: TossingRoomSplitIdentityGoalType
) -> None:
    """Guard: every column index below is named, so a feature added to any of a throw's
    four object types has to come here and be classified rather than silently joining the
    row. Checked on both throws, which also pins that the two rows are the same width --
    the property that makes the two samplers the same architecture."""
    rows, _required = _throw_rows(family=family, num_tasks=2)
    assert rows.shape[1] == len(_COLUMN_NAMES)


class TestTheAnswerIsInTheState:
    """Property 1, and the whole reason this arm exists. Every assertion in this class is
    the negation of one in the causal arm's file of the same name."""

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_column_four_equals_the_required_force_exactly(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        rows, required = _throw_rows(family=family)
        assert np.array_equal(rows[:, _ANSWER_INDEX], required)

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_a_single_column_predicts_the_required_force_with_zero_residual(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """The causal arm requires the best single-column affine fit's worst residual to
        EXCEED the tolerance. Here it is exactly zero, and by the widest possible margin:
        the fit is the identity map."""
        rows, required = _throw_rows(family=family)
        column = rows[:, _ANSWER_INDEX]
        slope, intercept = np.polyfit(column, required, 1)
        worst_residual = float(np.max(np.abs(slope * column + intercept - required)))
        assert worst_residual < 1e-12
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.0, abs=1e-12)

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_optimal_policy_is_the_literal_identity_on_column_four(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """`force* = x_4`, stated as behaviour rather than as arithmetic: a "sampler"
        that does nothing but copy input index 4 into the force lands EVERY throw.

        The force is read out of the row and then applied to the very task that row was
        built from, which is what makes this an assertion about index 4 specifically
        rather than a restatement that `required_force` agrees with itself -- reading any
        other index yields a different number and the throw misses."""
        env = _env()
        kind = _kind_for(env=env, family=family)
        bin_obj = env.bin_for_kind(kind=kind)
        num_tasks = 40
        landed = 0
        for state, row, _required in _applicable_throws(
            env=env, family=family, num_tasks=num_tasks, seed=3
        ):
            env.set_state(state=state)
            after = env.take_action(
                action=np.array([float(env.SKILL_THROW), float(kind), float(row[_ANSWER_INDEX])])
            )
            landed += int(after.get(obj=bin_obj, feature_name="count") == 1.0)
        assert landed == num_tasks

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_no_other_state_column_carries_any_signal_at_all(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """The other half of "the answer is a column": nothing else in the row varies.

        Within one throw skill the robot, bin and room bindings never change, so every
        remaining state column is a flat constant and could be deleted without loss. That
        makes the learning problem exactly "compare two of your own inputs" -- which is
        what the causal arm replaced, and what this arm restores."""
        rows, _required = _throw_rows(family=family)
        varying = [name for index, name in enumerate(_COLUMN_NAMES) if np.ptp(rows[:, index]) != 0]
        assert varying == ["item.target_force", "force"]


class TestRequiredForceIsTheIdentity:
    """Property 2 in this arm: `required_force` is not a relation to be learned, it is the
    identity function on an observed feature -- and it really is the dynamics, not a
    parallel description of them."""

    @staticmethod
    @pytest.mark.parametrize("target", [0.1, 0.25, 0.5, 0.749, 0.9])
    def test_required_force_is_the_identity_function(*, target: float) -> None:
        assert _env().required_force(item_target_force=target) == target

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_throwing_at_the_required_force_lands_and_just_outside_it_misses(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        env = _env()
        tasks = TossingRoomSplitIdentityTasks(env=env, seed=1, num_test_tasks=30)
        kind = _kind_for(env=env, family=family)
        item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)
        bin_room = env.bin_room_for_kind(kind=kind)
        for _ in range(10):
            task = tasks.build_task(goal_type=family, rng=tasks.train_rng)
            required = env.required_force(
                item_target_force=float(
                    task.initial_state.get(obj=item, feature_name="target_force")
                )
            )
            for force, should_land in (
                (required, True),
                (required + _TOLERANCE + 1e-6, False),
                (required - _TOLERANCE - 1e-6, False),
            ):
                state = task.initial_state.model_copy(deep=True)
                state.set(obj=env.robot, feature_name="room", feature_val=float(bin_room))
                state.set(obj=env.robot, feature_name="holding", feature_val=float(kind))
                env.set_state(state=state)
                after = env.take_action(
                    action=np.array([float(env.SKILL_THROW), float(kind), float(force)])
                )
                landed = bool(after.get(obj=bin_obj, feature_name="count") == 1.0)
                assert landed is should_land

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_every_sampled_task_is_reachable_by_a_force_the_base_sampler_can_draw(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """`sample_params` draws Uniform(0, 1) for BOTH throws -- deliberately the same
        prior as the causal arm's. If a task's required force sat within a tolerance of 0
        or 1 its winning window would be clipped and the task would be quietly harder,
        which is exactly the [0.5, 1.0) defect the [0.1, 0.9) draw range fixes."""
        _rows, required = _throw_rows(family=family, num_tasks=200, seed=2)
        assert float(np.min(required)) > _TOLERANCE
        assert float(np.max(required)) < 1.0 - _TOLERANCE

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_a_uniformly_random_force_lands_about_a_fifth_of_the_time(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """The matched-difficulty claim, measured end to end on the same rows a sampler
        would see rather than derived. Analytically the rate is exactly 0.20 on EVERY task
        (window width 2 x 0.1, never clipped); at 400 draws that is 80/400 in expectation
        with a standard deviation of 8, so the band below is +-4 sd.

        Measured at seed 0: 71/400 (TRASH) and 76/400 (RECYCLING). The causal arm on the
        identical measurement gives 76/400 and 70/400 -- matched, which is the point."""
        rows, required = _throw_rows(family=family, num_tasks=400)
        landed = int(np.sum(np.abs(rows[:, -1] - required) < _TOLERANCE))
        assert 48 <= landed <= 112, f"{family.value}: a random force landed {landed}/400"


class TestItVariesPerTask:
    """Property 3. A constant requirement would be memorised in one throw -- and in this
    domain a `ThrowRecycling` sampler gets about one throw per practice period, so a
    memorisable constant would hand it the whole domain for free."""

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_required_force_differs_between_tasks(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        _rows, required = _throw_rows(family=family)
        assert len(np.unique(np.round(required, 9))) == len(required)

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_no_single_fixed_force_lands_more_than_a_minority_of_throws(
        *, family: TossingRoomSplitIdentityGoalType
    ) -> None:
        """The state-blind cap: the best fixed force a sampler could settle on, i.e. what
        a sampler that learned nothing about the state could still score.

        **This quantity is matched to the causal arm exactly**, at 185/400 (TRASH) and
        186/400 (RECYCLING) at seed 0 -- element for element, because the two arms draw
        the same tasks with the same required forces. That is not automatic, and it is
        the reason `target_force` is a resolved pair of causes rather than a direct
        Uniform draw: a `Uniform[0.1, 0.9)` target would score **120/400** here against
        the causal arm's **185/400**, because the causal arm's required force is a sum of
        two uniforms and so is TRIANGULAR on that span, concentrating mass near 0.5 where
        one fixed force catches more of it. A cross-arm comparison that read that
        difference as "learning" would have been wrong.

        Asserted as an inequality rather than pinned, for the same reason the causal arm
        does: how far above the floor a sampler gets is what the experiment measures, and
        pinning it here would either duplicate that result or break when it legitimately
        moves. The exact cross-arm equality is asserted in `test_fork_equivalence.py`,
        which is where difficulty-matching claims belong. 400 groundings rather than 80,
        because this is an estimate of a rate."""
        _rows, required = _throw_rows(family=family, num_tasks=400)
        best = max(
            int(np.sum(np.abs(required - force) < _TOLERANCE))
            for force in np.linspace(0.0, 1.0, 1001)
        )
        assert best < len(required) // 2, (
            f"{family.value}: a single fixed force lands {best}/{len(required)} throws."
        )


@pytest.mark.parametrize("family", _FAMILIES)
def test_two_of_the_ten_state_and_force_columns_carry_signal(
    *, family: TossingRoomSplitIdentityGoalType
) -> None:
    """The redundancy measurement, re-run per throw on this arm's representation.

    Each column is classified over that throw's applicable groundings as *constant* (one
    distinct value), *redundant* (an exact affine function of an earlier non-constant
    column) or *free*.

    The result is **9 constant / 0 redundant / 2 free out of 11**, and the two free
    columns are `item.target_force` and `force` -- the answer and the dial. The causal arm
    on the same measurement gets 3 free (`item.weight`, `bin.throw_distance`, `force`),
    and crucially none of ITS free columns is the answer. That is the entire difference
    between the two arms, expressed as a count."""
    rows, _required = _throw_rows(family=family)
    free: list[int] = []
    constant: list[str] = []
    redundant: list[str] = []
    for index, name in enumerate(_COLUMN_NAMES):
        column = rows[:, index]
        if np.ptp(column) == 0:
            constant.append(name)
            continue
        duplicates = any(
            np.max(
                np.abs(np.polyval(np.polyfit(rows[:, other], column, 1), rows[:, other]) - column)
            )
            < 1e-9
            for other in free
        )
        if duplicates:
            redundant.append(name)
        else:
            free.append(index)
    free_names = [_COLUMN_NAMES[index] for index in free]

    assert constant == [
        "bias",
        "robot.room",
        "robot.holding",
        "item.kind",
        "bin.count",
        "bin.room",
        "bin.kind",
        "room.index",
        "room.blocks_right",
    ]
    assert redundant == []
    assert free_names == ["item.target_force", "force"]
    assert len(free_names) == 2
