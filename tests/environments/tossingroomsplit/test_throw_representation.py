"""The three properties each throw's representation has to have, asserted rather than
argued, plus the redundancy measurement that motivated replacing it.

The unsplit domain's version of this file
(`tests/environments/tossingroom/test_throw_representation.py`) carries the full
argument; this one exists because `environments/tossingroomsplit` is a **verbatim fork**,
not a subclass, so nothing propagates and the two domains would silently stop being the
same learning problem.

It is also a *stronger* measurement here, and that is worth knowing rather than
discovering. `EesMethod.sampler` keys its classifiers by `skill_name`, so `ThrowTrash`
and `ThrowRecycling` each see only their own skill's rows -- and within one skill the
bound bin, item and room never vary. Every column that was an affine copy of the `kind`
bit in the unsplit domain is a flat **constant** here, so each sampler's row would be
almost entirely dead weight and the two causes are the only thing that saves it:

1. **The answer is not in the state.** No state feature equals the required force, and no
   single feature predicts it affinely to within the tolerance.
2. **The causes are in the state**, so the problem stays well posed.
3. **It varies per task**, so a sampler learns a relation rather than a constant.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.skills import TossingRoomSplitSkills
from hitl_pmp.environments.tossingroomsplit.tasks import (
    TossingRoomSplitGoalType,
    TossingRoomSplitTasks,
)

_TOLERANCE = TossingRoomSplitEnvironment.model_fields["throw_tolerance"].default

# The row layout, by role, in each throw's own parameter order (robot, item, bin, room).
# Identical for both throws by design -- see the environment's docstring on why the split
# types keep identical feature schemas.
_COLUMN_NAMES = (
    "bias",
    "robot.room",
    "robot.holding",
    "item.kind",
    "item.weight",
    "bin.count",
    "bin.room",
    "bin.kind",
    "bin.throw_distance",
    "room.index",
    "room.blocks_right",
    "force",
)
_FAMILIES = (TossingRoomSplitGoalType.TRASH, TossingRoomSplitGoalType.RECYCLING)


def _env() -> TossingRoomSplitEnvironment:
    return TossingRoomSplitEnvironment()


def _throw_rows(
    *, family: TossingRoomSplitGoalType, num_tasks: int = 80, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """One classifier input row per applicable grounding of **one** throw skill, with
    that grounding's required force beside it. Per skill, not pooled, because that is
    what each `LearnedSkillSampler` actually sees.

    "Applicable" means the preconditions hold: holding that kind, standing in its bin's
    room, that bin empty. Returns (rows, required force per row)."""
    env = _env()
    tasks = TossingRoomSplitTasks(env=env, seed=seed, num_test_tasks=30)
    force_rng = np.random.default_rng(seed)
    kind = env.TRASH_KIND if family is TossingRoomSplitGoalType.TRASH else env.RECYCLING_KIND
    bin_room = env.bin_room_for_kind(kind=kind)
    item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)
    rows: list[list[float]] = []
    required: list[float] = []
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
        rows.append(row)
        required.append(
            env.required_force(
                throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
                item_weight=float(state.get(obj=item, feature_name="weight")),
            )
        )
    return np.array(rows), np.array(required)


@pytest.mark.parametrize("family", _FAMILIES)
def test_the_row_layout_is_the_one_these_tests_assume(*, family: TossingRoomSplitGoalType) -> None:
    """Guard: every column index below is named, so a feature added to any of a throw's
    four object types has to come here and be classified rather than silently joining the
    row. Checked on both throws, which also pins that the two rows are the same width --
    the property that makes the two samplers the same architecture."""
    rows, _required = _throw_rows(family=family, num_tasks=2)
    assert rows.shape[1] == len(_COLUMN_NAMES)


class TestTheAnswerIsNotInTheState:
    """Property 1. The defect this whole change exists to remove."""

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_no_state_feature_equals_the_required_force(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        rows, required = _throw_rows(family=family)
        for index, name in enumerate(_COLUMN_NAMES):
            if name == "force":
                continue  # the dial the sampler chooses, not a state feature
            assert not np.allclose(rows[:, index], required), (
                f"{family.value}: {name} equals the required force -- the sampler can "
                f"read the answer straight out of its own input row."
            )

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_no_single_state_feature_predicts_the_required_force_within_tolerance(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        """Equality is not the only way to leak an answer -- a rescaled copy would be just
        as readable. Fit the best affine map from each single column and require its
        worst-case residual to exceed the tolerance."""
        rows, required = _throw_rows(family=family)
        for index, name in enumerate(_COLUMN_NAMES):
            if name == "force":
                continue
            column = rows[:, index]
            if np.ptp(column) == 0:
                continue  # a constant column predicts nothing
            slope, intercept = np.polyfit(column, required, 1)
            worst_residual = float(np.max(np.abs(slope * column + intercept - required)))
            assert worst_residual >= _TOLERANCE, (
                f"{family.value}: {name} alone predicts the required force to within "
                f"{worst_residual:.4f}, inside the {_TOLERANCE} tolerance."
            )


class TestTheCausesAreInTheState:
    """Property 2. Removing the answer is only half of it: the problem has to stay
    solvable from what the agent can see."""

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_required_force_is_a_function_of_two_observed_columns(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        rows, required = _throw_rows(family=family)
        env = _env()
        distances = rows[:, _COLUMN_NAMES.index("bin.throw_distance")]
        weights = rows[:, _COLUMN_NAMES.index("item.weight")]
        predicted = np.array([
            env.required_force(throw_distance=float(d), item_weight=float(w))
            for d, w in zip(distances, weights, strict=True)
        ])
        assert np.allclose(predicted, required)

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_throwing_at_the_required_force_lands_and_just_outside_it_misses(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        """`required_force` is the dynamics, not a parallel description of them."""
        env = _env()
        tasks = TossingRoomSplitTasks(env=env, seed=1, num_test_tasks=30)
        kind = env.TRASH_KIND if family is TossingRoomSplitGoalType.TRASH else env.RECYCLING_KIND
        item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)
        bin_room = env.bin_room_for_kind(kind=kind)
        for _ in range(10):
            task = tasks.build_task(goal_type=family, rng=tasks.train_rng)
            required = env.required_force(
                throw_distance=float(
                    task.initial_state.get(obj=bin_obj, feature_name="throw_distance")
                ),
                item_weight=float(task.initial_state.get(obj=item, feature_name="weight")),
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
    def test_neither_cause_can_be_ignored() -> None:
        """Both causes have to be load-bearing, or the second one is decoration."""
        env = _env()
        tasks = TossingRoomSplitTasks(env=_env())
        distance_swing = abs(
            env.required_force(throw_distance=tasks.distance_high, item_weight=tasks.weight_low)
            - env.required_force(throw_distance=tasks.distance_low, item_weight=tasks.weight_low)
        )
        weight_swing = abs(
            env.required_force(throw_distance=tasks.distance_low, item_weight=tasks.weight_high)
            - env.required_force(throw_distance=tasks.distance_low, item_weight=tasks.weight_low)
        )
        assert distance_swing > 2 * _TOLERANCE
        assert weight_swing > 2 * _TOLERANCE

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_every_sampled_task_is_reachable_by_a_force_the_base_sampler_can_draw(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        """`TossingRoomSplitSkills.sample_params` draws Uniform(0, 1) for BOTH throws --
        deliberately the same prior, so a head start cannot confound the comparison. If a
        task's required force sat within a tolerance of 0 or 1 its winning window would be
        clipped, and one family could be quietly harder for a reason unrelated to its
        practice budget, which is the whole thing this domain measures."""
        _rows, required = _throw_rows(family=family, num_tasks=200, seed=2)
        assert float(np.min(required)) > _TOLERANCE
        assert float(np.max(required)) < 1.0 - _TOLERANCE

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_a_uniformly_random_force_lands_with_probability_exactly_one_fifth(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        """**The invariant any comparison against a different throw representation rests
        on.** A random draw's landing rate is how hard the throw is to hit *by luck*. If
        two representations differ on it, a difference in learned-sampler performance
        between them could be that, rather than anything a classifier did or did not
        learn.

        Asserted per task and exactly, not as a pooled mean, because a mean hides
        clipping: a representation whose tasks land 0.10 and 0.30 in equal numbers
        averages 0.2 while making half its tasks twice as hard. That is not hypothetical
        -- the pre-#80 `target_force ~ U[0.5, 1.0)` representation clips for
        `target > 0.9`, giving a mean of 0.19 over a per-task range of 0.10 to 0.20.

        Both halves of the probability are pinned here, because either alone is vacuous:

        * the **window**, `(required - 0.1, required + 0.1)`, must have width
          `2 * throw_tolerance` and lie wholly inside the draw band -- never clipped; and
        * the **draw band** must actually be `Uniform(0, 1)`, read off
          `TossingRoomSplitSkills.sample_params` rather than assumed. Without this,
          widening the band to `Uniform(0, 2)` would halve the real landing rate with
          every assertion still passing.

        The empirical count is then the two together, through the REAL dynamics
        (`env.take_action`), so the analytic window is checked against the code that
        implements it rather than against a restatement of itself.
        """
        _rows, required = _throw_rows(family=family, num_tasks=400, seed=3)
        overlap = np.minimum(1.0, required + _TOLERANCE) - np.maximum(0.0, required - _TOLERANCE)
        assert np.allclose(overlap, 2 * _TOLERANCE)
        exactly_one_fifth = int(np.sum(np.isclose(overlap, 0.2)))
        assert exactly_one_fifth == len(required), (
            f"{family.value}: {exactly_one_fifth}/{len(required)} tasks have a "
            f"landing probability of exactly 0.2."
        )

        # The draw band itself. `sample_params` is the only thing that decides it, and a
        # window of width 0.2 only means probability 0.2 against a Uniform(0, 1) draw.
        env = _env()
        kind = env.TRASH_KIND if family is TossingRoomSplitGoalType.TRASH else env.RECYCLING_KIND
        item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)
        bin_room = env.bin_room_for_kind(kind=kind)
        ground = GroundSkill(
            skill=TossingRoomSplitSkills.THROW_TRASH
            if family is TossingRoomSplitGoalType.TRASH
            else TossingRoomSplitSkills.THROW_RECYCLING,
            objects=(env.robot, item, bin_obj, env.get_rooms()[bin_room]),
        )
        band_rng = np.random.default_rng(4)
        sampled = np.array([
            TossingRoomSplitSkills.sample_params(ground_skill=ground, rng=band_rng)[0]
            for _ in range(4000)
        ])
        # Unit-interval uniform: the support and the first two moments a Uniform(0, 1)
        # has, which is what makes a window of width 0.2 a probability of 0.2.
        assert float(sampled.min()) >= 0.0
        assert float(sampled.max()) < 1.0
        assert float(sampled.mean()) == pytest.approx(0.5, abs=0.02)
        assert float(sampled.std()) == pytest.approx(1 / np.sqrt(12), abs=0.02)

        # End to end, through the real dynamics rather than the analytic window.
        tasks = TossingRoomSplitTasks(env=env, seed=7, num_test_tasks=30)
        rng = np.random.default_rng(7)
        landed = 0
        attempts = 0
        for _ in range(200):
            task = tasks.build_task(goal_type=family, rng=tasks.train_rng)
            for _ in range(25):
                params = TossingRoomSplitSkills.sample_params(ground_skill=ground, rng=rng)
                state = task.initial_state.model_copy(deep=True)
                state.set(obj=env.robot, feature_name="room", feature_val=float(bin_room))
                state.set(obj=env.robot, feature_name="holding", feature_val=float(kind))
                env.set_state(state=state)
                after = env.take_action(
                    action=TossingRoomSplitSkills.compute_action(
                        ground_skill=ground, params=params, state=state
                    )
                )
                landed += int(after.get(obj=bin_obj, feature_name="count") == 1.0)
                attempts += 1
        # 5000 draws at p = 0.2 has sd = 0.0057, so 4 sd is 0.023. A representation at
        # 0.19 (the pre-#80 one) fails this; noise does not.
        assert landed / attempts == pytest.approx(0.2, abs=0.023), (
            f"{family.value}: a base-sampler force landed {landed}/{attempts}."
        )


class TestItVariesPerTask:
    """Property 3. A constant requirement would be memorised in one throw -- and in this
    domain a `ThrowRecycling` sampler gets about one throw per practice period, so a
    memorisable constant would hand it the whole domain for free."""

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_the_required_force_differs_between_tasks(*, family: TossingRoomSplitGoalType) -> None:
        _rows, required = _throw_rows(family=family)
        assert len(np.unique(np.round(required, 9))) == len(required)

    @staticmethod
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_no_single_fixed_force_lands_more_than_a_minority_of_throws(
        *, family: TossingRoomSplitGoalType
    ) -> None:
        """The state-blind cap: the best fixed force a sampler could settle on. About
        7/16 here (the required force is triangular on [0.1, 0.9], a sum of two uniforms)
        against 2/5 for the old `U(0.5, 1.0)` `target_force` -- comparable, and far below
        what conditioning on the state buys. 400 groundings rather than 80, because this
        is an estimate of a rate."""
        _rows, required = _throw_rows(family=family, num_tasks=400)
        best = max(
            int(np.sum(np.abs(required - force) < _TOLERANCE))
            for force in np.linspace(0.0, 1.0, 1001)
        )
        assert best < len(required) // 2, (
            f"{family.value}: a single fixed force lands {best}/{len(required)} throws."
        )


@pytest.mark.parametrize("family", _FAMILIES)
def test_three_of_the_eleven_state_and_force_columns_carry_signal(
    *, family: TossingRoomSplitGoalType
) -> None:
    """The measurement that motivated the change, re-run per throw on the replacement.

    Each column is classified over that throw's applicable groundings as *constant* (one
    distinct value), *redundant* (an exact affine function of an earlier non-constant
    column) or *free*.

    **The split makes this starker than in the unsplit domain.** There, `robot.room`,
    `robot.holding`, `item.kind`, `bin.room`, `bin.kind` and `room.index` all varied
    together with the item kind, so five were classified redundant and one free. Here a
    sampler only ever sees one kind, so all six are flat **constants** and there is
    nothing redundant left to find: 9 constant / 0 redundant / 3 free. Under
    `item.target_force` those free columns were `target_force` and `force` and nothing
    else -- **2 of 10** carrying signal, with the answer among them. Now they are
    `item.weight`, `bin.throw_distance` and `force`: **3 of 11**, and none is the answer.
    """
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
    assert free_names == ["item.weight", "bin.throw_distance", "force"]
    assert len(free_names) == 3
