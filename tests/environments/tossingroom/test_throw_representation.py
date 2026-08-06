"""The three properties the throw representation has to have, asserted rather than
argued, plus the redundancy measurement that motivated replacing it.

`Throw`'s classifier input row is `[1.0] + concat(state[obj] for obj in
ground_skill.objects) + [force]` (`EesMethod.state_features` +
`LearnedSkillSampler.build_sampler_input`). The domain used to put `item.target_force`
in that row, which made "learn a skill sampler" mean "learn `|x_10 - x_4| < 0.1`" -- a
comparison between two of the net's own inputs, solvable from ~60 labelled throws.

1. **The answer is not in the state.** No state feature equals the required force, and
   no single feature predicts it affinely to within the tolerance either.
2. **The causes are in the state.** The required force is a function of two features the
   agent observes, so the problem stays well posed -- a classifier that could not in
   principle succeed would be worse than the identity it replaces.
3. **It varies per task**, so a sampler must learn a relation rather than memorise a
   constant.

Columns are named by ROLE (`item.weight`, `bin.throw_distance`, ...) rather than by the
bound object, because the classifier is per skill *name* and sees the row positionally:
the same slot holds `trash.weight` on one grounding and `recycling.weight` on the next.
"""

import numpy as np

from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks

_TOLERANCE = TossingRoomEnvironment.model_fields["throw_tolerance"].default

# The row layout, by role, in `Throw`'s own parameter order (robot, item, bin, room).
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


def _env() -> TossingRoomEnvironment:
    return TossingRoomEnvironment()


def _throw_rows(*, num_tasks: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One classifier input row per applicable `Throw` grounding, with that grounding's
    required force beside it. "Applicable" means the preconditions hold -- holding the
    item, standing in its bin's room, that bin empty -- which is the only situation the
    sampler is ever asked about, and therefore the only one whose feature variety
    matters. 40 tasks x both item kinds = 80 groundings, the same denominator the
    pre-change measurement used.

    Returns (rows, required force per row)."""
    env = _env()
    tasks = TossingRoomTasks(env=env, seed=seed, num_test_tasks=30)
    force_rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    required: list[float] = []
    for _ in range(num_tasks):
        for goal_type in (TossingRoomGoalType.TRASH, TossingRoomGoalType.RECYCLING):
            task = tasks.build_task(goal_type=goal_type, rng=tasks.train_rng)
            kind = env.TRASH_KIND if goal_type is TossingRoomGoalType.TRASH else env.RECYCLING_KIND
            bin_room = env.bin_room_for_kind(kind=kind)
            state = task.initial_state.model_copy(deep=True)
            state.set(obj=env.robot, feature_name="holding", feature_val=float(kind))
            state.set(obj=env.robot, feature_name="room", feature_val=float(bin_room))
            item, bin_obj = env.item_for_kind(kind=kind), env.bin_for_kind(kind=kind)

            row = [1.0]
            for obj in (env.robot, item, bin_obj, env.get_rooms()[bin_room]):
                row.extend(
                    float(state.get(obj=obj, feature_name=feature_name))
                    for feature_name in obj.type.feature_names
                )
            # The dial the sampler picks. Drawn the way the base sampler draws it, so the
            # column has the same variety the classifier actually sees.
            row.append(float(force_rng.uniform(0.0, 1.0)))
            rows.append(row)
            required.append(
                env.required_force(
                    throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
                    item_weight=float(state.get(obj=item, feature_name="weight")),
                )
            )
    return np.array(rows), np.array(required)


def test_the_row_layout_is_the_one_these_tests_assume() -> None:
    """Guard: every column index below is named, so a feature added to any of Throw's
    four object types has to come here and be classified rather than silently joining
    the row."""
    rows, _required = _throw_rows(num_tasks=1)
    assert rows.shape[1] == len(_COLUMN_NAMES)


class TestTheAnswerIsNotInTheState:
    """Property 1. The defect this whole change exists to remove."""

    @staticmethod
    def test_no_state_feature_equals_the_required_force() -> None:
        rows, required = _throw_rows()
        for index, name in enumerate(_COLUMN_NAMES):
            if name == "force":
                continue  # the dial the sampler chooses, not a state feature
            assert not np.allclose(rows[:, index], required), (
                f"{name} equals the required force -- the sampler can read the answer "
                f"straight out of its own input row, which is the defect this replaces."
            )

    @staticmethod
    def test_no_single_state_feature_predicts_the_required_force_within_tolerance() -> None:
        """Stronger, and the one that matters: equality is not the only way to leak the
        answer -- a rescaled copy would be just as readable. So fit the best affine map
        from each single column and require its worst-case residual to exceed the
        tolerance, i.e. no one-column read is good enough to land a throw. This is what
        forces the design to carry *two* causes: with one, reading it would be enough."""
        rows, required = _throw_rows()
        for index, name in enumerate(_COLUMN_NAMES):
            if name == "force":
                continue
            column = rows[:, index]
            if np.ptp(column) == 0:
                continue  # a constant column predicts nothing
            slope, intercept = np.polyfit(column, required, 1)
            worst_residual = float(np.max(np.abs(slope * column + intercept - required)))
            assert worst_residual >= _TOLERANCE, (
                f"{name} alone predicts the required force to within {worst_residual:.4f}, "
                f"inside the {_TOLERANCE} tolerance -- one column would solve the domain."
            )


class TestTheCausesAreInTheState:
    """Property 2. Removing the answer is only half of it: the problem has to stay
    solvable from what the agent can see, or the sampler is being asked for something no
    amount of practice could supply."""

    @staticmethod
    def test_the_required_force_is_a_function_of_two_observed_columns() -> None:
        rows, required = _throw_rows()
        env = _env()
        distances = rows[:, _COLUMN_NAMES.index("bin.throw_distance")]
        weights = rows[:, _COLUMN_NAMES.index("item.weight")]
        predicted = np.array([
            env.required_force(throw_distance=float(d), item_weight=float(w))
            for d, w in zip(distances, weights, strict=True)
        ])
        assert np.allclose(predicted, required)

    @staticmethod
    def test_throwing_at_the_required_force_lands_and_just_outside_it_misses() -> None:
        """`required_force` is the dynamics, not a parallel description of them."""
        env = _env()
        tasks = TossingRoomTasks(env=env, seed=1, num_test_tasks=30)
        for _ in range(10):
            task = tasks.build_task(goal_type=TossingRoomGoalType.RECYCLING, rng=tasks.train_rng)
            required = env.required_force(
                throw_distance=float(
                    task.initial_state.get(obj=env.recycling_bin, feature_name="throw_distance")
                ),
                item_weight=float(task.initial_state.get(obj=env.recycling, feature_name="weight")),
            )
            for force, should_land in (
                (required, True),
                (required + _TOLERANCE + 1e-6, False),
                (required - _TOLERANCE - 1e-6, False),
            ):
                state = task.initial_state.model_copy(deep=True)
                state.set(
                    obj=env.robot, feature_name="room", feature_val=float(env.recycling_bin_room)
                )
                state.set(
                    obj=env.robot, feature_name="holding", feature_val=float(env.RECYCLING_KIND)
                )
                env.set_state(state=state)
                after = env.take_action(
                    action=np.array([
                        float(env.SKILL_THROW),
                        float(env.RECYCLING_KIND),
                        float(force),
                    ])
                )
                landed = bool(after.get(obj=env.recycling_bin, feature_name="count") == 1.0)
                assert landed is should_land

    @staticmethod
    def test_neither_cause_can_be_ignored() -> None:
        """Both causes have to be load-bearing, or the second one is decoration. Holding
        one fixed and sweeping the other must move the required force by more than the
        tolerance across its own draw range."""
        env = _env()
        tasks = TossingRoomTasks(env=_env())
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
    def test_every_sampled_task_is_reachable_by_a_force_the_base_sampler_can_draw() -> None:
        """`TossingRoomSkills.sample_params` draws Uniform(0, 1). If a task's required
        force sat within a tolerance of 0 or 1 its winning window would be clipped, so
        some tasks would be quietly harder than others for reasons unrelated to learning
        -- and the random-force base rate would stop being one number."""
        env = _env()
        tasks = TossingRoomTasks(env=env, seed=2, num_test_tasks=30)
        for _ in range(200):
            task = tasks.build_task(goal_type=TossingRoomGoalType.TRASH, rng=tasks.train_rng)
            required = env.required_force(
                throw_distance=float(
                    task.initial_state.get(obj=env.trash_bin, feature_name="throw_distance")
                ),
                item_weight=float(task.initial_state.get(obj=env.trash, feature_name="weight")),
            )
            assert _TOLERANCE < required < 1.0 - _TOLERANCE


class TestItVariesPerTask:
    """Property 3. A constant requirement would be memorised in one throw."""

    @staticmethod
    def test_the_required_force_differs_between_tasks() -> None:
        _rows, required = _throw_rows()
        assert len(np.unique(np.round(required, 9))) == len(required)

    @staticmethod
    def test_no_single_fixed_force_lands_more_than_a_minority_of_throws() -> None:
        """The sharp version: sweep the whole draw range and find the best fixed force,
        which is what a STATE-BLIND sampler is capped at. The required force is triangular
        on [0.1, 0.9] (a sum of two uniforms), so that cap is about 7/16 -- against 2/5
        for the old `U(0.5, 1.0)` `target_force`, i.e. comparable rather than a
        regression, and far below what conditioning on the state buys. 800 groundings
        rather than 80, because this is an estimate of a rate and 80 is too few to place
        it either side of a half."""
        _rows, required = _throw_rows(num_tasks=400)
        best = max(
            int(np.sum(np.abs(required - force) < _TOLERANCE))
            for force in np.linspace(0.0, 1.0, 1001)
        )
        assert best < len(required) // 2, (
            f"a single fixed force lands {best}/{len(required)} throws -- the sampler "
            f"could ignore the state entirely."
        )


def test_three_of_the_eleven_state_and_force_columns_carry_signal() -> None:
    """The measurement that motivated the change, re-run on the replacement.

    Each column is classified over the applicable groundings as *constant* (one distinct
    value), *redundant* (an exact affine function of an earlier non-constant column) or
    *free*. Under `item.target_force` this read **3 constant / 5 redundant / 3 free** on
    an 11-column row, and only **2 of the 10** state-plus-force columns carried signal
    (`target_force` and `force`) -- 5 of the rest were affine copies of the `kind` bit
    that `Throw`'s preconditions force equal across `robot.room`, `robot.holding`,
    `bin.room`, `bin.kind` and `room.index`.

    Here it must read **3 constant / 5 redundant / 4 free** on a 12-column row, with
    **3 of the 11** carrying signal: `item.weight`, `bin.throw_distance` and `force`.
    The redundant and constant blocks are unchanged, which is the honest statement --
    they are structural (a bin's room *is* the robot's room whenever a throw applies),
    not something a feature swap could remove. What moved is that the free block gained
    a column and none of the free columns is the answer any more."""
    rows, _required = _throw_rows()
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

    assert constant == ["bias", "bin.count", "room.blocks_right"]
    assert redundant == ["robot.holding", "item.kind", "bin.room", "bin.kind", "room.index"]
    assert free_names == ["robot.room", "item.weight", "bin.throw_distance", "force"]
    # `robot.room` is free but carries no signal: it is the one surviving copy of the
    # kind bit, and the label does not depend on it (the required force is the same
    # function in either bin's room).
    signal = [name for name in free_names if name != "robot.room"]
    assert signal == ["item.weight", "bin.throw_distance", "force"]
    assert len(signal) == 3
