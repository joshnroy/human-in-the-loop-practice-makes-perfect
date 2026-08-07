"""The one thing this domain variant exists for: an item's `weight` is drawn **at
pickup**, off a per-task array pre-sampled before the episode starts, instead of being
a per-task constant frozen into the task's initial state.

Everything else about the world is `tossingroomsplit` verbatim, so this file is the
whole behavioural difference. The tests below pin four separate claims, each of which a
plausible implementation gets wrong on its own:

  * the schedule is **pre-sampled**, not drawn inside `take_action` (so a second
    Environment instance built from the same config draws nothing and cannot shift what
    practice sees -- the property `PracticeLoop`'s separate evaluation environment
    depends on);
  * it is **per task**, not one flat per-run stream (so two arms that pick up different
    numbers of items still see the same weights for the same task);
  * it **never wraps** -- exhausting it raises rather than replaying, because
    periodicity in the training distribution would be indistinguishable from learning;
  * `throw_distance` is **fixed**, so `required_force` is a one-dimensional function of
    weight alone.
"""

import numpy as np
import pytest

from hitl_pmp.environments.tossingroom.environment import (
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.tasks import (
    TossingRoomGoalType,
    TossingRoomTasks,
)

_ENV = TossingRoomEnvironment
_ROBOT = _ENV.robot
_PILE = _ENV.pile
_TRASH = _ENV.trash
_RECYCLING = _ENV.recycling
_TRASH_BIN = _ENV.trash_bin
_RECYCLING_BIN = _ENV.recycling_bin


def _pickup(*, kind: int) -> np.ndarray:
    return np.array([float(_ENV.SKILL_PICKUP), float(kind), 0.0])


def _throw(*, kind: int, force: float) -> np.ndarray:
    return np.array([float(_ENV.SKILL_THROW), float(kind), force])


def _move(*, to_room: int) -> np.ndarray:
    return np.array([float(_ENV.SKILL_MOVE_ROOM), float(to_room), 0.0])


def _started(*, env: TossingRoomEnvironment, weight_seed: int = 11):
    state = env.build_initial_state(weight_seed=weight_seed)
    env.set_state(state=state)
    return state


def test_the_initial_state_carries_a_placeholder_weight_not_a_drawn_one() -> None:
    """Before any pickup nothing has been drawn, so both items sit at the canonical
    placeholder -- identical across two different weight seeds. This is the feature that
    used to BE the per-task draw."""
    env = _ENV()
    first = env.build_initial_state(weight_seed=1)
    second = env.build_initial_state(weight_seed=2)
    assert first.get(obj=_TRASH, feature_name="weight") == env.canonical_item_weight
    assert first.get(obj=_RECYCLING, feature_name="weight") == env.canonical_item_weight
    assert second.get(obj=_TRASH, feature_name="weight") == env.canonical_item_weight


def test_a_pickup_writes_the_next_scheduled_weight_onto_the_item() -> None:
    env = _ENV()
    _started(env=env)
    schedule = env.weight_schedule(weight_seed=11)
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    state = env.get_current_state()
    assert state.get(obj=_TRASH, feature_name="weight") == pytest.approx(schedule[0])
    assert state.get(obj=_PILE, feature_name="num_pickups") == 1.0


def test_consecutive_pickups_walk_the_schedule_forward() -> None:
    """One shared cursor across both kinds: the pile issues items in order, and which
    kind each one is does not change which weight it carries. That is what makes two
    arms taking different item kinds still comparable."""
    env = _ENV()
    _started(env=env)
    schedule = env.weight_schedule(weight_seed=11)
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    # Throw it away (in the wrong room, so it misses) to free the hand, then pick again.
    env.take_action(action=_throw(kind=_ENV.TRASH_KIND, force=0.0))
    env.take_action(action=_pickup(kind=_ENV.RECYCLING_KIND))
    state = env.get_current_state()
    assert state.get(obj=_TRASH, feature_name="weight") == pytest.approx(schedule[0])
    assert state.get(obj=_RECYCLING, feature_name="weight") == pytest.approx(schedule[1])
    assert state.get(obj=_PILE, feature_name="num_pickups") == 2.0


def test_a_refused_pickup_does_not_advance_the_schedule() -> None:
    """A pickup outside the pile's room, or with a full hand, is a silent no-op like
    every other out-of-context action here -- and must not burn a weight, or the
    schedule two arms walk would depend on how many illegal actions each tried."""
    env = _ENV()
    _started(env=env)
    # Rightward and back: stepping LEFT out of start_room crosses the one-way ledge, so
    # the robot could not return to the pile at all.
    env.take_action(action=_move(to_room=env.start_room + 1))
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))  # wrong room
    assert env.get_current_state().get(obj=_PILE, feature_name="num_pickups") == 0.0
    env.take_action(action=_move(to_room=env.start_room))
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    env.take_action(action=_pickup(kind=_ENV.RECYCLING_KIND))  # hand already full
    assert env.get_current_state().get(obj=_PILE, feature_name="num_pickups") == 1.0


def test_the_schedule_is_pre_sampled_before_the_episode_starts() -> None:
    """`build_initial_state` materialises the whole array, so `take_action` only ever
    INDEXES it. Nothing in the dynamics consumes randomness, which is what lets this
    domain be run with a separate evaluation Environment without shifting practice."""
    env = _ENV()
    assert env.pre_sampled_seeds() == frozenset()
    env.build_initial_state(weight_seed=3)
    assert env.pre_sampled_seeds() == frozenset({3})


def test_two_independent_environments_derive_the_same_schedule() -> None:
    """The seed travels in the STATE, and the array is a pure function of it, so the
    practice and evaluation instances agree without sharing anything."""
    practice, evaluation = _ENV(), _ENV()
    assert practice.weight_schedule(weight_seed=42) == pytest.approx(
        evaluation.weight_schedule(weight_seed=42)
    )


def test_different_seeds_give_different_schedules() -> None:
    env = _ENV()
    assert env.weight_schedule(weight_seed=1)[0] != env.weight_schedule(weight_seed=2)[0]


def test_every_scheduled_weight_lies_in_the_configured_range() -> None:
    env = _ENV(weight_schedule_length=512)
    schedule = env.weight_schedule(weight_seed=5)
    assert len(schedule) == 512
    assert min(schedule) >= env.pickup_weight_low
    assert max(schedule) < env.pickup_weight_high


def test_exhausting_the_schedule_raises_rather_than_wrapping() -> None:
    """Never wrap around: a periodic training distribution is a horrible bug -- a
    sampler would be re-fit on the same handful of weights forever while the run looked
    like it was drawing fresh ones."""
    env = _ENV(weight_schedule_length=2)
    _started(env=env)
    for _ in range(2):
        env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
        env.take_action(action=_throw(kind=_ENV.TRASH_KIND, force=0.0))
    with pytest.raises(RuntimeError, match="weight schedule"):
        env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))


def test_restoring_a_task_state_replays_that_task_s_weights_from_the_start() -> None:
    """The cursor lives in the State, so `reset_to_task` rewinds it -- which is what
    makes the held-out test set genuinely fixed: the same test task attempted at every
    checkpoint is attempted at the same weights."""
    env = _ENV()
    initial = _started(env=env)
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    first = env.get_current_state().get(obj=_TRASH, feature_name="weight")
    env.set_state(state=initial)
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    assert env.get_current_state().get(obj=_TRASH, feature_name="weight") == pytest.approx(first)


def test_a_task_s_weights_do_not_depend_on_how_many_pickups_an_earlier_task_took() -> None:
    """Per-task arrays, not one flat per-run stream. Two arms that spend different
    numbers of pickups on task A must still meet task B at identical weights."""
    tasks = TossingRoomTasks(env=_ENV(), seed=0)
    task_a, task_b = tasks.sample_train_task(), tasks.sample_train_task()
    env = tasks.env

    def first_weight_of(*, task, warmup_pickups: int) -> float:
        env.set_state(state=task.initial_state.model_copy(deep=True))
        for _ in range(warmup_pickups):
            env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
            env.take_action(action=_throw(kind=_ENV.TRASH_KIND, force=0.0))
        env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
        return float(env.get_current_state().get(obj=_TRASH, feature_name="weight"))

    first_weight_of(task=task_a, warmup_pickups=0)
    patient = first_weight_of(task=task_b, warmup_pickups=0)
    first_weight_of(task=task_a, warmup_pickups=4)
    hasty = first_weight_of(task=task_b, warmup_pickups=0)
    assert patient == pytest.approx(hasty)


def test_throw_distance_is_fixed_rather_than_drawn_per_task() -> None:
    tasks = TossingRoomTasks(env=_ENV(), seed=0)
    distances = {
        float(
            tasks.sample_train_task().initial_state.get(
                obj=_TRASH_BIN, feature_name="throw_distance"
            )
        )
        for _ in range(20)
    }
    assert distances == {tasks.env.throw_distance}


def test_required_force_is_a_function_of_weight_alone() -> None:
    """With the distance pinned, the two-cause relation collapses to one dimension --
    the whole point of fixing it. The relation itself is unchanged, so the domain stays
    a strict specialisation of `tossingroomsplit` rather than a different world."""
    env = _ENV()
    at_low = env.required_force(
        throw_distance=env.throw_distance, item_weight=env.pickup_weight_low
    )
    at_high = env.required_force(
        throw_distance=env.throw_distance, item_weight=env.pickup_weight_high
    )
    # Both ends inside the U(0, 1) band a uniformly random force is drawn from, with the
    # tolerance window wholly inside it too -- the constraint tossingroomsplit's numbers
    # were chosen to satisfy, preserved here.
    # The span is exactly [throw_tolerance, 1 - throw_tolerance], so every winning window
    # sits wholly inside the U(0, 1) band and none is clipped -- exactly
    # tossingroomsplit's span, restored by doubling weight_coefficient once the distance
    # stopped varying. (The upper end is the excluded endpoint of the half-open weight
    # range, so it is approached rather than reached.)
    assert at_low == pytest.approx(env.throw_tolerance)
    assert at_high == pytest.approx(1.0 - env.throw_tolerance)


def test_a_throw_lands_at_the_force_the_weight_drawn_at_pickup_requires() -> None:
    """End to end: the weight that decides the throw is the one the pickup drew, not
    anything the task's initial state carried."""
    env = _ENV()
    _started(env=env)
    env.take_action(action=_pickup(kind=_ENV.TRASH_KIND))
    state = env.get_current_state()
    drawn = float(state.get(obj=_TRASH, feature_name="weight"))
    for room in range(env.start_room + 1, env.trash_bin_room + 1):
        env.take_action(action=_move(to_room=room))
    required = env.required_force(throw_distance=env.throw_distance, item_weight=drawn)
    env.take_action(action=_throw(kind=_ENV.TRASH_KIND, force=required))
    assert env.get_current_state().get(obj=_TRASH_BIN, feature_name="count") == 1.0


def test_each_task_carries_its_own_weight_seed() -> None:
    tasks = TossingRoomTasks(env=_ENV(), seed=0, forced_goal_type=TossingRoomGoalType.TRASH)
    seeds = {
        float(tasks.sample_train_task().initial_state.get(obj=_PILE, feature_name="weight_seed"))
        for _ in range(20)
    }
    assert len(seeds) == 20


def test_the_test_stream_is_reproducible_across_instances() -> None:
    """Two identically-configured Tasks (practice's and evaluation's) must hand out the
    same test tasks, weight seeds included."""
    first = TossingRoomTasks(env=_ENV(), seed=4, num_test_tasks=10)
    second = TossingRoomTasks(env=_ENV(), seed=4, num_test_tasks=10)
    for _ in range(10):
        left = first.sample_test_task().initial_state.get(obj=_PILE, feature_name="weight_seed")
        right = second.sample_test_task().initial_state.get(obj=_PILE, feature_name="weight_seed")
        assert left == right
