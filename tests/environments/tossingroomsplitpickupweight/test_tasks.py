from collections import Counter

import pytest

from hitl_pmp.environments.tossingroomsplitpickupweight.environment import (
    TossingRoomSplitPickupWeightEnvironment,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.tasks import (
    TossingRoomSplitPickupWeightGoalType,
    TossingRoomSplitPickupWeightTasks,
)


def _tasks(**kwargs) -> TossingRoomSplitPickupWeightTasks:
    return TossingRoomSplitPickupWeightTasks(
        env=TossingRoomSplitPickupWeightEnvironment(), **kwargs
    )


def _weight_seed(*, task) -> float:
    """The one value a task draws here, standing in for "what this task drew" throughout
    this file. In `tossingroomsplit` that was a per-item weight and a per-bin distance
    frozen into the initial state; here the weight belongs to whichever item gets picked
    up, so all a task carries is the seed naming its pre-sampled array."""
    return task.initial_state.get(
        obj=TossingRoomSplitPickupWeightEnvironment.pile, feature_name="weight_seed"
    )


def _goal_type(*, task) -> TossingRoomSplitPickupWeightGoalType:
    """Which goal family a built Task belongs to, read back off its goal atoms -- the
    same inference an analysis script reading a run's tasks back would make."""
    if all(atom.predicate in (TRASH_BIN_EMPTY, RECYCLING_BIN_EMPTY) for atom in task.goal.atoms):
        return TossingRoomSplitPickupWeightGoalType.EMPTY
    (atom,) = task.goal.atoms
    # The in-bin predicate is split per kind here, so the family is read off the
    # PREDICATE rather than off the bound item -- one of the few genuine test diffs
    # against Tossing Room.
    if atom.predicate == RECYCLING_IN_BIN:
        return TossingRoomSplitPickupWeightGoalType.RECYCLING
    assert atom.predicate == TRASH_IN_BIN
    return TossingRoomSplitPickupWeightGoalType.TRASH


def _test_goal_types(
    *, tasks: TossingRoomSplitPickupWeightTasks, count: int
) -> list[TossingRoomSplitPickupWeightGoalType]:
    return [_goal_type(task=tasks.sample_test_task()) for _ in range(count)]


@pytest.mark.parametrize("goal_type", list(TossingRoomSplitPickupWeightGoalType))
def test_forced_goal_type_initial_state_never_already_satisfies_goal(
    *, goal_type: TossingRoomSplitPickupWeightGoalType
) -> None:
    tasks = _tasks(seed=0, forced_goal_type=goal_type)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.is_satisfied(state=task.initial_state) is False


def test_recycling_goal_is_a_single_item_in_recycling_bin_atom() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == RECYCLING_IN_BIN
    assert atom.objects == (
        TossingRoomSplitPickupWeightEnvironment.recycling,
        TossingRoomSplitPickupWeightEnvironment.recycling_bin,
    )


def test_trash_goal_is_a_single_item_in_trash_bin_atom() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitPickupWeightGoalType.TRASH
    ).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == TRASH_IN_BIN
    assert atom.objects == (
        TossingRoomSplitPickupWeightEnvironment.trash,
        TossingRoomSplitPickupWeightEnvironment.trash_bin,
    )


def test_throw_tasks_start_with_empty_bins() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    assert (
        task.initial_state.get(
            obj=TossingRoomSplitPickupWeightEnvironment.recycling_bin, feature_name="count"
        )
        == 0.0
    )


def test_empty_goal_prefills_exactly_one_item_per_bin() -> None:
    """A bin holds at most one item, so the old Uniform{1, 2, 3} prefill (and its
    initial_count_low/high knobs) has exactly one legal value left."""
    tasks = _tasks(seed=2, forced_goal_type=TossingRoomSplitPickupWeightGoalType.EMPTY)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.atoms == frozenset({
            RECYCLING_BIN_EMPTY(
                state=task.initial_state,
                objects=(TossingRoomSplitPickupWeightEnvironment.recycling_bin,),
            ),
            TRASH_BIN_EMPTY(
                state=task.initial_state,
                objects=(TossingRoomSplitPickupWeightEnvironment.trash_bin,),
            ),
        })
        assert (
            task.initial_state.get(
                obj=TossingRoomSplitPickupWeightEnvironment.recycling_bin, feature_name="count"
            )
            == 1.0
        )
        assert (
            task.initial_state.get(
                obj=TossingRoomSplitPickupWeightEnvironment.trash_bin, feature_name="count"
            )
            == 1.0
        )


def test_every_task_draws_a_distinct_weight_seed_inside_its_range() -> None:
    tasks = _tasks(seed=3)
    seeds = [_weight_seed(task=tasks.sample_train_task()) for _ in range(20)]
    assert len(set(seeds)) == 20
    assert all(0 <= seed < tasks.weight_seed_high for seed in seeds)


def test_the_bin_distance_is_the_environment_s_fixed_one() -> None:
    """No longer a per-task draw: `required_force` is one-dimensional in the weight."""
    tasks = _tasks(seed=3)
    for _ in range(20):
        task = tasks.sample_train_task()
        for bin_obj in (
            TossingRoomSplitPickupWeightEnvironment.recycling_bin,
            TossingRoomSplitPickupWeightEnvironment.trash_bin,
        ):
            assert (
                task.initial_state.get(obj=bin_obj, feature_name="throw_distance")
                == tasks.env.throw_distance
            )


def test_seed_is_deterministic() -> None:
    first = _tasks(
        seed=42, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    second = _tasks(
        seed=42, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    assert _weight_seed(task=first) == _weight_seed(task=second)


def test_different_seeds_change_the_drawn_weight_seed() -> None:
    a = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    b = _tasks(
        seed=2, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING
    ).sample_train_task()
    assert _weight_seed(task=a) != _weight_seed(task=b)


def test_train_and_test_use_independent_streams() -> None:
    """Disjointness keyed on a continuous per-task draw, not goal type -- with only three
    discrete goal types a goal-type comparison would collide ~1/3 of the time (advisor's
    item 4)."""
    tasks = _tasks(seed=7, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING)
    train_weight = _weight_seed(task=tasks.sample_train_task())
    test_weight = _weight_seed(task=tasks.sample_test_task())
    assert train_weight != test_weight


def test_set_seed_rederives_both_streams() -> None:
    tasks = _tasks(seed=5, forced_goal_type=TossingRoomSplitPickupWeightGoalType.RECYCLING)
    first_train = _weight_seed(task=tasks.sample_train_task())
    first_test = _weight_seed(task=tasks.sample_test_task())
    tasks.set_seed(seed=5)
    assert _weight_seed(task=tasks.sample_train_task()) == first_train
    assert _weight_seed(task=tasks.sample_test_task()) == first_test


def test_default_mix_produces_more_than_one_goal_type() -> None:
    tasks = _tasks(seed=0)
    seen = set()
    for _ in range(50):
        task = tasks.sample_train_task()
        predicates = {atom.predicate for atom in task.goal.atoms}
        seen.add(frozenset(p.name for p in predicates))
    assert len(seen) > 1


# --- The fixed test-set composition (see TossingRoomSplitPickupWeightTasks' class docstring) ---


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_test_set_composition_is_exactly_14_14_2_at_30_tasks(*, seed: int) -> None:
    """The headline guarantee: realised counts, not expected ones, on every seed."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    assert Counter(_test_goal_types(tasks=tasks, count=30)) == {
        TossingRoomSplitPickupWeightGoalType.TRASH: 14,
        TossingRoomSplitPickupWeightGoalType.RECYCLING: 14,
        TossingRoomSplitPickupWeightGoalType.EMPTY: 2,
    }


@pytest.mark.parametrize("num_test_tasks", [1, 2, 3, 5, 10, 11, 30])
def test_composition_rule_is_well_formed_at_any_test_set_size(*, num_test_tasks: int) -> None:
    counts = _tasks(seed=0, num_test_tasks=num_test_tasks).test_goal_type_counts()
    assert sum(counts.values()) == num_test_tasks
    assert all(count >= 0 for count in counts.values())
    # EMPTY must never crowd out a throw family: it is the trivially-solved one, so an
    # all-EMPTY (or single-throw-family) test set would report a meaningless score.
    if num_test_tasks >= 2:
        assert counts[TossingRoomSplitPickupWeightGoalType.TRASH] >= 1
        assert counts[TossingRoomSplitPickupWeightGoalType.RECYCLING] >= 1


def test_composition_matches_the_documented_rule_at_key_sizes() -> None:
    assert _tasks(seed=0, num_test_tasks=30).test_goal_type_counts() == {
        TossingRoomSplitPickupWeightGoalType.TRASH: 14,
        TossingRoomSplitPickupWeightGoalType.RECYCLING: 14,
        TossingRoomSplitPickupWeightGoalType.EMPTY: 2,
    }
    assert _tasks(seed=0, num_test_tasks=10).test_goal_type_counts() == {
        TossingRoomSplitPickupWeightGoalType.TRASH: 4,
        TossingRoomSplitPickupWeightGoalType.RECYCLING: 4,
        TossingRoomSplitPickupWeightGoalType.EMPTY: 2,
    }
    # Below min_test_tasks_per_empty there is no room for an EMPTY sanity task, and the
    # odd leftover task breaks toward TRASH.
    assert _tasks(seed=0, num_test_tasks=3).test_goal_type_counts() == {
        TossingRoomSplitPickupWeightGoalType.TRASH: 2,
        TossingRoomSplitPickupWeightGoalType.RECYCLING: 1,
        TossingRoomSplitPickupWeightGoalType.EMPTY: 0,
    }


def test_realised_test_counts_match_test_goal_type_counts_at_every_size() -> None:
    for num_test_tasks in (1, 2, 3, 5, 10, 11, 30):
        tasks = _tasks(seed=4, num_test_tasks=num_test_tasks)
        realised = Counter(_test_goal_types(tasks=tasks, count=num_test_tasks))
        expected = {
            goal_type: count
            for goal_type, count in tasks.test_goal_type_counts().items()
            if count > 0
        }
        assert realised == expected


def test_test_set_order_is_shuffled_and_seed_dependent() -> None:
    """Same composition, different order per seed -- a family-sorted test set would make
    any task_index-keyed analysis misleading."""
    first = _test_goal_types(tasks=_tasks(seed=0, num_test_tasks=30), count=30)
    second = _test_goal_types(tasks=_tasks(seed=1, num_test_tasks=30), count=30)
    assert first != second
    assert Counter(first) == Counter(second)
    assert first != sorted(first, key=lambda goal_type: goal_type.value)


def test_drawing_past_the_test_set_size_starts_a_fresh_block() -> None:
    """The schedule wraps rather than running dry: a second block of test draws has the
    same composition again (the harness only draws num_test_tasks, but nothing here may
    depend on that)."""
    tasks = _tasks(seed=0, num_test_tasks=30)
    drawn = _test_goal_types(tasks=tasks, count=60)
    assert Counter(drawn[:30]) == Counter(drawn[30:])


def test_set_seed_rebuilds_the_test_schedule() -> None:
    """Unforced, unlike test_set_seed_rederives_both_streams above -- forced_goal_type
    bypasses the schedule entirely, so a stale half-consumed queue would slip past it."""
    tasks = _tasks(seed=5, num_test_tasks=30)
    first = _test_goal_types(tasks=tasks, count=30)
    tasks.set_seed(seed=5)
    assert _test_goal_types(tasks=tasks, count=30) == first


def test_forced_goal_type_still_overrides_the_test_schedule() -> None:
    tasks = _tasks(
        seed=0, num_test_tasks=30, forced_goal_type=TossingRoomSplitPickupWeightGoalType.EMPTY
    )
    assert set(_test_goal_types(tasks=tasks, count=10)) == {
        TossingRoomSplitPickupWeightGoalType.EMPTY
    }


@pytest.mark.parametrize(
    ("seed", "expected_goal_types", "expected_first_weight_seed"),
    [
        (
            0,
            "trash recycling recycling empty trash trash empty recycling empty trash "
            "recycling trash",
            1097657232,
        ),
        (
            1,
            "trash recycling empty trash empty trash recycling trash recycling recycling "
            "trash trash",
            1621709875,
        ),
        (
            2,
            "recycling empty recycling trash recycling recycling trash recycling trash "
            "trash trash trash",
            234731698,
        ),
    ],
)
def test_the_train_stream_is_pinned(
    *,
    seed: int,
    expected_goal_types: str,
    expected_first_weight_seed: int,
) -> None:
    """Golden values pinning the training stream -- pinning training tasks has silently
    turned a run into a different experiment on this project before.

    **These literals are deliberately NOT `tossingroomsplit`'s**, and that is the whole
    reason this is a separate domain. `build_task` here draws ONE integer per task (the
    weight seed) where `tossingroomsplit` draws four uniforms (two item weights, two bin
    distances), so every task after the first shifts and the goal-family sequences move
    wholesale. A number measured on this domain is measured on different tasks from any
    number measured on `tossingroomsplit`, and the two cannot be pooled."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    drawn = [tasks.sample_train_task() for _ in range(12)]
    assert [_goal_type(task=task).value for task in drawn] == expected_goal_types.split()
    assert _weight_seed(task=drawn[0]) == expected_first_weight_seed


def test_train_stream_still_samples_its_goal_types_rather_than_fixing_them() -> None:
    """The counterpart of the test-set guarantee: train family counts stay seed-dependent
    (which is exactly what the test set no longer is)."""
    counts_per_seed = set()
    for seed in range(5):
        tasks = _tasks(seed=seed)
        drawn = Counter(_goal_type(task=tasks.sample_train_task()).value for _ in range(30))
        counts_per_seed.add(tuple(sorted(drawn.items())))
    assert len(counts_per_seed) > 1
