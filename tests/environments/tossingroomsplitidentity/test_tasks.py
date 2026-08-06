from collections import Counter

import pytest

from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplitidentity.tasks import (
    TossingRoomSplitIdentityGoalType,
    TossingRoomSplitIdentityTasks,
)


def _tasks(**kwargs) -> TossingRoomSplitIdentityTasks:
    return TossingRoomSplitIdentityTasks(env=TossingRoomSplitIdentityEnvironment(), **kwargs)


def _recycling_target(*, task) -> float:
    """The one continuous value a recycling task draws, standing in for "the continuous
    value this task drew" throughout this file.

    Unlike the causal arm's `weight`/`throw_distance` pair, this single feature IS the
    required throw force -- `required_force` is the identity on it -- and the agent reads
    it directly at index 4 of every `ThrowRecycling` classifier row. See
    tests/environments/tossingroomsplitidentity/test_throw_representation.py."""
    return task.initial_state.get(
        obj=TossingRoomSplitIdentityEnvironment.recycling, feature_name="target_force"
    )


def _goal_type(*, task) -> TossingRoomSplitIdentityGoalType:
    """Which goal family a built Task belongs to, read back off its goal atoms -- the
    same inference an analysis script reading a run's tasks back would make."""
    if all(atom.predicate in (TRASH_BIN_EMPTY, RECYCLING_BIN_EMPTY) for atom in task.goal.atoms):
        return TossingRoomSplitIdentityGoalType.EMPTY
    (atom,) = task.goal.atoms
    # The in-bin predicate is split per kind here, so the family is read off the
    # PREDICATE rather than off the bound item -- one of the few genuine test diffs
    # against Tossing Room.
    if atom.predicate == RECYCLING_IN_BIN:
        return TossingRoomSplitIdentityGoalType.RECYCLING
    assert atom.predicate == TRASH_IN_BIN
    return TossingRoomSplitIdentityGoalType.TRASH


def _test_goal_types(
    *, tasks: TossingRoomSplitIdentityTasks, count: int
) -> list[TossingRoomSplitIdentityGoalType]:
    return [_goal_type(task=tasks.sample_test_task()) for _ in range(count)]


@pytest.mark.parametrize("goal_type", list(TossingRoomSplitIdentityGoalType))
def test_forced_goal_type_initial_state_never_already_satisfies_goal(
    *, goal_type: TossingRoomSplitIdentityGoalType
) -> None:
    tasks = _tasks(seed=0, forced_goal_type=goal_type)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.is_satisfied(state=task.initial_state) is False


def test_recycling_goal_is_a_single_item_in_recycling_bin_atom() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == RECYCLING_IN_BIN
    assert atom.objects == (
        TossingRoomSplitIdentityEnvironment.recycling,
        TossingRoomSplitIdentityEnvironment.recycling_bin,
    )


def test_trash_goal_is_a_single_item_in_trash_bin_atom() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitIdentityGoalType.TRASH
    ).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == TRASH_IN_BIN
    assert atom.objects == (
        TossingRoomSplitIdentityEnvironment.trash,
        TossingRoomSplitIdentityEnvironment.trash_bin,
    )


def test_throw_tasks_start_with_empty_bins() -> None:
    task = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    assert (
        task.initial_state.get(
            obj=TossingRoomSplitIdentityEnvironment.recycling_bin, feature_name="count"
        )
        == 0.0
    )


def test_empty_goal_prefills_exactly_one_item_per_bin() -> None:
    """A bin holds at most one item, so the old Uniform{1, 2, 3} prefill (and its
    initial_count_low/high knobs) has exactly one legal value left."""
    tasks = _tasks(seed=2, forced_goal_type=TossingRoomSplitIdentityGoalType.EMPTY)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.atoms == frozenset({
            RECYCLING_BIN_EMPTY(
                state=task.initial_state,
                objects=(TossingRoomSplitIdentityEnvironment.recycling_bin,),
            ),
            TRASH_BIN_EMPTY(
                state=task.initial_state,
                objects=(TossingRoomSplitIdentityEnvironment.trash_bin,),
            ),
        })
        assert (
            task.initial_state.get(
                obj=TossingRoomSplitIdentityEnvironment.recycling_bin, feature_name="count"
            )
            == 1.0
        )
        assert (
            task.initial_state.get(
                obj=TossingRoomSplitIdentityEnvironment.trash_bin, feature_name="count"
            )
            == 1.0
        )


def test_both_items_target_forces_are_within_the_resolved_range() -> None:
    """Each item's target is its own two causes resolved by the relation, so it lies in
    [0.1, 0.9] -- the span the causal arm's `required_force` occupies. Both items are
    checked, since `ThrowTrash` and `ThrowRecycling` each read only their own feature."""
    tasks = _tasks(seed=3)
    low = tasks.target_force(throw_distance=tasks.distance_low, item_weight=tasks.weight_low)
    high = tasks.target_force(throw_distance=tasks.distance_high, item_weight=tasks.weight_high)
    assert (low, high) == pytest.approx((0.1, 0.9))
    for _ in range(20):
        task = tasks.sample_train_task()
        for item in (
            TossingRoomSplitIdentityEnvironment.trash,
            TossingRoomSplitIdentityEnvironment.recycling,
        ):
            target = task.initial_state.get(obj=item, feature_name="target_force")
            assert low <= target <= high


def test_the_resolved_range_leaves_every_winning_window_unclipped() -> None:
    """`sample_params` draws a force from U(0, 1), so a task's winning window
    [target - tolerance, target + tolerance] must sit wholly inside that band or the task
    is quietly harder than the causal arm's matching one. The pre-#80 design drew
    `target_force` from [0.5, 1.0) and failed this at the top of its range.

    Asserted on the *bounds* the cause ranges resolve to rather than on a sample, so
    widening a range in config fails here rather than in a mis-calibrated experiment."""
    tasks = _tasks(seed=0)
    tolerance = tasks.env.throw_tolerance
    low = tasks.target_force(throw_distance=tasks.distance_low, item_weight=tasks.weight_low)
    high = tasks.target_force(throw_distance=tasks.distance_high, item_weight=tasks.weight_high)
    # The bound is exact rather than slack -- the resolved range IS
    # [tolerance, 1 - tolerance], the same span the causal arm's required_force occupies
    # -- so this compares up to floating-point error rather than demanding a strict
    # inequality of a value that lands exactly on the boundary.
    assert low == pytest.approx(tolerance) or low > tolerance
    assert high == pytest.approx(1.0 - tolerance) or high < 1.0 - tolerance


def test_seed_is_deterministic() -> None:
    first = _tasks(
        seed=42, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    second = _tasks(
        seed=42, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    assert _recycling_target(task=first) == _recycling_target(task=second)


def test_different_seeds_change_the_sampled_target_force() -> None:
    a = _tasks(
        seed=1, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    b = _tasks(
        seed=2, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING
    ).sample_train_task()
    assert _recycling_target(task=a) != _recycling_target(task=b)


def test_train_and_test_use_independent_streams() -> None:
    """Disjointness keyed on a continuous per-task draw, not goal type -- with only three
    discrete goal types a goal-type comparison would collide ~1/3 of the time (advisor's
    item 4)."""
    tasks = _tasks(seed=7, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING)
    train_target = _recycling_target(task=tasks.sample_train_task())
    test_target = _recycling_target(task=tasks.sample_test_task())
    assert train_target != test_target


def test_set_seed_rederives_both_streams() -> None:
    tasks = _tasks(seed=5, forced_goal_type=TossingRoomSplitIdentityGoalType.RECYCLING)
    first_train = _recycling_target(task=tasks.sample_train_task())
    first_test = _recycling_target(task=tasks.sample_test_task())
    tasks.set_seed(seed=5)
    assert _recycling_target(task=tasks.sample_train_task()) == first_train
    assert _recycling_target(task=tasks.sample_test_task()) == first_test


def test_default_mix_produces_more_than_one_goal_type() -> None:
    tasks = _tasks(seed=0)
    seen = set()
    for _ in range(50):
        task = tasks.sample_train_task()
        predicates = {atom.predicate for atom in task.goal.atoms}
        seen.add(frozenset(p.name for p in predicates))
    assert len(seen) > 1


# --- The fixed test-set composition (see TossingRoomSplitIdentityTasks' class docstring) ---


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_test_set_composition_is_exactly_14_14_2_at_30_tasks(*, seed: int) -> None:
    """The headline guarantee: realised counts, not expected ones, on every seed."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    assert Counter(_test_goal_types(tasks=tasks, count=30)) == {
        TossingRoomSplitIdentityGoalType.TRASH: 14,
        TossingRoomSplitIdentityGoalType.RECYCLING: 14,
        TossingRoomSplitIdentityGoalType.EMPTY: 2,
    }


@pytest.mark.parametrize("num_test_tasks", [1, 2, 3, 5, 10, 11, 30])
def test_composition_rule_is_well_formed_at_any_test_set_size(*, num_test_tasks: int) -> None:
    counts = _tasks(seed=0, num_test_tasks=num_test_tasks).test_goal_type_counts()
    assert sum(counts.values()) == num_test_tasks
    assert all(count >= 0 for count in counts.values())
    # EMPTY must never crowd out a throw family: it is the trivially-solved one, so an
    # all-EMPTY (or single-throw-family) test set would report a meaningless score.
    if num_test_tasks >= 2:
        assert counts[TossingRoomSplitIdentityGoalType.TRASH] >= 1
        assert counts[TossingRoomSplitIdentityGoalType.RECYCLING] >= 1


def test_composition_matches_the_documented_rule_at_key_sizes() -> None:
    assert _tasks(seed=0, num_test_tasks=30).test_goal_type_counts() == {
        TossingRoomSplitIdentityGoalType.TRASH: 14,
        TossingRoomSplitIdentityGoalType.RECYCLING: 14,
        TossingRoomSplitIdentityGoalType.EMPTY: 2,
    }
    assert _tasks(seed=0, num_test_tasks=10).test_goal_type_counts() == {
        TossingRoomSplitIdentityGoalType.TRASH: 4,
        TossingRoomSplitIdentityGoalType.RECYCLING: 4,
        TossingRoomSplitIdentityGoalType.EMPTY: 2,
    }
    # Below min_test_tasks_per_empty there is no room for an EMPTY sanity task, and the
    # odd leftover task breaks toward TRASH.
    assert _tasks(seed=0, num_test_tasks=3).test_goal_type_counts() == {
        TossingRoomSplitIdentityGoalType.TRASH: 2,
        TossingRoomSplitIdentityGoalType.RECYCLING: 1,
        TossingRoomSplitIdentityGoalType.EMPTY: 0,
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
        seed=0, num_test_tasks=30, forced_goal_type=TossingRoomSplitIdentityGoalType.EMPTY
    )
    assert set(_test_goal_types(tasks=tasks, count=10)) == {TossingRoomSplitIdentityGoalType.EMPTY}


@pytest.mark.parametrize(
    ("seed", "expected_goal_types", "expected_first_target"),
    [
        (
            0,
            "trash empty empty recycling recycling recycling trash recycling trash empty "
            "trash recycling",
            0.441698,
        ),
        (
            1,
            "trash trash trash trash trash trash trash empty trash trash trash recycling",
            0.282396,
        ),
        (
            2,
            "recycling trash trash trash recycling empty recycling trash empty recycling "
            "trash empty",
            0.665731,
        ),
    ],
)
def test_train_stream_is_pinned_for_this_arm(
    *,
    seed: int,
    expected_goal_types: str,
    expected_first_target: float,
) -> None:
    """Golden values pinning the training stream, so a later edit cannot move it without
    saying so -- pinning training tasks has silently turned a run into a different
    experiment on this project before.

    **These literals ARE the causal arm's**, and that is the point: `build_task` here
    draws the same four uniforms per task, from the same ranges, in the same order, then
    resolves each item's pair into the target the State carries. So the two arms consume
    `train_rng` in lockstep and present the same goal-family sequence with the same
    required force, task for task. `test_fork_equivalence.py` asserts that agreement
    directly; these golden values are the second lock on it, and a divergence here means
    the arms have stopped being paired."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    drawn = [tasks.sample_train_task() for _ in range(12)]
    assert [_goal_type(task=task).value for task in drawn] == expected_goal_types.split()
    assert _recycling_target(task=drawn[0]) == pytest.approx(expected_first_target, abs=1e-6)


def test_train_stream_still_samples_its_goal_types_rather_than_fixing_them() -> None:
    """The counterpart of the test-set guarantee: train family counts stay seed-dependent
    (which is exactly what the test set no longer is)."""
    counts_per_seed = set()
    for seed in range(5):
        tasks = _tasks(seed=seed)
        drawn = Counter(_goal_type(task=tasks.sample_train_task()).value for _ in range(30))
        counts_per_seed.add(tuple(sorted(drawn.items())))
    assert len(counts_per_seed) > 1
