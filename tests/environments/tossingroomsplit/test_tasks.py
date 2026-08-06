from collections import Counter

import pytest

from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplit.tasks import (
    TossingRoomSplitGoalType,
    TossingRoomSplitTasks,
)


def _tasks(**kwargs) -> TossingRoomSplitTasks:
    return TossingRoomSplitTasks(env=TossingRoomSplitEnvironment(), **kwargs)


def _recycling_weight(*, task) -> float:
    """One of the two per-task CAUSES of the throw force, standing in for "the continuous
    value this task drew" throughout this file. There is no `target_force` feature any
    more -- see tests/environments/tossingroomsplit/test_throw_representation.py."""
    return task.initial_state.get(obj=TossingRoomSplitEnvironment.recycling, feature_name="weight")


def _recycling_distance(*, task) -> float:
    return task.initial_state.get(
        obj=TossingRoomSplitEnvironment.recycling_bin, feature_name="throw_distance"
    )


def _goal_type(*, task) -> TossingRoomSplitGoalType:
    """Which goal family a built Task belongs to, read back off its goal atoms -- the
    same inference an analysis script reading a run's tasks back would make."""
    if all(atom.predicate in (TRASH_BIN_EMPTY, RECYCLING_BIN_EMPTY) for atom in task.goal.atoms):
        return TossingRoomSplitGoalType.EMPTY
    (atom,) = task.goal.atoms
    # The in-bin predicate is split per kind here, so the family is read off the
    # PREDICATE rather than off the bound item -- one of the few genuine test diffs
    # against Tossing Room.
    if atom.predicate == RECYCLING_IN_BIN:
        return TossingRoomSplitGoalType.RECYCLING
    assert atom.predicate == TRASH_IN_BIN
    return TossingRoomSplitGoalType.TRASH


def _test_goal_types(*, tasks: TossingRoomSplitTasks, count: int) -> list[TossingRoomSplitGoalType]:
    return [_goal_type(task=tasks.sample_test_task()) for _ in range(count)]


@pytest.mark.parametrize("goal_type", list(TossingRoomSplitGoalType))
def test_forced_goal_type_initial_state_never_already_satisfies_goal(
    *, goal_type: TossingRoomSplitGoalType
) -> None:
    tasks = _tasks(seed=0, forced_goal_type=goal_type)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.is_satisfied(state=task.initial_state) is False


def test_recycling_goal_is_a_single_item_in_recycling_bin_atom() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomSplitGoalType.RECYCLING).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == RECYCLING_IN_BIN
    assert atom.objects == (
        TossingRoomSplitEnvironment.recycling,
        TossingRoomSplitEnvironment.recycling_bin,
    )


def test_trash_goal_is_a_single_item_in_trash_bin_atom() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomSplitGoalType.TRASH).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == TRASH_IN_BIN
    assert atom.objects == (
        TossingRoomSplitEnvironment.trash,
        TossingRoomSplitEnvironment.trash_bin,
    )


def test_throw_tasks_start_with_empty_bins() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomSplitGoalType.RECYCLING).sample_train_task()
    assert (
        task.initial_state.get(obj=TossingRoomSplitEnvironment.recycling_bin, feature_name="count")
        == 0.0
    )


def test_empty_goal_prefills_exactly_one_item_per_bin() -> None:
    """A bin holds at most one item, so the old Uniform{1, 2, 3} prefill (and its
    initial_count_low/high knobs) has exactly one legal value left."""
    tasks = _tasks(seed=2, forced_goal_type=TossingRoomSplitGoalType.EMPTY)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.atoms == frozenset({
            RECYCLING_BIN_EMPTY(
                state=task.initial_state, objects=(TossingRoomSplitEnvironment.recycling_bin,)
            ),
            TRASH_BIN_EMPTY(
                state=task.initial_state, objects=(TossingRoomSplitEnvironment.trash_bin,)
            ),
        })
        assert (
            task.initial_state.get(
                obj=TossingRoomSplitEnvironment.recycling_bin, feature_name="count"
            )
            == 1.0
        )
        assert (
            task.initial_state.get(obj=TossingRoomSplitEnvironment.trash_bin, feature_name="count")
            == 1.0
        )


def test_both_throw_force_causes_are_within_their_sampled_ranges() -> None:
    tasks = _tasks(seed=3)
    for _ in range(20):
        task = tasks.sample_train_task()
        assert tasks.weight_low <= _recycling_weight(task=task) <= tasks.weight_high
        assert tasks.distance_low <= _recycling_distance(task=task) <= tasks.distance_high


def test_seed_is_deterministic() -> None:
    first = _tasks(seed=42, forced_goal_type=TossingRoomSplitGoalType.RECYCLING).sample_train_task()
    second = _tasks(
        seed=42, forced_goal_type=TossingRoomSplitGoalType.RECYCLING
    ).sample_train_task()
    assert _recycling_weight(task=first) == _recycling_weight(task=second)


def test_different_seeds_change_the_sampled_causes() -> None:
    a = _tasks(seed=1, forced_goal_type=TossingRoomSplitGoalType.RECYCLING).sample_train_task()
    b = _tasks(seed=2, forced_goal_type=TossingRoomSplitGoalType.RECYCLING).sample_train_task()
    assert _recycling_weight(task=a) != _recycling_weight(task=b)


def test_train_and_test_use_independent_streams() -> None:
    """Disjointness keyed on a continuous per-task draw, not goal type -- with only three
    discrete goal types a goal-type comparison would collide ~1/3 of the time (advisor's
    item 4)."""
    tasks = _tasks(seed=7, forced_goal_type=TossingRoomSplitGoalType.RECYCLING)
    train_weight = _recycling_weight(task=tasks.sample_train_task())
    test_weight = _recycling_weight(task=tasks.sample_test_task())
    assert train_weight != test_weight


def test_set_seed_rederives_both_streams() -> None:
    tasks = _tasks(seed=5, forced_goal_type=TossingRoomSplitGoalType.RECYCLING)
    first_train = _recycling_weight(task=tasks.sample_train_task())
    first_test = _recycling_weight(task=tasks.sample_test_task())
    tasks.set_seed(seed=5)
    assert _recycling_weight(task=tasks.sample_train_task()) == first_train
    assert _recycling_weight(task=tasks.sample_test_task()) == first_test


def test_default_mix_produces_more_than_one_goal_type() -> None:
    tasks = _tasks(seed=0)
    seen = set()
    for _ in range(50):
        task = tasks.sample_train_task()
        predicates = {atom.predicate for atom in task.goal.atoms}
        seen.add(frozenset(p.name for p in predicates))
    assert len(seen) > 1


# --- The fixed test-set composition (see TossingRoomSplitTasks' class docstring) ---


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_test_set_composition_is_exactly_14_14_2_at_30_tasks(*, seed: int) -> None:
    """The headline guarantee: realised counts, not expected ones, on every seed."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    assert Counter(_test_goal_types(tasks=tasks, count=30)) == {
        TossingRoomSplitGoalType.TRASH: 14,
        TossingRoomSplitGoalType.RECYCLING: 14,
        TossingRoomSplitGoalType.EMPTY: 2,
    }


@pytest.mark.parametrize("num_test_tasks", [1, 2, 3, 5, 10, 11, 30])
def test_composition_rule_is_well_formed_at_any_test_set_size(*, num_test_tasks: int) -> None:
    counts = _tasks(seed=0, num_test_tasks=num_test_tasks).test_goal_type_counts()
    assert sum(counts.values()) == num_test_tasks
    assert all(count >= 0 for count in counts.values())
    # EMPTY must never crowd out a throw family: it is the trivially-solved one, so an
    # all-EMPTY (or single-throw-family) test set would report a meaningless score.
    if num_test_tasks >= 2:
        assert counts[TossingRoomSplitGoalType.TRASH] >= 1
        assert counts[TossingRoomSplitGoalType.RECYCLING] >= 1


def test_composition_matches_the_documented_rule_at_key_sizes() -> None:
    assert _tasks(seed=0, num_test_tasks=30).test_goal_type_counts() == {
        TossingRoomSplitGoalType.TRASH: 14,
        TossingRoomSplitGoalType.RECYCLING: 14,
        TossingRoomSplitGoalType.EMPTY: 2,
    }
    assert _tasks(seed=0, num_test_tasks=10).test_goal_type_counts() == {
        TossingRoomSplitGoalType.TRASH: 4,
        TossingRoomSplitGoalType.RECYCLING: 4,
        TossingRoomSplitGoalType.EMPTY: 2,
    }
    # Below min_test_tasks_per_empty there is no room for an EMPTY sanity task, and the
    # odd leftover task breaks toward TRASH.
    assert _tasks(seed=0, num_test_tasks=3).test_goal_type_counts() == {
        TossingRoomSplitGoalType.TRASH: 2,
        TossingRoomSplitGoalType.RECYCLING: 1,
        TossingRoomSplitGoalType.EMPTY: 0,
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
    tasks = _tasks(seed=0, num_test_tasks=30, forced_goal_type=TossingRoomSplitGoalType.EMPTY)
    assert set(_test_goal_types(tasks=tasks, count=10)) == {TossingRoomSplitGoalType.EMPTY}


@pytest.mark.parametrize(
    ("seed", "expected_goal_types", "expected_first_weight", "expected_first_distance"),
    [
        (
            0,
            "trash empty empty recycling recycling recycling trash recycling trash empty "
            "trash recycling",
            0.540974,
            2.626540,
        ),
        (
            1,
            "trash trash trash trash trash trash trash empty trash trash trash recycling",
            0.644160,
            1.623663,
        ),
        (
            2,
            "recycling trash trash trash recycling empty recycling trash empty recycling "
            "trash empty",
            1.314226,
            2.200201,
        ),
    ],
)
def test_train_stream_is_unchanged_by_the_fixed_test_composition(
    *,
    seed: int,
    expected_goal_types: str,
    expected_first_weight: float,
    expected_first_distance: float,
) -> None:
    """Golden values pinning the training stream, which the fixed-test-set change
    deliberately left untouched -- pinning training tasks too has silently turned a run
    into a different experiment on this project before.

    RE-RECORDED ONCE BEFORE, for the capacity-1 bin redesign, and RE-RECORDED AGAIN here
    for the throw-representation change ported from Tossing Room. `build_task` now draws
    FOUR uniforms per task (two item weights, two bin distances) where it drew two
    `target_force`s, so every draw after the first shifts and the goal-family sequences
    move wholesale. That is a real change to the training distribution, not noise --
    **every Tossing Room (split) number produced after this change is measured on
    different tasks than every number produced before it**.

    **The literals are byte-identical to `tests/environments/tossingroom/test_tasks.py`'s,
    and that is the point**: the two domains are the same task distribution under two
    skill decompositions, so the split-throw experiment can be compared against Tossing
    Room's baseline. If these two files ever disagree, that comparison is invalid."""
    tasks = _tasks(seed=seed, num_test_tasks=30)
    drawn = [tasks.sample_train_task() for _ in range(12)]
    assert [_goal_type(task=task).value for task in drawn] == expected_goal_types.split()
    assert _recycling_weight(task=drawn[0]) == pytest.approx(expected_first_weight, abs=1e-6)
    assert _recycling_distance(task=drawn[0]) == pytest.approx(expected_first_distance, abs=1e-6)


def test_train_stream_still_samples_its_goal_types_rather_than_fixing_them() -> None:
    """The counterpart of the test-set guarantee: train family counts stay seed-dependent
    (which is exactly what the test set no longer is)."""
    counts_per_seed = set()
    for seed in range(5):
        tasks = _tasks(seed=seed)
        drawn = Counter(_goal_type(task=tasks.sample_train_task()).value for _ in range(30))
        counts_per_seed.add(tuple(sorted(drawn.items())))
    assert len(counts_per_seed) > 1
