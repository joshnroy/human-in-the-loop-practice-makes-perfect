import pytest

from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import BIN_EMPTY, ITEM_IN_BIN
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks


def _tasks(**kwargs) -> TossingRoomTasks:
    return TossingRoomTasks(env=TossingRoomEnvironment(), **kwargs)


def _recycling_target(*, task) -> float:
    return task.initial_state.get(obj=TossingRoomEnvironment.recycling, feature_name="target_force")


@pytest.mark.parametrize("goal_type", list(TossingRoomGoalType))
def test_forced_goal_type_initial_state_never_already_satisfies_goal(
    *, goal_type: TossingRoomGoalType
) -> None:
    tasks = _tasks(seed=0, forced_goal_type=goal_type)
    for _ in range(10):
        task = tasks.sample_train_task()
        assert task.goal.is_satisfied(state=task.initial_state) is False


def test_recycling_goal_is_a_single_item_in_recycling_bin_atom() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == ITEM_IN_BIN
    assert atom.objects == (TossingRoomEnvironment.recycling, TossingRoomEnvironment.recycling_bin)


def test_trash_goal_is_a_single_item_in_trash_bin_atom() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomGoalType.TRASH).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == ITEM_IN_BIN
    assert atom.objects == (TossingRoomEnvironment.trash, TossingRoomEnvironment.trash_bin)


def test_throw_tasks_start_with_empty_bins() -> None:
    task = _tasks(seed=1, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    assert (
        task.initial_state.get(obj=TossingRoomEnvironment.recycling_bin, feature_name="count")
        == 0.0
    )


def test_empty_goal_starts_with_both_bins_non_empty() -> None:
    task = _tasks(seed=2, forced_goal_type=TossingRoomGoalType.EMPTY).sample_train_task()
    assert task.goal.atoms == frozenset({
        BIN_EMPTY(state=task.initial_state, objects=(TossingRoomEnvironment.recycling_bin,)),
        BIN_EMPTY(state=task.initial_state, objects=(TossingRoomEnvironment.trash_bin,)),
    })
    assert (
        task.initial_state.get(obj=TossingRoomEnvironment.recycling_bin, feature_name="count")
        >= 1.0
    )
    assert task.initial_state.get(obj=TossingRoomEnvironment.trash_bin, feature_name="count") >= 1.0


def test_target_force_is_within_the_sampled_range() -> None:
    tasks = _tasks(seed=3)
    for _ in range(20):
        target = _recycling_target(task=tasks.sample_train_task())
        assert tasks.target_low <= target <= tasks.target_high


def test_seed_is_deterministic() -> None:
    first = _tasks(seed=42, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    second = _tasks(seed=42, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    assert _recycling_target(task=first) == _recycling_target(task=second)


def test_different_seeds_change_the_sampled_target() -> None:
    a = _tasks(seed=1, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    b = _tasks(seed=2, forced_goal_type=TossingRoomGoalType.RECYCLING).sample_train_task()
    assert _recycling_target(task=a) != _recycling_target(task=b)


def test_train_and_test_use_independent_streams() -> None:
    """Disjointness keyed on the continuous target_force, not goal type -- with only
    three discrete goal types a goal-type comparison would collide ~1/3 of the time
    (advisor's item 4)."""
    tasks = _tasks(seed=7, forced_goal_type=TossingRoomGoalType.RECYCLING)
    train_target = _recycling_target(task=tasks.sample_train_task())
    test_target = _recycling_target(task=tasks.sample_test_task())
    assert train_target != test_target


def test_set_seed_rederives_both_streams() -> None:
    tasks = _tasks(seed=5, forced_goal_type=TossingRoomGoalType.RECYCLING)
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
