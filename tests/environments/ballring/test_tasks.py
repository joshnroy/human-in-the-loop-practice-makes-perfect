import numpy as np

from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.predicates import BALL_ON_TABLE
from hitl_pmp.environments.ballring.tasks import BallRingTasks

E = BallRingEnvironment


def _target_table_x(*, task: Task) -> float:
    tables = E.get_tables(state=task.initial_state)
    return task.initial_state.get(obj=tables[-1], feature_name="x")


def test_sample_train_task_goal_is_ball_on_the_sticky_target_table() -> None:
    task = BallRingTasks(env=E()).sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == BALL_ON_TABLE
    ball, target = atom.objects
    assert ball == E.ball
    assert target.name == "sticky-table-0"


def test_sample_train_task_is_not_already_solved() -> None:
    task = BallRingTasks(env=E()).sample_train_task()
    assert task.goal.is_satisfied(state=task.initial_state) is False


def test_sample_test_task_goal_is_ball_on_target_table() -> None:
    task = BallRingTasks(env=E()).sample_test_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == BALL_ON_TABLE
    assert task.goal.is_satisfied(state=task.initial_state) is False


def test_seed_is_deterministic() -> None:
    a = BallRingTasks(env=E(), seed=42).sample_train_task()
    b = BallRingTasks(env=E(), seed=42).sample_train_task()
    for obj in a.initial_state.data:
        assert np.array_equal(a.initial_state[obj], b.initial_state[obj])


def test_different_seeds_change_the_sampled_state() -> None:
    a = BallRingTasks(env=E(), seed=1).sample_train_task()
    b = BallRingTasks(env=E(), seed=2).sample_train_task()
    assert a.initial_state.get(obj=E.ball, feature_name="x") != b.initial_state.get(
        obj=E.ball, feature_name="x"
    )


def test_train_and_test_streams_are_independent() -> None:
    tasks = BallRingTasks(env=E(), seed=3)
    train = tasks.sample_train_task().initial_state.get(obj=E.ball, feature_name="x")
    test = tasks.sample_test_task().initial_state.get(obj=E.ball, feature_name="x")
    assert train != test


def test_test_stream_uses_the_seed_offset() -> None:
    """The test stream is seed + test_env_seed_offset, so one Tasks' test draw
    equals another Tasks' *train* draw when that offset is baked into the seed."""
    tasks = BallRingTasks(env=E(), seed=0, test_env_seed_offset=10000)
    test_task = tasks.sample_test_task()
    shifted = BallRingTasks(env=E(), seed=10000)
    train_task = shifted.sample_train_task()
    test_x = test_task.initial_state.get(obj=E.ball, feature_name="x")
    assert test_x == train_task.initial_state.get(obj=E.ball, feature_name="x")


def test_set_seed_rederives_both_streams_deterministically() -> None:
    tasks = BallRingTasks(env=E(), seed=7)
    first_train = tasks.sample_train_task().initial_state.get(obj=E.ball, feature_name="x")
    first_test = tasks.sample_test_task().initial_state.get(obj=E.ball, feature_name="x")
    tasks.set_seed(seed=7)
    assert tasks.sample_train_task().initial_state.get(obj=E.ball, feature_name="x") == first_train
    assert tasks.sample_test_task().initial_state.get(obj=E.ball, feature_name="x") == first_test
