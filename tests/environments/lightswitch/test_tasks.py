from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import LIGHT_ON
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _assert_valid_task(*, task: Task) -> None:
    robot = LightSwitchEnvironment.robot
    light = LightSwitchEnvironment.light
    assert task.initial_state.get(obj=robot, feature_name="x") == 0.5
    assert task.initial_state.get(obj=light, feature_name="level") == 0.0
    target = task.initial_state.get(obj=light, feature_name="target")
    assert 0.5 <= target <= 1.0


def test_sample_train_task_has_canonical_start_and_randomized_target() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment())
    _assert_valid_task(task=tasks.sample_train_task())


def test_sample_test_task_has_canonical_start_and_randomized_target() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment())
    _assert_valid_task(task=tasks.sample_test_task())


def test_sample_train_task_goal_is_light_on_for_the_light_object() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment())
    task = tasks.sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == LIGHT_ON
    assert atom.objects == (LightSwitchEnvironment.light,)


def test_sample_train_task_initial_state_does_not_already_satisfy_goal() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment())
    task = tasks.sample_train_task()
    assert task.goal.is_satisfied(state=task.initial_state) is False


def test_train_and_test_targets_vary_across_samples() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment())
    targets = {
        tasks.sample_train_task().initial_state.get(
            obj=LightSwitchEnvironment.light, feature_name="target"
        )
        for _ in range(10)
    }
    assert len(targets) > 1


def test_seed_is_deterministic() -> None:
    first = LightSwitchTasks(env=LightSwitchEnvironment(), seed=42).sample_train_task()
    second = LightSwitchTasks(env=LightSwitchEnvironment(), seed=42).sample_train_task()
    assert first.initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    ) == second.initial_state.get(obj=LightSwitchEnvironment.light, feature_name="target")


def test_different_seeds_change_sampled_targets() -> None:
    a = LightSwitchTasks(env=LightSwitchEnvironment(), seed=1).sample_train_task()
    b = LightSwitchTasks(env=LightSwitchEnvironment(), seed=2).sample_train_task()
    assert a.initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    ) != b.initial_state.get(obj=LightSwitchEnvironment.light, feature_name="target")


def test_set_seed_rederives_both_streams_deterministically() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment(), seed=7)
    first_train = tasks.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    first_test = tasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    tasks.set_seed(seed=7)
    second_train = tasks.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    second_test = tasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    assert first_train == second_train
    assert first_test == second_test


def test_train_and_test_use_independent_rng_streams() -> None:
    tasks = LightSwitchTasks(env=LightSwitchEnvironment(), seed=3)
    train_target = tasks.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    test_target = tasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    assert train_target != test_target


def test_two_instances_with_the_same_seed_produce_independent_but_equal_streams() -> None:
    """The whole point of this refactor: no shared ClassVar RNG anymore -- two
    separately-constructed Tasks with the same seed don't step on each other."""
    a = LightSwitchTasks(env=LightSwitchEnvironment(), seed=99)
    b = LightSwitchTasks(env=LightSwitchEnvironment(), seed=99)
    a.sample_train_task()  # advances a.train_rng only
    assert a.train_rng is not b.train_rng
    a_target = a.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    b_target = b.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )
    # b hasn't been advanced by a's extra sample_train_task call above, so its first
    # *and* second draws differ from a's second draw.
    assert a_target != b_target
