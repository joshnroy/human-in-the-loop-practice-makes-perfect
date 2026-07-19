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
    task = LightSwitchTasks.sample_train_task()
    _assert_valid_task(task=task)


def test_sample_test_task_has_canonical_start_and_randomized_target() -> None:
    task = LightSwitchTasks.sample_test_task()
    _assert_valid_task(task=task)


def test_sample_train_task_goal_is_light_on_for_the_light_object() -> None:
    task = LightSwitchTasks.sample_train_task()
    (atom,) = task.goal.atoms
    assert atom.predicate == LIGHT_ON
    assert atom.objects == (LightSwitchEnvironment.light,)


def test_sample_train_task_initial_state_does_not_already_satisfy_goal() -> None:
    task = LightSwitchTasks.sample_train_task()
    assert task.goal.is_satisfied(state=task.initial_state) is False


def test_train_and_test_targets_vary_across_samples() -> None:
    targets = {
        LightSwitchTasks.sample_train_task().initial_state.get(
            obj=LightSwitchEnvironment.light, feature_name="target"
        )
        for _ in range(10)
    }
    assert len(targets) > 1


def test_train_and_test_use_independent_rng_streams() -> None:
    LightSwitchTasks.train_rng = LightSwitchTasks._make_rng(offset=0)
    LightSwitchTasks.test_rng = LightSwitchTasks._make_rng(
        offset=LightSwitchTasks.test_env_seed_offset
    )

    train_target = LightSwitchTasks.sample_train_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    LightSwitchTasks.train_rng = LightSwitchTasks._make_rng(offset=0)
    test_target = LightSwitchTasks.sample_test_task().initial_state.get(
        obj=LightSwitchEnvironment.light, feature_name="target"
    )

    assert train_target != test_target
