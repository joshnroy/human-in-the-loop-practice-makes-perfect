import numpy as np
import pytest

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.oracle_policy import ORACLE_POLICY
from hitl_pmp.environments.lightswitch.predicates import LIGHT_ON
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _compute_never_moves_action(*, state: State) -> Action:
    del state
    return np.array([0.0, 0.0])


_never_moves_policy: Policy = lambda state: _compute_never_moves_action(state=state)  # noqa: E731


def test_run_task_episode_requires_wiring_first() -> None:
    for attr in ("env", "tasks"):
        if attr in vars(Problem):
            delattr(Problem, attr)

    task = LightSwitchTasks.sample_train_task()
    with pytest.raises(AttributeError):
        LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY)


def test_run_task_episode_succeeds_immediately_for_an_already_satisfied_task() -> None:
    LightSwitchProblem.wire()
    initial_state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))

    assert LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy) is True


def test_run_task_episode_succeeds_with_the_oracle_policy() -> None:
    LightSwitchProblem.wire()
    task = LightSwitchTasks.sample_train_task()
    assert LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY) is True


def test_run_task_episode_fails_when_policy_never_solves_it() -> None:
    LightSwitchProblem.wire()
    task = LightSwitchTasks.sample_train_task()
    assert LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy) is False


def test_run_task_episode_sets_env_state_from_task_initial_state() -> None:
    LightSwitchProblem.wire()
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    task = LightSwitchTasks.sample_train_task()

    LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy)

    robot = LightSwitchEnvironment.robot
    assert LightSwitchEnvironment.get_current_state().get(
        obj=robot, feature_name="x"
    ) == task.initial_state.get(obj=robot, feature_name="x")
