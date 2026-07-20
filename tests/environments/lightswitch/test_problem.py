import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.oracle_policy import ORACLE_POLICY
from hitl_pmp.environments.lightswitch.predicates import LIGHT_ON
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _compute_never_moves_action(*, state: State) -> Action:
    del state
    return np.array([0.0, 0.0])


_never_moves_policy: Policy = lambda state: _compute_never_moves_action(state=state)  # noqa: E731


def test_light_switch_problem_is_wired_to_its_own_domain_by_default() -> None:
    assert LightSwitchProblem.env is LightSwitchEnvironment
    assert LightSwitchProblem.tasks is LightSwitchTasks


def test_max_episode_steps_matches_the_papers_horizon_formula() -> None:
    original_grid_size = LightSwitchEnvironment.grid_size
    try:
        LightSwitchEnvironment.grid_size = 7
        assert LightSwitchProblem.max_episode_steps() == 9
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size


def test_run_task_episode_succeeds_immediately_for_an_already_satisfied_task() -> None:
    initial_state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))

    solved, frames = LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy)
    assert solved is True
    assert frames == []


def test_run_task_episode_succeeds_with_the_oracle_policy() -> None:
    task = LightSwitchTasks.sample_train_task()
    solved, frames = LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY)
    assert solved is True
    assert frames == []


def test_run_task_episode_fails_when_policy_never_solves_it() -> None:
    task = LightSwitchTasks.sample_train_task()
    solved, frames = LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy)
    assert solved is False
    assert frames == []


def test_run_task_episode_sets_env_state_from_task_initial_state() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    task = LightSwitchTasks.sample_train_task()

    LightSwitchProblem.run_task_episode(task=task, policy=_never_moves_policy)

    robot = LightSwitchEnvironment.robot
    assert LightSwitchEnvironment.get_current_state().get(
        obj=robot, feature_name="x"
    ) == task.initial_state.get(obj=robot, feature_name="x")


def test_run_task_episode_respects_a_smaller_grid_size_at_call_time() -> None:
    original_grid_size = LightSwitchEnvironment.grid_size
    try:
        LightSwitchEnvironment.grid_size = 3
        task = LightSwitchTasks.sample_train_task()
        solved, _ = LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY)
        assert solved is True
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size


def test_run_task_episode_captures_no_frames_without_a_renderer() -> None:
    task = LightSwitchTasks.sample_train_task()
    _, frames = LightSwitchProblem.run_task_episode(task=task, policy=ORACLE_POLICY)
    assert frames == []


def test_run_task_episode_captures_one_frame_per_step_with_a_renderer() -> None:
    task = LightSwitchTasks.sample_train_task()
    solved, frames = LightSwitchProblem.run_task_episode(
        task=task, policy=ORACLE_POLICY, renderer=LightSwitchRenderer
    )
    assert solved is True
    # Oracle always solves in exactly 2 actions: initial frame + one per action.
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_run_task_episode_renderer_frames_stop_once_goal_is_satisfied() -> None:
    initial_state = LightSwitchEnvironment.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))

    solved, frames = LightSwitchProblem.run_task_episode(
        task=task, policy=_never_moves_policy, renderer=LightSwitchRenderer
    )
    assert solved is True
    # Already satisfied at t=0: only the initial frame, no actions taken.
    assert len(frames) == 1
