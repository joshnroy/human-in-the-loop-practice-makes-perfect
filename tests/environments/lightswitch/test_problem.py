from typing import ClassVar

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.environments.lightswitch.action_oracle_policy import ACTION_ORACLE_POLICY
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.predicates import LIGHT_ON
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def _compute_never_moves_action(*, state: State) -> LabeledAction:
    del state
    return LabeledAction(action=np.array([0.0, 0.0]), label="never moves")


_never_moves_policy: Policy = lambda state: _compute_never_moves_action(state=state)  # noqa: E731


class _LabelSpyRenderer(Renderer):
    """Records every label render_frame was called with, so a test can confirm
    run_task_episode forwards each LabeledAction.label through, in order, rather
    than rendering blind. A static-method container, never instantiated, same as
    every other business-logic class in this project."""

    received_labels: ClassVar[list[str | None]] = []

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del state, env
        _LabelSpyRenderer.received_labels.append(label)
        return np.zeros((1, 1, 3), dtype=np.uint8)


def _build_problem(*, grid_size: int = 100) -> LightSwitchProblem:
    env = LightSwitchEnvironment(grid_size=grid_size)
    return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env))


def test_light_switch_problem_requires_its_own_domains_environment_and_tasks_types() -> None:
    """env/tasks are still required fields (inherited from Problem), but narrowed to
    this domain's own concrete types -- passing a foreign Environment/Tasks
    subclass is a validation error, not silently accepted."""
    with pytest.raises(ValueError, match="env"):
        LightSwitchProblem(
            env=object(),  # type: ignore[arg-type]
            tasks=LightSwitchTasks(env=LightSwitchEnvironment()),
        )


def test_max_episode_steps_matches_the_papers_horizon_formula() -> None:
    problem = _build_problem(grid_size=7)
    assert problem.max_episode_steps() == 9


def test_run_task_episode_succeeds_immediately_for_an_already_satisfied_task() -> None:
    problem = _build_problem()
    initial_state = problem.env.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))

    solved, frames = problem.run_task_episode(task=task, policy=_never_moves_policy)
    assert solved is True
    assert frames == []


def test_run_task_episode_succeeds_with_the_oracle_policy() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    solved, frames = problem.run_task_episode(task=task, policy=ACTION_ORACLE_POLICY)
    assert solved is True
    assert frames == []


def test_run_task_episode_fails_when_policy_never_solves_it() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    solved, frames = problem.run_task_episode(task=task, policy=_never_moves_policy)
    assert solved is False
    assert frames == []


def test_run_task_episode_sets_env_state_from_task_initial_state() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()

    problem.run_task_episode(task=task, policy=_never_moves_policy)

    robot = LightSwitchEnvironment.robot
    assert problem.env.get_current_state().get(
        obj=robot, feature_name="x"
    ) == task.initial_state.get(obj=robot, feature_name="x")


def test_run_task_episode_respects_grid_size_from_construction() -> None:
    problem = _build_problem(grid_size=3)
    task = problem.tasks.sample_train_task()
    solved, _ = problem.run_task_episode(task=task, policy=ACTION_ORACLE_POLICY)
    assert solved is True


def test_run_task_episode_captures_no_frames_without_a_renderer() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    _, frames = problem.run_task_episode(task=task, policy=ACTION_ORACLE_POLICY)
    assert frames == []


def test_run_task_episode_captures_one_frame_per_step_with_a_renderer() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    solved, frames = problem.run_task_episode(
        task=task, policy=ACTION_ORACLE_POLICY, renderer=LightSwitchRenderer
    )
    assert solved is True
    # Oracle always solves in exactly 2 actions: initial frame + one per action.
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_run_task_episode_forwards_each_labeled_actions_label_to_the_renderer() -> None:
    _LabelSpyRenderer.received_labels = []
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    problem.run_task_episode(task=task, policy=ACTION_ORACLE_POLICY, renderer=_LabelSpyRenderer)
    # First frame is the initial state, before any action -- no label yet. Every
    # frame after that was produced by an action, so it carries that action's label.
    assert _LabelSpyRenderer.received_labels[0] is None
    assert all(label is not None for label in _LabelSpyRenderer.received_labels[1:])
    assert len(_LabelSpyRenderer.received_labels) == 3


def test_run_task_episode_renderer_frames_stop_once_goal_is_satisfied() -> None:
    problem = _build_problem()
    initial_state = problem.env.build_initial_state(light_level=0.5, light_target=0.5)
    light_on = LIGHT_ON(state=initial_state, objects=(LightSwitchEnvironment.light,))
    task = Task(initial_state=initial_state, goal=Goal(atoms=frozenset({light_on})))

    solved, frames = problem.run_task_episode(
        task=task, policy=_never_moves_policy, renderer=LightSwitchRenderer
    )
    assert solved is True
    # Already satisfied at t=0: only the initial frame, no actions taken.
    assert len(frames) == 1
