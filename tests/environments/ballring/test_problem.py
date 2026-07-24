from typing import ClassVar

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.environments.ballring.environment import BallRingEnvironment
from hitl_pmp.environments.ballring.predicates import BALL_ON_TABLE
from hitl_pmp.environments.ballring.problem import BallRingProblem
from hitl_pmp.environments.ballring.tasks import BallRingTasks

E = BallRingEnvironment


def _compute_never_moves(*, state: State) -> LabeledAction:
    del state
    return LabeledAction(action=np.array([0.0, 0.0, 0.0, 0.5, 0.5]), label="stay")


_never_moves_policy: Policy = lambda state: _compute_never_moves(state=state)  # noqa: E731


def _place_cup_on_target(*, state: State) -> LabeledAction:
    del state
    return LabeledAction(action=np.array([1.0, 3.0, 0.0, 0.5, 0.5]), label="place cup on target")


_solve_in_one_policy: Policy = lambda state: _place_cup_on_target(state=state)  # noqa: E731


class _LabelSpyRenderer(Renderer):
    received_labels: ClassVar[list[str | None]] = []

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del state, env
        _LabelSpyRenderer.received_labels.append(label)
        return np.zeros((1, 1, 3), dtype=np.uint8)


def _target_table() -> Object:
    return Object(name="sticky-table-0", type=E.table_type)


def _one_step_solve_task() -> Task:
    """Robot at a sticky target table holding a cup with the ball inside; a single
    place onto the safe sticky region carries the ball onto the table (goal met)."""
    target = _target_table()
    data: dict[Object, np.ndarray] = {
        target: np.array([0.5, 0.5, 0.1, 1.0, 0.0, 0.0, 0.05]),  # safe region at center
        E.robot: np.array([0.55, 0.5]),  # reachable to the table
        E.ball: np.array([0.5, 0.5, 0.02, 1.0]),  # held, inside cup
        E.cup: np.array([0.5, 0.5, 0.03, 1.0]),  # held
    }
    state = State(data=data)
    goal = Goal(atoms=frozenset({BALL_ON_TABLE(state=state, objects=(E.ball, target))}))
    return Task(initial_state=state, goal=goal)


def _already_solved_task() -> Task:
    target = _target_table()
    data: dict[Object, np.ndarray] = {
        target: np.array([0.5, 0.5, 0.1, 1.0, 0.0, 0.0, 0.05]),
        E.robot: np.array([0.1, 0.1]),
        E.ball: np.array([0.5, 0.5, 0.02, 0.0]),  # resting on the target table
        E.cup: np.array([0.9, 0.9, 0.03, 0.0]),
    }
    state = State(data=data)
    goal = Goal(atoms=frozenset({BALL_ON_TABLE(state=state, objects=(E.ball, target))}))
    return Task(initial_state=state, goal=goal)


def _build_problem() -> BallRingProblem:
    env = E()
    return BallRingProblem(env=env, tasks=BallRingTasks(env=env))


def test_problem_requires_its_own_domains_env_and_tasks_types() -> None:
    with pytest.raises(ValueError, match="env"):
        BallRingProblem(env=object(), tasks=BallRingTasks(env=E()))  # type: ignore[arg-type]


def test_max_episode_steps_matches_the_papers_horizon() -> None:
    assert _build_problem().max_episode_steps() == 8


def test_run_task_episode_succeeds_immediately_for_an_already_satisfied_task() -> None:
    problem = _build_problem()
    solved, frames = problem.run_task_episode(
        task=_already_solved_task(), policy=_never_moves_policy
    )
    assert solved is True
    assert frames == []


def test_run_task_episode_solves_within_the_loop() -> None:
    problem = _build_problem()
    solved, frames = problem.run_task_episode(
        task=_one_step_solve_task(), policy=_solve_in_one_policy
    )
    assert solved is True
    assert frames == []


def test_run_task_episode_fails_when_policy_never_solves() -> None:
    problem = _build_problem()
    solved, _ = problem.run_task_episode(task=_one_step_solve_task(), policy=_never_moves_policy)
    assert solved is False


def test_run_task_episode_sets_env_state_from_task_initial_state() -> None:
    problem = _build_problem()
    task = problem.tasks.sample_train_task()
    problem.run_task_episode(task=task, policy=_never_moves_policy)
    current_x = problem.env.get_current_state().get(obj=E.ball, feature_name="x")
    assert current_x == task.initial_state.get(obj=E.ball, feature_name="x")


def test_run_task_episode_captures_no_frames_without_a_renderer() -> None:
    problem = _build_problem()
    _, frames = problem.run_task_episode(task=_one_step_solve_task(), policy=_solve_in_one_policy)
    assert frames == []


def test_run_task_episode_captures_one_frame_per_step_with_a_renderer() -> None:
    problem = _build_problem()
    solved, frames = problem.run_task_episode(
        task=_one_step_solve_task(), policy=_solve_in_one_policy, renderer=_LabelSpyRenderer
    )
    assert solved is True
    # Initial frame + one action frame, then solved.
    assert len(frames) == 2
    for frame in frames:
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_run_task_episode_forwards_labels_to_the_renderer() -> None:
    _LabelSpyRenderer.received_labels = []
    problem = _build_problem()
    problem.run_task_episode(
        task=_one_step_solve_task(), policy=_solve_in_one_policy, renderer=_LabelSpyRenderer
    )
    assert _LabelSpyRenderer.received_labels[0] is None
    assert _LabelSpyRenderer.received_labels[1] == "place cup on target"
