import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.problem import TossingRoomProblem
from hitl_pmp.environments.tossingroom.renderer import TossingRoomRenderer
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _no_op_action(*, state) -> LabeledAction:
    del state
    return LabeledAction(
        action=np.array([TossingRoomEnvironment.SKILL_PRESS, 0.0, 0.0]),
        label="press (no-op at start)",
    )


_never_solves_policy: Policy = lambda state: _no_op_action(state=state)  # noqa: E731


def _build_problem(*, num_rooms: int = 7, goal_type: TossingRoomGoalType | None = None):
    env = TossingRoomEnvironment(num_rooms=num_rooms)
    tasks = TossingRoomTasks(env=env, forced_goal_type=goal_type)
    return TossingRoomProblem(env=env, tasks=tasks)


def _oracle_policy(*, problem, task) -> Policy:
    return SkillOracleMethod(env=problem.env).get_task_policy(task=task)


def test_problem_requires_its_own_domains_types() -> None:
    with pytest.raises(ValueError, match="env"):
        TossingRoomProblem(
            env=object(),  # type: ignore[arg-type]
            tasks=TossingRoomTasks(env=TossingRoomEnvironment()),
        )


def test_max_episode_steps_scales_with_num_rooms() -> None:
    assert _build_problem(num_rooms=7).max_episode_steps() == 16


def test_run_task_episode_solves_a_recycling_task_with_the_oracle() -> None:
    problem = _build_problem(goal_type=TossingRoomGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, frames = problem.run_task_episode(
        task=task, policy=_oracle_policy(problem=problem, task=task)
    )
    assert solved is True
    assert frames == []


def test_run_task_episode_fails_when_the_policy_never_solves() -> None:
    problem = _build_problem(goal_type=TossingRoomGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, _ = problem.run_task_episode(task=task, policy=_never_solves_policy)
    assert solved is False


def test_run_task_episode_records_a_frame_per_step_with_a_renderer() -> None:
    problem = _build_problem(goal_type=TossingRoomGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, frames = problem.run_task_episode(
        task=task, policy=_oracle_policy(problem=problem, task=task), renderer=TossingRoomRenderer
    )
    assert solved is True
    assert len(frames) >= 2  # at least the initial frame plus one action
    for frame in frames:
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_run_task_episode_sets_env_state_from_the_task_initial_state() -> None:
    problem = _build_problem(goal_type=TossingRoomGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    problem.run_task_episode(task=task, policy=_never_solves_policy)
    assert problem.env.get_current_state().get(
        obj=TossingRoomEnvironment.robot, feature_name="room"
    ) == task.initial_state.get(obj=TossingRoomEnvironment.robot, feature_name="room")
