import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.renderer import TossingRoomSplitRenderer
from hitl_pmp.environments.tossingroomsplit.skill_provider import TossingRoomSplitOracle
from hitl_pmp.environments.tossingroomsplit.tasks import (
    TossingRoomSplitGoalType,
    TossingRoomSplitTasks,
)
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _no_op_action(*, state) -> LabeledAction:
    del state
    return LabeledAction(
        action=np.array([TossingRoomSplitEnvironment.SKILL_PRESS, 0.0, 0.0]),
        label="press (no-op at start)",
    )


_never_solves_policy: Policy = lambda state: _no_op_action(state=state)  # noqa: E731


def _build_problem(*, num_rooms: int = 7, goal_type: TossingRoomSplitGoalType | None = None):
    env = TossingRoomSplitEnvironment(num_rooms=num_rooms)
    tasks = TossingRoomSplitTasks(env=env, forced_goal_type=goal_type)
    return TossingRoomSplitProblem(env=env, tasks=tasks)


def _oracle_policy(*, problem, task) -> Policy:
    return SkillOracleMethod(
        env=problem.env, oracle=TossingRoomSplitOracle(env=problem.env)
    ).get_task_policy(task=task)


def test_problem_requires_its_own_domains_types() -> None:
    with pytest.raises(ValueError, match="env"):
        TossingRoomSplitProblem(
            env=object(),  # type: ignore[arg-type]
            tasks=TossingRoomSplitTasks(env=TossingRoomSplitEnvironment()),
        )


def test_max_episode_steps_is_the_longest_shortest_solve_plus_two() -> None:
    """The paper's H_eval convention (Appendix F), the same one
    LightSwitchProblem cites: exactly two spare actions beyond the longest solve
    this layout admits.

    On the default layout the longest is now EMPTY, because each bin has its own button
    beside it and the one-way ledge forces an ORDER: press the trash button (room 6)
    first, then cross down to the recycling one (room 1). That is three MoveRooms (3->6),
    a press, five MoveRooms (6->1), a press = 10, so the horizon is 12. It was 7 when a
    single button in room 6 emptied both bins.

    TRASH is unchanged at 5 (PickupTrash, 3->6, ThrowTrash) and still admits no second
    attempt: a retry costs the eight-step round trip back to the pile, i.e. step 13."""
    assert _build_problem().max_episode_steps() == 12


def test_max_episode_steps_tracks_the_layout_not_the_room_count() -> None:
    """Padding a layout with rooms nobody has to walk to must not buy extra
    attempts. This is the property the old `2 * num_rooms + 2` got wrong: it grew
    with rooms the robot never visits, and every extra step is a free retry of the
    one stochastic skill (Throw), which is what the evaluation is supposed to
    measure."""
    assert _build_problem(num_rooms=40).max_episode_steps() == 12


def test_max_episode_steps_grows_when_a_bin_moves_further_away() -> None:
    env = TossingRoomSplitEnvironment(num_rooms=12, trash_bin_room=11)
    problem = TossingRoomSplitProblem(env=env, tasks=TossingRoomSplitTasks(env=env))
    # EMPTY: 8 MoveRooms (3->11) + a press + 10 MoveRooms (11->1) + a press = 20, plus two.
    assert problem.max_episode_steps() == 22


def test_max_episode_steps_ignores_a_target_the_one_way_ledge_makes_unreachable() -> None:
    """A room behind the ledge in the blocked direction cannot be reached at any
    horizon, so it must not inflate the budget for the goals that *are* solvable --
    including EMPTY, which needs BOTH buttons and is unsolvable when either bin's button
    sits behind the ledge."""
    env = TossingRoomSplitEnvironment(
        num_rooms=12, start_room=1, blocked_right_from=2, trash_bin_room=11
    )
    problem = TossingRoomSplitProblem(env=env, tasks=TossingRoomSplitTasks(env=env))
    # Only the recycling bin (room 1, distance 0) is reachable: Pickup + Throw = 2.
    assert problem.max_episode_steps() == 4


def test_run_task_episode_solves_a_recycling_task_with_the_oracle() -> None:
    problem = _build_problem(goal_type=TossingRoomSplitGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, frames = problem.run_task_episode(
        task=task, policy=_oracle_policy(problem=problem, task=task)
    )
    assert solved is True
    assert frames == []


def test_run_task_episode_fails_when_the_policy_never_solves() -> None:
    problem = _build_problem(goal_type=TossingRoomSplitGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, _ = problem.run_task_episode(task=task, policy=_never_solves_policy)
    assert solved is False


def test_run_task_episode_records_a_frame_per_step_with_a_renderer() -> None:
    problem = _build_problem(goal_type=TossingRoomSplitGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    solved, frames = problem.run_task_episode(
        task=task,
        policy=_oracle_policy(problem=problem, task=task),
        renderer=TossingRoomSplitRenderer,
    )
    assert solved is True
    assert len(frames) >= 2  # at least the initial frame plus one action
    for frame in frames:
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


def test_run_task_episode_sets_env_state_from_the_task_initial_state() -> None:
    problem = _build_problem(goal_type=TossingRoomSplitGoalType.RECYCLING)
    task = problem.tasks.sample_test_task()
    problem.run_task_episode(task=task, policy=_never_solves_policy)
    assert problem.env.get_current_state().get(
        obj=TossingRoomSplitEnvironment.robot, feature_name="room"
    ) == task.initial_state.get(obj=TossingRoomSplitEnvironment.robot, feature_name="room")
