import pytest

from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.problem import TossingRoomProblem
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomOracle
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


@pytest.mark.parametrize("goal_type", list(TossingRoomGoalType))
def test_oracle_solves_every_goal_type_on_train_and_test_tasks(
    *, goal_type: TossingRoomGoalType
) -> None:
    env = TossingRoomEnvironment()
    tasks = TossingRoomTasks(env=env, seed=0, forced_goal_type=goal_type)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomOracle(env=env))

    for sample in (tasks.sample_train_task, tasks.sample_test_task):
        for _ in range(10):
            task = sample()
            solved, _ = problem.run_task_episode(
                task=task, policy=method.get_task_policy(task=task)
            )
            assert solved is True


def test_oracle_solves_the_default_mixed_distribution() -> None:
    env = TossingRoomEnvironment()
    tasks = TossingRoomTasks(env=env, seed=0)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomOracle(env=env))
    solved = 0
    for _ in range(30):
        task = tasks.sample_test_task()
        ok, _ = problem.run_task_episode(task=task, policy=method.get_task_policy(task=task))
        solved += int(ok)
    assert solved == 30


def test_oracle_presses_both_buttons_in_the_only_feasible_order() -> None:
    """EMPTY is now an ordering task: the trash button (room 6) must be pressed BEFORE
    crossing the one-way ledge down to the recycling one (room 1), because nothing to
    the right of the ledge is reachable again afterwards. Asserted on the realised
    action sequence, not just on the solved flag, so a solve that happened to work by
    another route would still be visible."""
    env = TossingRoomEnvironment()
    tasks = TossingRoomTasks(env=env, seed=0, forced_goal_type=TossingRoomGoalType.EMPTY)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomOracle(env=env))
    for _ in range(10):
        task = tasks.sample_test_task()
        state = problem.reset_to_task(task=task)
        policy = method.get_task_policy(task=task)
        pressed: list[int] = []
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            action = policy(state).action
            if int(round(action[0])) == TossingRoomEnvironment.SKILL_PRESS:
                pressed.append(int(round(action[1])))
            state = env.take_action(action=action)
        assert task.goal.is_satisfied(state=state) is True
        assert pressed == [
            TossingRoomEnvironment.TRASH_KIND,
            TossingRoomEnvironment.RECYCLING_KIND,
        ]


def test_oracle_never_issues_the_blocked_rightward_ledge_step() -> None:
    """The oracle is forward-only: it should solve without ever attempting the single
    irreversible-blocked move (rightward across the ledge), so it never needs help."""
    env = TossingRoomEnvironment()
    tasks = TossingRoomTasks(env=env, seed=0)
    problem = TossingRoomProblem(env=env, tasks=tasks)
    method = SkillOracleMethod(env=env, oracle=TossingRoomOracle(env=env))
    for _ in range(30):
        task = tasks.sample_test_task()
        state = problem.reset_to_task(task=task)
        policy = method.get_task_policy(task=task)
        for _ in range(problem.max_episode_steps()):
            if task.goal.is_satisfied(state=state):
                break
            action = policy(state).action
            robot_room = int(round(state.get(obj=env.robot, feature_name="room")))
            if int(round(action[0])) == TossingRoomEnvironment.SKILL_MOVE_ROOM:
                to_room = int(round(action[1]))
                assert not (
                    robot_room == env.blocked_right_from and to_room == env.blocked_right_from + 1
                )
            state = env.take_action(action=action)
