from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.tasks import TossingRoomGoalType, TossingRoomTasks
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def test_get_task_policy_dispatches_to_tossingroom_and_solves() -> None:
    """Tossing Room's oracle logic lives in get_task_policy (not get_labeled_action),
    since it needs the task's goal to know which item/bin/room to head for."""
    env = TossingRoomEnvironment()
    tasks = TossingRoomTasks(env=env, seed=0, forced_goal_type=TossingRoomGoalType.RECYCLING)
    task = tasks.sample_test_task()
    env.set_state(state=task.initial_state)
    method = SkillOracleMethod(env=env)
    policy = method.get_task_policy(task=task)

    state = env.get_current_state()
    assert task.goal.is_satisfied(state=state) is False
    for _ in range(env.num_rooms + 3):
        if task.goal.is_satisfied(state=state):
            break
        state = env.take_action(action=policy(state).action)
    assert task.goal.is_satisfied(state=state) is True


def test_task_policy_is_goal_specific() -> None:
    """A recycling task and a trash task from the same start state produce different
    first actions (Pickup of a different kind), proving the closure captures the
    goal rather than ignoring it."""
    env = TossingRoomEnvironment()
    recycling_task = TossingRoomTasks(
        env=env, seed=0, forced_goal_type=TossingRoomGoalType.RECYCLING
    ).sample_test_task()
    trash_task = TossingRoomTasks(
        env=env, seed=0, forced_goal_type=TossingRoomGoalType.TRASH
    ).sample_test_task()
    method = SkillOracleMethod(env=env)

    state = recycling_task.initial_state
    recycling_first = method.get_task_policy(task=recycling_task)(state).action
    trash_first = method.get_task_policy(task=trash_task)(state).action
    assert recycling_first[1] == TossingRoomEnvironment.RECYCLING_KIND
    assert trash_first[1] == TossingRoomEnvironment.TRASH_KIND
