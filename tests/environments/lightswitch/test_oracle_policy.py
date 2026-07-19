from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.oracle_policy import ORACLE_POLICY
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def test_first_action_moves_straight_to_the_light() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    action = ORACLE_POLICY(state)
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    light_x = state.get(obj=LightSwitchEnvironment.light, feature_name="x")
    assert action[0] == light_x - robot_x
    assert action[1] == 0.0


def test_second_action_dials_exactly_to_target_once_at_the_light() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.2, light_target=0.9)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    action = ORACLE_POLICY(state)
    assert action[0] == 0.0
    assert action[1] == 0.9 - 0.2


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=ORACLE_POLICY(state))
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=ORACLE_POLICY(state))
    assert task.goal.is_satisfied(state=state) is True
