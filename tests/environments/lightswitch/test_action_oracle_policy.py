from hitl_pmp.environments.lightswitch.action_oracle_policy import (
    ACTION_ORACLE_POLICY,
    ActionOraclePolicy,
)
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks

# ActionOraclePolicy itself never reads instance-level Environment state (only the
# ClassVar constants robot/light/same_position_tolerance -- see its own docstring),
# so it needs no changes at all; these tests just need instances to build states/tasks.
_ENV = LightSwitchEnvironment()
_TASKS = LightSwitchTasks(env=_ENV)


def test_get_action_moves_straight_to_the_light_first() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.7)
    action = ActionOraclePolicy.get_action(state=state)
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    light_x = state.get(obj=LightSwitchEnvironment.light, feature_name="x")
    assert action[0] == light_x - robot_x
    assert action[1] == 0.0


def test_get_action_dials_exactly_to_target_once_at_the_light() -> None:
    light_x = float(_ENV.grid_size - 0.5)
    state = _ENV.build_initial_state(light_level=0.2, light_target=0.9)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    action = ActionOraclePolicy.get_action(state=state)
    assert action[0] == 0.0
    assert action[1] == 0.9 - 0.2


def test_get_labeled_action_wraps_get_action_with_a_description() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = ActionOraclePolicy.get_labeled_action(state=state)
    assert labeled.action.tolist() == ActionOraclePolicy.get_action(state=state).tolist()
    assert "raw action" in labeled.label


def test_action_oracle_policy_constant_adapts_get_labeled_action_to_the_policy_contract() -> None:
    state = _ENV.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = ACTION_ORACLE_POLICY(state)
    assert labeled.action.tolist() == ActionOraclePolicy.get_action(state=state).tolist()


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    task = _TASKS.sample_train_task()
    _ENV.set_state(state=task.initial_state)

    state = _ENV.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    state = _ENV.take_action(action=ACTION_ORACLE_POLICY(state).action)
    assert task.goal.is_satisfied(state=state) is False

    state = _ENV.take_action(action=ACTION_ORACLE_POLICY(state).action)
    assert task.goal.is_satisfied(state=state) is True
