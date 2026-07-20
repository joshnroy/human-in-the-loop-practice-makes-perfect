from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import (
    SKILL_ORACLE_POLICY,
    SkillOraclePolicy,
)
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks


def test_get_labeled_action_moves_straight_to_the_light_first() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = SkillOraclePolicy.get_labeled_action(state=state)
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    light_x = state.get(obj=LightSwitchEnvironment.light, feature_name="x")
    assert labeled.action.tolist() == [light_x - robot_x, 0.0]
    assert labeled.label.startswith("MoveRobot(")


def test_get_labeled_action_dials_exactly_to_target_once_at_the_light() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.2, light_target=0.9)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    labeled = SkillOraclePolicy.get_labeled_action(state=state)
    assert labeled.action[0] == 0.0
    assert labeled.action[1] == 0.9 - 0.2
    assert labeled.label.startswith("TurnOnLight(")


def test_skill_oracle_policy_constant_adapts_get_labeled_action_to_the_policy_contract() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = SKILL_ORACLE_POLICY(state)
    assert (
        labeled.action.tolist() == SkillOraclePolicy.get_labeled_action(state=state).action.tolist()
    )


def test_solves_a_sampled_task_in_exactly_two_actions() -> None:
    task = LightSwitchTasks.sample_train_task()
    LightSwitchEnvironment.set_state(state=task.initial_state)

    state = LightSwitchEnvironment.get_current_state()
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=SKILL_ORACLE_POLICY(state).action)
    assert task.goal.is_satisfied(state=state) is False

    state = LightSwitchEnvironment.take_action(action=SKILL_ORACLE_POLICY(state).action)
    assert task.goal.is_satisfied(state=state) is True
