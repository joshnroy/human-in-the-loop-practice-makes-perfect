from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_oracle_policy import SkillOraclePolicy


def test_get_labeled_action_moves_straight_to_the_light_first() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.7)
    labeled = SkillOraclePolicy.get_labeled_action(state=state)
    robot_x = state.get(obj=LightSwitchEnvironment.robot, feature_name="x")
    light_x = state.get(obj=LightSwitchEnvironment.light, feature_name="x")
    assert labeled.action.tolist() == [light_x - robot_x, 0.0]
    assert labeled.label.startswith("MoveRobot(")
    # MoveRobot has param_dim=0 -- no continuous parameters to show.
    assert "params=" not in labeled.label


def test_get_labeled_action_dials_exactly_to_target_once_at_the_light() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.2, light_target=0.9)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    labeled = SkillOraclePolicy.get_labeled_action(state=state)
    assert labeled.action[0] == 0.0
    assert labeled.action[1] == 0.9 - 0.2
    assert labeled.label.startswith("TurnOnLight(")
    # TurnOnLight has param_dim=1 -- the actual dlight value sent to compute_action
    # should be visible in the label, not just the skill name and objects.
    assert "params=[0.7]" in labeled.label
