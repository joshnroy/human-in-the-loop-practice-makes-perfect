import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment


def test_hard_reset_sets_canonical_starting_state() -> None:
    LightSwitchEnvironment.hard_reset()
    state = LightSwitchEnvironment.get_current_state()
    robot = LightSwitchEnvironment.robot
    light = LightSwitchEnvironment.light
    assert state.get(obj=robot, feature_name="x") == 0.5
    assert state.get(obj=light, feature_name="level") == 0.0
    assert (
        state.get(obj=light, feature_name="target") == LightSwitchEnvironment.canonical_light_target
    )
    assert state.get(obj=light, feature_name="x") == LightSwitchEnvironment.grid_size - 0.5


def test_hard_reset_state_is_not_already_on() -> None:
    LightSwitchEnvironment.hard_reset()
    state = LightSwitchEnvironment.get_current_state()
    light = LightSwitchEnvironment.light
    level = state.get(obj=light, feature_name="level")
    target = state.get(obj=light, feature_name="target")
    assert abs(level - target) >= LightSwitchEnvironment.light_on_tolerance


def test_take_action_moves_robot_by_dx() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    next_state = LightSwitchEnvironment.take_action(action=np.array([2.0, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == 2.5


def test_take_action_updates_current_state() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    LightSwitchEnvironment.take_action(action=np.array([2.0, 0.0]))
    assert (
        LightSwitchEnvironment.get_current_state().get(
            obj=LightSwitchEnvironment.robot, feature_name="x"
        )
        == 2.5
    )


def test_take_action_clips_robot_position_at_lower_bound() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    next_state = LightSwitchEnvironment.take_action(action=np.array([-10.0, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == 0.0


def test_take_action_clips_robot_position_at_upper_bound() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    next_state = LightSwitchEnvironment.take_action(action=np.array([1e9, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == float(
        LightSwitchEnvironment.grid_size
    )


def test_take_action_ignores_dlight_when_robot_not_at_light() -> None:
    LightSwitchEnvironment.set_state(
        state=LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    )
    next_state = LightSwitchEnvironment.take_action(action=np.array([0.0, 0.7]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 0.0


def test_take_action_applies_dlight_when_robot_at_light() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    LightSwitchEnvironment.set_state(state=state)

    next_state = LightSwitchEnvironment.take_action(action=np.array([0.0, 0.4]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 0.4


def test_take_action_clips_light_level_to_unit_interval() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state = LightSwitchEnvironment.build_initial_state(light_level=0.9, light_target=0.5)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    LightSwitchEnvironment.set_state(state=state)

    next_state = LightSwitchEnvironment.take_action(action=np.array([0.0, 0.5]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 1.0


def test_get_valid_actions_is_empty_for_continuous_unbounded_space() -> None:
    assert LightSwitchEnvironment.get_valid_actions() == []


def test_action_space_is_two_dimensional_unbounded_box() -> None:
    assert isinstance(LightSwitchEnvironment.action_space, Box)
    assert LightSwitchEnvironment.action_space.shape == (2,)
    assert np.all(np.isinf(LightSwitchEnvironment.action_space.low))
    assert np.all(np.isinf(LightSwitchEnvironment.action_space.high))
