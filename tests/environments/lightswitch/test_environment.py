import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment


def test_hard_reset_sets_canonical_starting_state() -> None:
    env = LightSwitchEnvironment()
    env.hard_reset()
    state = env.get_current_state()
    robot = LightSwitchEnvironment.robot
    light = LightSwitchEnvironment.light
    assert state.get(obj=robot, feature_name="x") == 0.5
    assert state.get(obj=light, feature_name="level") == 0.0
    assert state.get(obj=light, feature_name="target") == env.canonical_light_target
    assert state.get(obj=light, feature_name="x") == env.grid_size - 0.5


def test_hard_reset_state_is_not_already_on() -> None:
    env = LightSwitchEnvironment()
    env.hard_reset()
    state = env.get_current_state()
    light = LightSwitchEnvironment.light
    level = state.get(obj=light, feature_name="level")
    target = state.get(obj=light, feature_name="target")
    assert abs(level - target) >= LightSwitchEnvironment.light_on_tolerance


def test_take_action_moves_robot_by_dx() -> None:
    env = LightSwitchEnvironment()
    env.set_state(state=env.build_initial_state(light_level=0.0, light_target=0.5))
    next_state = env.take_action(action=np.array([2.0, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == 2.5


def test_take_action_updates_current_state() -> None:
    env = LightSwitchEnvironment()
    env.set_state(state=env.build_initial_state(light_level=0.0, light_target=0.5))
    env.take_action(action=np.array([2.0, 0.0]))
    assert env.get_current_state().get(obj=LightSwitchEnvironment.robot, feature_name="x") == 2.5


def test_take_action_clips_robot_position_at_lower_bound() -> None:
    env = LightSwitchEnvironment()
    env.set_state(state=env.build_initial_state(light_level=0.0, light_target=0.5))
    next_state = env.take_action(action=np.array([-10.0, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == 0.0


def test_take_action_clips_robot_position_at_upper_bound() -> None:
    env = LightSwitchEnvironment()
    env.set_state(state=env.build_initial_state(light_level=0.0, light_target=0.5))
    next_state = env.take_action(action=np.array([1e9, 0.0]))
    assert next_state.get(obj=LightSwitchEnvironment.robot, feature_name="x") == float(
        env.grid_size
    )


def test_take_action_ignores_dlight_when_robot_not_at_light() -> None:
    env = LightSwitchEnvironment()
    env.set_state(state=env.build_initial_state(light_level=0.0, light_target=0.5))
    next_state = env.take_action(action=np.array([0.0, 0.7]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 0.0


def test_take_action_applies_dlight_when_robot_at_light() -> None:
    env = LightSwitchEnvironment()
    light_x = float(env.grid_size - 0.5)
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    env.set_state(state=state)

    next_state = env.take_action(action=np.array([0.0, 0.4]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 0.4


def test_take_action_clips_light_level_to_unit_interval() -> None:
    env = LightSwitchEnvironment()
    light_x = float(env.grid_size - 0.5)
    state = env.build_initial_state(light_level=0.9, light_target=0.5)
    state.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)
    env.set_state(state=state)

    next_state = env.take_action(action=np.array([0.0, 0.5]))
    assert next_state.get(obj=LightSwitchEnvironment.light, feature_name="level") == 1.0


def test_get_valid_actions_is_empty_for_continuous_unbounded_space() -> None:
    assert LightSwitchEnvironment().get_valid_actions() == []


def test_action_space_is_two_dimensional_unbounded_box() -> None:
    assert isinstance(LightSwitchEnvironment.action_space, Box)
    assert LightSwitchEnvironment.action_space.shape == (2,)
    assert np.all(np.isinf(LightSwitchEnvironment.action_space.low))
    assert np.all(np.isinf(LightSwitchEnvironment.action_space.high))


def test_get_cells_returns_one_cell_per_grid_position() -> None:
    cells = LightSwitchEnvironment(grid_size=5).get_cells()
    assert len(cells) == 5
    assert [cell.name for cell in cells] == [f"cell{i}" for i in range(5)]


def test_get_cells_reflects_grid_size_at_construction() -> None:
    """Constructing two independently-sized instances is now the whole story --
    no shared ClassVar to override-then-restore around each test."""
    assert len(LightSwitchEnvironment(grid_size=10).get_cells()) == 10
    assert len(LightSwitchEnvironment(grid_size=3).get_cells()) == 3


def test_build_initial_state_includes_each_cell_at_its_position() -> None:
    env = LightSwitchEnvironment(grid_size=4)
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    cells = env.get_cells()
    for i, cell in enumerate(cells):
        assert state.get(obj=cell, feature_name="x") == i + 0.5


def test_same_position_true_within_tolerance() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    cells = env.get_cells()
    robot = LightSwitchEnvironment.robot
    assert LightSwitchEnvironment.same_position(state=state, obj1=robot, obj2=cells[0]) is True


def test_same_position_false_outside_tolerance() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    cells = env.get_cells()
    robot = LightSwitchEnvironment.robot
    assert LightSwitchEnvironment.same_position(state=state, obj1=robot, obj2=cells[1]) is False
