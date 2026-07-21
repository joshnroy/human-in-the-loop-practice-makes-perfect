import numpy as np

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer


def test_render_frame_returns_an_rgb_uint8_array() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    frame = LightSwitchRenderer.render_frame(state=state, env=env)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_render_frame_differs_between_meaningfully_different_robot_positions() -> None:
    env = LightSwitchEnvironment(grid_size=10)
    state_a = env.build_initial_state(light_level=0.0, light_target=0.5)
    state_b = state_a.model_copy(deep=True)
    state_b.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=9.0)

    frame_a = LightSwitchRenderer.render_frame(state=state_a, env=env)
    frame_b = LightSwitchRenderer.render_frame(state=state_b, env=env)

    assert frame_a.shape == frame_b.shape
    assert not np.array_equal(frame_a, frame_b)


def test_render_frame_differs_between_light_on_and_off() -> None:
    env = LightSwitchEnvironment()
    light_x = float(env.grid_size - 0.5)
    state_off = env.build_initial_state(light_level=0.0, light_target=0.9)
    state_off.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    state_on = state_off.model_copy(deep=True)
    state_on.set(obj=LightSwitchEnvironment.light, feature_name="level", feature_val=0.9)

    frame_off = LightSwitchRenderer.render_frame(state=state_off, env=env)
    frame_on = LightSwitchRenderer.render_frame(state=state_on, env=env)

    assert not np.array_equal(frame_off, frame_on)


def test_render_frame_shape_is_consistent_across_calls() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    frame1 = LightSwitchRenderer.render_frame(state=state, env=env)
    frame2 = LightSwitchRenderer.render_frame(state=state, env=env)
    assert frame1.shape == frame2.shape


def test_render_frame_with_a_label_differs_from_without_one() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    frame_unlabeled = LightSwitchRenderer.render_frame(state=state, env=env)
    frame_labeled = LightSwitchRenderer.render_frame(
        state=state, env=env, label="MoveRobot(robot, cell99)"
    )
    assert frame_unlabeled.shape == frame_labeled.shape
    assert not np.array_equal(frame_unlabeled, frame_labeled)


def test_render_frame_with_different_labels_differ() -> None:
    env = LightSwitchEnvironment()
    state = env.build_initial_state(light_level=0.0, light_target=0.5)
    frame_a = LightSwitchRenderer.render_frame(
        state=state, env=env, label="MoveRobot(robot, cell99)"
    )
    frame_b = LightSwitchRenderer.render_frame(
        state=state, env=env, label="TurnOnLight(robot, light)"
    )
    assert not np.array_equal(frame_a, frame_b)
