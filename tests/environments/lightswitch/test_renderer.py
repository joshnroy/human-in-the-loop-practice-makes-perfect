import numpy as np

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer


def test_render_frame_returns_an_rgb_uint8_array() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    frame = LightSwitchRenderer.render_frame(state=state)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_render_frame_differs_between_meaningfully_different_robot_positions() -> None:
    original_grid_size = LightSwitchEnvironment.grid_size
    try:
        LightSwitchEnvironment.grid_size = 10
        state_a = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
        state_b = state_a.model_copy(deep=True)
        state_b.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=9.0)

        frame_a = LightSwitchRenderer.render_frame(state=state_a)
        frame_b = LightSwitchRenderer.render_frame(state=state_b)

        assert frame_a.shape == frame_b.shape
        assert not np.array_equal(frame_a, frame_b)
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size


def test_render_frame_differs_between_light_on_and_off() -> None:
    light_x = float(LightSwitchEnvironment.grid_size - 0.5)
    state_off = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.9)
    state_off.set(obj=LightSwitchEnvironment.robot, feature_name="x", feature_val=light_x)

    state_on = state_off.model_copy(deep=True)
    state_on.set(obj=LightSwitchEnvironment.light, feature_name="level", feature_val=0.9)

    frame_off = LightSwitchRenderer.render_frame(state=state_off)
    frame_on = LightSwitchRenderer.render_frame(state=state_on)

    assert not np.array_equal(frame_off, frame_on)


def test_render_frame_shape_is_consistent_across_calls() -> None:
    state = LightSwitchEnvironment.build_initial_state(light_level=0.0, light_target=0.5)
    frame1 = LightSwitchRenderer.render_frame(state=state)
    frame2 = LightSwitchRenderer.render_frame(state=state)
    assert frame1.shape == frame2.shape
