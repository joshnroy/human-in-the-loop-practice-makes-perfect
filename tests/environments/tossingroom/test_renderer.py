import numpy as np

from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.renderer import TossingRoomRenderer


def _state(*, env: TossingRoomEnvironment):
    return env.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)


def test_render_frame_returns_an_rgb_uint8_array() -> None:
    env = TossingRoomEnvironment()
    frame = TossingRoomRenderer.render_frame(state=_state(env=env), env=env)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_render_frame_dimensions_are_divisible_by_sixteen() -> None:
    """ffmpeg's default macro_block_size is 16; non-divisible dims trigger a resize
    warning when writing mp4 (same reasoning as Light Switch's renderer)."""
    env = TossingRoomEnvironment()
    frame = TossingRoomRenderer.render_frame(state=_state(env=env), env=env)
    assert frame.shape[0] % 16 == 0
    assert frame.shape[1] % 16 == 0


def test_render_frame_differs_between_robot_positions() -> None:
    env = TossingRoomEnvironment()
    state_a = _state(env=env)
    state_b = state_a.model_copy(deep=True)
    state_b.set(obj=TossingRoomEnvironment.robot, feature_name="room", feature_val=0.0)
    frame_a = TossingRoomRenderer.render_frame(state=state_a, env=env)
    frame_b = TossingRoomRenderer.render_frame(state=state_b, env=env)
    assert frame_a.shape == frame_b.shape
    assert not np.array_equal(frame_a, frame_b)


def test_render_frame_differs_when_holding_an_item() -> None:
    env = TossingRoomEnvironment()
    empty = _state(env=env)
    holding = empty.model_copy(deep=True)
    holding.set(
        obj=TossingRoomEnvironment.robot,
        feature_name="holding",
        feature_val=float(TossingRoomEnvironment.RECYCLING_KIND),
    )
    assert not np.array_equal(
        TossingRoomRenderer.render_frame(state=empty, env=env),
        TossingRoomRenderer.render_frame(state=holding, env=env),
    )


def test_render_frame_differs_with_a_label() -> None:
    env = TossingRoomEnvironment()
    state = _state(env=env)
    assert not np.array_equal(
        TossingRoomRenderer.render_frame(state=state, env=env),
        TossingRoomRenderer.render_frame(state=state, env=env, label="Throw(robot, recycling)"),
    )
