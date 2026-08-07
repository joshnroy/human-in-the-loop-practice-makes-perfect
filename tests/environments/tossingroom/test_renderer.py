import numpy as np

from hitl_pmp.environments.tossingroom.environment import (
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.renderer import (
    TossingRoomRenderer,
)


def _state(*, env: TossingRoomEnvironment):
    return env.build_initial_state(weight_seed=0)


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


def test_render_frame_shows_a_bins_single_item() -> None:
    """A bin holds 0 or 1 items, and the drawing has to distinguish those two -- the
    only two states there are."""
    env = TossingRoomEnvironment()
    empty = _state(env=env)
    full = env.build_initial_state(
        weight_seed=0,
        trash_count=1,
    )
    assert not np.array_equal(
        TossingRoomRenderer.render_frame(state=empty, env=env),
        TossingRoomRenderer.render_frame(state=full, env=env),
    )


def test_render_frame_distinguishes_the_two_bins_buttons() -> None:
    """Each bin has its own button beside it, so emptying the trash bin and emptying the
    recycling bin must not look the same -- a viewer has to be able to tell which button
    was pressed by what changed."""
    env = TossingRoomEnvironment()
    both_full = env.build_initial_state(
        weight_seed=0,
        trash_count=1,
        recycling_count=1,
    )
    trash_emptied = both_full.model_copy(deep=True)
    trash_emptied.set(obj=TossingRoomEnvironment.trash_bin, feature_name="count", feature_val=0.0)
    recycling_emptied = both_full.model_copy(deep=True)
    recycling_emptied.set(
        obj=TossingRoomEnvironment.recycling_bin,
        feature_name="count",
        feature_val=0.0,
    )
    assert not np.array_equal(
        TossingRoomRenderer.render_frame(state=trash_emptied, env=env),
        TossingRoomRenderer.render_frame(state=recycling_emptied, env=env),
    )


def test_render_frame_differs_with_a_label() -> None:
    env = TossingRoomEnvironment()
    state = _state(env=env)
    assert not np.array_equal(
        TossingRoomRenderer.render_frame(state=state, env=env),
        TossingRoomRenderer.render_frame(state=state, env=env, label="Throw(robot, recycling)"),
    )


def test_render_frame_annotates_the_ledge_differently_when_it_is_two_way() -> None:
    """The caption says "One-way" and a red X marks the blocked crossing. Under
    --two-way-ledge neither is true, so the drawing has to change -- otherwise every
    demo video and every --record-full-loop recording of the positive-control arm
    carries a false claim about the world it is showing."""
    one_way = TossingRoomEnvironment()
    two_way = TossingRoomEnvironment(two_way_ledge=True)
    one_way_frame = TossingRoomRenderer.render_frame(state=_state(env=one_way), env=one_way)
    two_way_frame = TossingRoomRenderer.render_frame(state=_state(env=two_way), env=two_way)
    assert one_way_frame.shape == two_way_frame.shape
    assert not np.array_equal(one_way_frame, two_way_frame)
