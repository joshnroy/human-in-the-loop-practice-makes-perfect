"""Offline tests for the caption. The frame itself comes from MuJoCo -- see
`test_kinder_fidelity.py`."""

import numpy as np

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment, Tossing3DTaskConfig
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

from .observations import state


def test_the_caption_reports_the_cube_position_the_goal_box_and_the_verdict() -> None:
    """ "The cube landed in the bin and this scores a failure" is the single most
    misreadable thing about this domain, and it is illegible from pixels alone."""
    env = Tossing3DEnvironment()
    lines = Tossing3DRenderer.caption(
        state=state(cube_x=1.9902, cube_y=0.0105, cube_z=0.0444), env=env, label="Toss(...)"
    )
    assert "1.9902" in lines[1]
    assert "[1.8500, 2.1500]" in lines[1]
    assert "InGoalRegion = True" in lines[1]


def test_the_caption_names_the_scene_because_the_verdict_inverts_between_them() -> None:
    """The same throw scores True on the coincident config and False on stock. Two clips
    that differ only in a number a viewer cannot see would be unreadable."""
    coincident = Tossing3DRenderer.caption(state=state(), env=Tossing3DEnvironment())[0]
    stock = Tossing3DRenderer.caption(
        state=state(), env=Tossing3DEnvironment(task_config=Tossing3DTaskConfig.STOCK)
    )[0]
    assert "coincident" in coincident
    assert "stock" in stock


def test_the_stock_landing_is_captioned_as_outside_the_goal_region() -> None:
    """x = 2.2197 is where the same throw rests on stock: in the bin, past the goal box."""
    lines = Tossing3DRenderer.caption(
        state=state(cube_x=2.2197, cube_z=0.0444), env=Tossing3DEnvironment()
    )
    assert "InGoalRegion = False" in lines[1]


def test_the_first_frame_of_an_episode_is_labelled_as_the_initial_state() -> None:
    """`Renderer.render_frame` gets `label=None` before any action has been taken."""
    lines = Tossing3DRenderer.caption(state=state(), env=Tossing3DEnvironment(), label=None)
    assert "initial state" in lines[0]


def test_the_caption_bar_keeps_both_frame_dimensions_divisible_by_sixteen() -> None:
    """ffmpeg's macro_block_size. KINDER renders 640x480, so the bar has to be a multiple
    of 16 too or every mp4 this domain writes gets silently rescaled."""
    assert Tossing3DRenderer.caption_height % 16 == 0
    assert (480 + Tossing3DRenderer.caption_height) % 16 == 0


def test_the_caption_bar_is_an_rgb_uint8_image_of_the_requested_width() -> None:
    bar = Tossing3DRenderer._caption_bar(
        state=state(), env=Tossing3DEnvironment(), width=640, label=None
    )
    assert bar.shape == (Tossing3DRenderer.caption_height, 640, 3)
    assert bar.dtype.name == "uint8"


def test_every_substep_frame_carries_the_caption_bar() -> None:
    """A clip cannot mix heights -- ffmpeg needs one frame size -- so the sub-step frames
    that make the episode smooth have to be captioned too, not just the skill boundary."""
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    captioned = Tossing3DRenderer.render_substep_frames(
        frames=[raw, raw, raw], state=state(), env=Tossing3DEnvironment(), label="Toss(...)"
    )

    assert len(captioned) == 3
    for frame in captioned:
        assert frame.shape == (480 + Tossing3DRenderer.caption_height, 640, 3)
        assert frame.dtype.name == "uint8"


def test_a_substep_frames_caption_is_the_one_that_skills_own_frame_carries() -> None:
    """The bar is built once from the state the skill produced and held across that
    skill's frames -- the same shape of caption `scripts/tossing3d_oracle_demo.py` stacks
    under all 128 frames of its per-tick clip. The numbers describe the skill's outcome,
    so they are measured, not invented, and they stay legible for longer than one frame."""
    env = Tossing3DEnvironment()
    measured = state(cube_x=1.9902, cube_y=0.0105, cube_z=0.0444)
    bar = Tossing3DRenderer._caption_bar(state=measured, env=env, width=640, label="Toss(...)")

    captioned = Tossing3DRenderer.render_substep_frames(
        frames=[np.zeros((480, 640, 3), dtype=np.uint8)],
        state=measured,
        env=env,
        label="Toss(...)",
    )

    assert np.array_equal(captioned[0][480:], bar)


def test_substep_frames_keep_their_order_and_their_pixels() -> None:
    """The whole point is a physics-rate clip, so a drop or a reorder is the one defect
    that would make it worse than the four-frame storyboard it replaces."""
    frames = [np.full((480, 640, 3), fill, dtype=np.uint8) for fill in (1, 2, 3)]
    captioned = Tossing3DRenderer.render_substep_frames(
        frames=frames, state=state(), env=Tossing3DEnvironment(), label="Pick(...)"
    )

    for original, result in zip(frames, captioned, strict=True):
        assert np.array_equal(result[:480], original)


def test_captioning_no_substep_frames_produces_no_frames() -> None:
    """A skill whose controller failed to launch steps the simulator zero times. That is
    an ordinary outcome and must not synthesise a frame out of nothing."""
    assert (
        Tossing3DRenderer.render_substep_frames(
            frames=[], state=state(), env=Tossing3DEnvironment(), label="Pick(...)"
        )
        == []
    )
