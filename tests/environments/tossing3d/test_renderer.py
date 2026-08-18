"""`Tossing3DRenderer`'s caption. The frame itself comes from MuJoCo -- see
`test_kinder_fidelity.py`.

**The caption is simulator-backed now, and that follows from what it reports.** It used to
print an `InBin` verdict computed by a hitl classifier over a box carried in the `State`,
which was pure arithmetic and therefore offline. It now reports `MovableInGoalRegion`, read
off upstream's abstractor, so a caption cannot disagree with the episode's own outcome --
and evaluating it needs the live scene.

The frame-geometry tests below stay offline, because they are arithmetic over `ClassVar`s
and ffmpeg's constraint does not depend on any scene.
"""

import numpy as np

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer

from .conftest import CANONICAL_SEED, requires_kinder


def _state_with_cube_at(
    *, env: Tossing3DEnvironment, x: float, y: float = 0.0, z: float = 0.0444
) -> State:
    """The live scene with the cube moved, so a caption can be checked at a landing.

    Moves it in KINDER's own state and re-translates, rather than editing the `core.State`
    directly: the verdict is computed by the abstractor from a state handed back to it, so
    a hand-edited `core.State` would be testing a path the renderer never takes.
    """
    kinder_state = env.backend().kinder_state().copy()
    cube = kinder_state.get_object_from_name(env.cube.name)
    kinder_state.set(cube, "x", x)
    kinder_state.set(cube, "y", y)
    kinder_state.set(cube, "z", z)
    env.backend().restore(snapshot=kinder_state)
    return env.build_state(kinder_state=kinder_state, seed=CANONICAL_SEED, steps_taken=1)


# --- offline: frame geometry, which no scene can change -------------------------------


def test_the_caption_bar_keeps_both_frame_dimensions_divisible_by_sixteen() -> None:
    """ffmpeg's macro_block_size. KINDER renders 640x480, so the bar has to be a multiple
    of 16 too or every mp4 this domain writes gets silently rescaled."""
    assert Tossing3DRenderer.caption_height % 16 == 0
    assert (480 + Tossing3DRenderer.caption_height) % 16 == 0


def test_captioning_no_substep_frames_produces_no_frames() -> None:
    """A skill whose controller failed to launch steps the simulator zero times. That is
    an ordinary outcome and must not synthesise a frame out of nothing.

    Offline, and deliberately so: the empty case returns before it builds a bar, so this
    is the one caption path that still needs no scene -- which is exactly why a `State`
    carrying nothing but the scene object is enough to exercise it."""
    bare = State(data={Tossing3DEnvironment.scene: np.array([float(CANONICAL_SEED), 0.0])})
    assert (
        Tossing3DRenderer.render_substep_frames(
            frames=[], state=bare, env=Tossing3DEnvironment(), label="pick_cube(...)"
        )
        == []
    )


# --- simulator-backed: everything that evaluates the verdict --------------------------


@requires_kinder
def test_the_caption_reports_the_cube_position_the_bin_and_the_verdict(
    *, live_env: Tossing3DEnvironment
) -> None:
    """Whether a landing scored is illegible from pixels alone -- the goal region is a flat
    patch of floor the bin merely sits on -- so the verdict and the numbers it was derived
    from are burned under every frame.

    **The goal box itself is no longer in the caption, and could not be.** It used to print
    `[1.8500, 2.1500]` off six features carried on the bin object; the region is read from
    the live simulator now and never enters the `State`. The bin's measured x is what still
    tells a reader whether the geometry moved."""
    state = _state_with_cube_at(env=live_env, x=2.02)

    lines = Tossing3DRenderer.caption(
        state=state, env=live_env, label="move_to_toss_location_and_toss(...)"
    )

    assert "2.0200" in lines[1]
    assert "bin x=2.0000" in lines[1]
    assert "MovableInGoalRegion = True" in lines[1]


@requires_kinder
def test_a_landing_short_of_the_goal_region_is_captioned_as_outside_it(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The caption has to report a miss as a miss. x = 1.90 is inside the bin's own
    0.3 m footprint but short of the scored region, which is the case a viewer is least
    able to judge from the picture and most likely to misread."""
    state = _state_with_cube_at(env=live_env, x=1.90)

    lines = Tossing3DRenderer.caption(state=state, env=live_env, label="toss")

    assert "MovableInGoalRegion = False" in lines[1]


@requires_kinder
def test_the_caption_names_the_variant_but_no_longer_a_scene_choice(
    *, live_env: Tossing3DEnvironment, live_state: State
) -> None:
    """The first line used to carry a `[stock]`/`[coincident]` token, because the same
    throw scored `True` on one and `False` on the other and a viewer could not see which
    clip they were looking at. There is one scene now, so a token there would be a
    constant that reads like a choice."""
    header = Tossing3DRenderer.caption(state=live_state, env=live_env)[0]
    assert "Tossing3D-o1" in header
    assert "stock" not in header
    assert "coincident" not in header


@requires_kinder
def test_the_first_frame_of_an_episode_is_labelled_as_the_initial_state(
    *, live_env: Tossing3DEnvironment, live_state: State
) -> None:
    """`Renderer.render_frame` gets `label=None` before any action has been taken."""
    lines = Tossing3DRenderer.caption(state=live_state, env=live_env, label=None)
    assert "initial state" in lines[0]


@requires_kinder
def test_the_caption_bar_is_an_rgb_uint8_image_of_the_requested_width(
    *, live_env: Tossing3DEnvironment, live_state: State
) -> None:
    bar = Tossing3DRenderer._caption_bar(  # noqa: SLF001
        state=live_state, env=live_env, width=640, label=None
    )
    assert bar.shape == (Tossing3DRenderer.caption_height, 640, 3)
    assert bar.dtype.name == "uint8"


@requires_kinder
def test_every_substep_frame_carries_the_caption_bar(
    *, live_env: Tossing3DEnvironment, live_state: State
) -> None:
    """A clip cannot mix heights -- ffmpeg needs one frame size -- so the sub-step frames
    that make the episode smooth have to be captioned too, not just the skill boundary."""
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    captioned = Tossing3DRenderer.render_substep_frames(
        frames=[raw, raw, raw], state=live_state, env=live_env, label="toss"
    )

    assert len(captioned) == 3
    for frame in captioned:
        assert frame.shape == (480 + Tossing3DRenderer.caption_height, 640, 3)
        assert frame.dtype.name == "uint8"


@requires_kinder
def test_a_substep_frames_caption_is_the_one_that_skills_own_frame_carries(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The bar is built once from the state the skill produced and held across that skill's
    frames. The numbers describe the skill's outcome, so they are measured rather than
    invented, and they stay legible for longer than one frame."""
    measured = _state_with_cube_at(env=live_env, x=2.02)
    label = "move_to_toss_location_and_toss(...)"
    bar = Tossing3DRenderer._caption_bar(  # noqa: SLF001
        state=measured, env=live_env, width=640, label=label
    )

    captioned = Tossing3DRenderer.render_substep_frames(
        frames=[np.zeros((480, 640, 3), dtype=np.uint8)],
        state=measured,
        env=live_env,
        label=label,
    )

    assert np.array_equal(captioned[0][480:], bar)


@requires_kinder
def test_substep_frames_keep_their_order_and_their_pixels(
    *, live_env: Tossing3DEnvironment, live_state: State
) -> None:
    """The whole point is a physics-rate clip, so a drop or a reorder is the one defect
    that would make it worse than the four-frame storyboard it replaces."""
    frames = [np.full((480, 640, 3), fill, dtype=np.uint8) for fill in (1, 2, 3)]
    captioned = Tossing3DRenderer.render_substep_frames(
        frames=frames, state=live_state, env=live_env, label="pick_cube(...)"
    )

    for original, result in zip(frames, captioned, strict=True):
        assert np.array_equal(result[:480], original)
