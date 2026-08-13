"""Tossing3D's `core.Renderer`: MuJoCo's own frame, plus a measured caption.

Two things about this renderer are unlike every other one in this repo, and both follow
from the domain rather than from a choice made here.

**It renders the live simulator, not the `state` it is handed.** Every other domain draws
a picture *of* the `State` argument with matplotlib, so it can render any state at any
time. Here the picture comes from MuJoCo's offscreen render of the scene as it actually
is, and a flat `core.State` cannot put the simulator into an arbitrary configuration
(same reason `Tossing3DEnvironment.set_state` refuses a mid-episode state). In practice
this is not a discrepancy: `Problem.run_task_episode` renders immediately after the
transition that produced `state`, so the simulator *is* that state. The caption's numbers
are read from the `state` argument, so if the two ever came apart the frame would say so.

**`render_frame` is one frame per skill; `render_substep_frames` is the movie.**
`core.Renderer` emits one frame per transition, and one transition here is a whole
controller execution -- tens to hundreds of MuJoCo ticks. Rendering only at those
boundaries gave a four-frame clip of a domain whose entire point is a throw, so
`Tossing3DProblem.run_task_episode` now captions the per-tick frames the
`gymnasium.wrappers.RenderCollection` wrapper collects (see `kinder_backend.py`) and
`render_frame` is the fallback for a skill that stepped the simulator zero times.
`render_frame`'s own signature is untouched -- it still returns exactly one frame, which
is right, and nothing in `core/` had to learn about sub-steps.

A clip cannot mix frame sizes, so the sub-step frames are captioned too. The bar is built
once per skill, from the state that skill produced, and held across that skill's frames:
the same shape of caption `scripts/tossing3d_oracle_demo.py` stacks under all 128 frames
of its per-tick clip, where the numbers are likewise the clip's measured *outcome*. So
the caption describes the skill that is running and how it ended, and every number in it
is still measured rather than restated.

The caption reports the cube's position, the live goal box and the `InGoalRegion` verdict,
because "the cube landed in the bin and this scores a failure" is the single most
misreadable thing about this domain and is illegible from pixels alone.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.renderer.renderer import Renderer

from .environment import Tossing3DEnvironment
from .predicates import InGoalRegionClassifier


class Tossing3DRenderer(Renderer):
    """Renders the live KINDER scene with a measured caption bar beneath it."""

    # KINDER renders this scene at 640x480 (`task_view`'s own `resolution` in the task
    # JSON, matching `camera_width`/`camera_height`). 48 keeps the total height 528, and
    # both dimensions divisible by 16 -- ffmpeg's macro_block_size, which every renderer
    # in this repo is sized against.
    caption_height: ClassVar[int] = 48
    caption_padding: ClassVar[int] = 5
    caption_font_size: ClassVar[int] = 11
    caption_background: ClassVar[tuple[int, int, int]] = (16, 16, 16)
    caption_foreground: ClassVar[tuple[int, int, int]] = (238, 238, 238)

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        assert isinstance(env, Tossing3DEnvironment), (
            f"Tossing3DRenderer needs a Tossing3DEnvironment, got {type(env).__name__}"
        )
        frame = env.backend().render()
        bar = Tossing3DRenderer._caption_bar(
            state=state, env=env, width=frame.shape[1], label=label
        )
        return np.vstack([frame, bar])

    @staticmethod
    def render_substep_frames(
        *,
        frames: Sequence[np.ndarray],
        state: State,
        env: Environment,
        label: str | None = None,
    ) -> list[np.ndarray]:
        """Caption a whole skill's worth of already-rendered per-tick frames.

        Deliberately *not* an override of `render_frame`: that returns one frame from one
        `State`, which is the correct core interface and is left alone. This takes frames
        the simulator already produced -- MuJoCo has moved on since, so they cannot be
        re-rendered -- and stacks the identical caption bar under each, so the sub-step
        frames and the boundary frame are the same size and ffmpeg accepts the clip.

        An empty input gives an empty output: a controller whose motion planning failed
        steps the simulator zero times, and inventing a frame for it would be a lie about
        what happened.
        """
        assert isinstance(env, Tossing3DEnvironment), (
            f"Tossing3DRenderer needs a Tossing3DEnvironment, got {type(env).__name__}"
        )
        if not frames:
            return []
        bar = Tossing3DRenderer._caption_bar(
            state=state, env=env, width=frames[0].shape[1], label=label
        )
        return [np.vstack([np.asarray(frame, dtype=np.uint8), bar]) for frame in frames]

    @staticmethod
    def caption(*, state: State, env: Tossing3DEnvironment, label: str | None = None) -> list[str]:
        """The two lines burned under every frame.

        Shared with nothing else on purpose -- unlike the demo script, which captions a
        per-tick clip -- but built the same way: every number is measured from the state
        being rendered, never restated from a doc.
        """
        x, y, z = (state.get(obj=env.cube, feature_name=name) for name in ("x", "y", "z"))
        in_region = InGoalRegionClassifier.holds(
            state=state, cube=env.cube, goal_region=env.goal_region
        )
        x_min = state.get(obj=env.goal_region, feature_name="x_min")
        x_max = state.get(obj=env.goal_region, feature_name="x_max")
        return [
            # The scene name used to be stamped here too, because the same throw scored
            # True on the coincident config and False on stock and a viewer could not
            # see which clip they were looking at. There is one scene now, so the token
            # would be a constant -- and a constant that reads like a choice is worse
            # than none. `goal x in [...]` below is measured live and still distinguishes
            # a scene whose geometry moved.
            f"Tossing3D-{env.variant} | {label or 'initial state'}",
            f"cube x={x:.4f} y={y:.4f} z={z:.4f} | goal x in [{x_min:.4f}, {x_max:.4f}] "
            f"| InGoalRegion = {in_region}",
        ]

    @staticmethod
    def _caption_bar(
        *, state: State, env: Tossing3DEnvironment, width: int, label: str | None
    ) -> np.ndarray:
        lines = Tossing3DRenderer.caption(state=state, env=env, label=label)
        image = Image.new(
            "RGB",
            (width, Tossing3DRenderer.caption_height),
            Tossing3DRenderer.caption_background,
        )
        draw = ImageDraw.Draw(image)
        font = Tossing3DRenderer._font()
        line_height = Tossing3DRenderer.caption_font_size + 5
        for index, line in enumerate(lines):
            draw.text(
                (
                    Tossing3DRenderer.caption_padding,
                    Tossing3DRenderer.caption_padding + index * line_height,
                ),
                line,
                font=font,
                fill=Tossing3DRenderer.caption_foreground,
            )
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        """DejaVu, which ships with matplotlib -- already a hard dependency of this repo,
        so no system font has to be found. `load_default()` is a last resort that keeps a
        caption legible-ish rather than failing a render at the very end."""
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(str(path), Tossing3DRenderer.caption_font_size)
        except OSError:
            return ImageFont.load_default()
