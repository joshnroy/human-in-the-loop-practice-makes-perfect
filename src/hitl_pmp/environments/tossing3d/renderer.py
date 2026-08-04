from typing import ClassVar, cast

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from hitl_pmp.core.problem.environment.environment import Environment  # noqa: E402
from hitl_pmp.core.problem.environment.types import State  # noqa: E402
from hitl_pmp.core.renderer.renderer import Renderer  # noqa: E402

from .environment import Tossing3DEnvironment  # noqa: E402


class Tossing3DRenderer(Renderer):
    """Renders KINDER's own `task_view` camera, with the skill that produced the frame
    and the cube's position overlaid.

    Unlike every other renderer here, this one does NOT draw the passed `state`: the
    picture comes from the live MuJoCo scene, which `Problem.run_task_episode` has just
    advanced to exactly that state. The state is still read, for the caption -- so a
    reader can see the cube's x against the goal region's edges rather than having to
    judge a 3-D projection by eye.

    Consequence worth stating: one `take_action` is a whole skill execution (a few
    hundred MuJoCo ticks), so a recorded episode is a ~5-frame *storyboard* -- initial
    scene, after the pick, after moving to the throw pose, after the toss -- not a
    smooth video of the arm swinging. That is enough to read off whether the cube ended
    in the goal region, which is what a checkpoint comparison needs, and it costs one
    render call per transition instead of hundreds.

    A static-method container, never instantiated.
    """

    # 640x480 is KINDER's own task_view resolution, plus a caption strip above it.
    # 640x528 at 100 dpi -- both divisible by 16, avoiding ffmpeg's macro_block_size
    # resize warning.
    figure_size: ClassVar[tuple[float, float]] = (6.4, 5.28)
    dpi: ClassVar[int] = 100

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        tossing_env = cast(Tossing3DEnvironment, env)
        frame = tossing_env.backend().render()

        figure = plt.figure(figsize=Tossing3DRenderer.figure_size, dpi=Tossing3DRenderer.dpi)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_axes((0.0, 0.0, 1.0, 0.9))
        axes.imshow(frame)
        axes.set_axis_off()
        figure.text(
            0.5,
            0.95,
            Tossing3DRenderer._caption(state=state, env=tossing_env, label=label),
            ha="center",
            va="center",
            fontsize=11,
        )
        canvas.draw()
        rendered = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
        plt.close(figure)
        return rendered

    @staticmethod
    def _caption(*, state: State, env: Tossing3DEnvironment, label: str | None) -> str:
        cube_x = state.get(obj=env.cube, feature_name="x")
        x_min = state.get(obj=env.goal_region, feature_name="x_min")
        x_max = state.get(obj=env.goal_region, feature_name="x_max")
        holding = int(round(state.get(obj=env.robot, feature_name="holding")))
        prefix = "start" if label is None else label
        held = "  holding" if holding else ""
        return f"{prefix}   cube x={cube_x:.2f}   goal x in [{x_min:.2f}, {x_max:.2f}]{held}"
