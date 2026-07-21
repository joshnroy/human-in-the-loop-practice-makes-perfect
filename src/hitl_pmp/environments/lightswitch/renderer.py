from typing import ClassVar, cast

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from hitl_pmp.core.problem.environment.environment import Environment  # noqa: E402
from hitl_pmp.core.problem.environment.types import State  # noqa: E402
from hitl_pmp.core.renderer.renderer import Renderer  # noqa: E402

from .environment import LightSwitchEnvironment  # noqa: E402
from .predicates import LightOnClassifier  # noqa: E402


class LightSwitchRenderer(Renderer):
    """Draws the robot (red circle) and light (square, gold when
    LightOnClassifier.holds else light gray) on a 1D strip spanning the grid, with a
    thin vertical line at each integer boundary marking off the individual cells
    ("rooms"). A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    # At the default 100 dpi this is 800x192px -- both divisible by 16, avoiding an
    # ffmpeg macro_block_size resize warning when writing to mp4. The extra height
    # (vs. an earlier 800x160) leaves room for a second title line when a label is
    # overlaid (see render_frame's label param).
    figure_size: ClassVar[tuple[float, float]] = (8.0, 1.92)
    marker_size: ClassVar[float] = 300.0

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        # env is typed as the base Environment here (matching the Renderer ABC's
        # signature exactly -- narrowing a parameter type on override would violate
        # Liskov substitution, per mypy); narrow it back before reading Light-Switch-
        # specific fields (grid_size, robot, light) that only LightSwitchEnvironment
        # actually has.
        assert isinstance(env, LightSwitchEnvironment)
        robot_x = state.get(obj=env.robot, feature_name="x")
        light_x = state.get(obj=env.light, feature_name="x")
        is_on = LightOnClassifier.holds(state=state, light=env.light)

        fig, ax = plt.subplots(figsize=LightSwitchRenderer.figure_size)
        try:
            ax.set_xlim(0, env.grid_size)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            title = f"Light Switch ({'on' if is_on else 'off'})"
            ax.set_title(f"{title}\n{label}" if label else title, fontsize=9)
            ax.vlines(
                np.arange(0, env.grid_size + 1),
                ymin=0,
                ymax=1,
                colors="lightgray",
                linewidth=0.5,
                zorder=0,
            )
            ax.scatter(
                [light_x],
                [0.5],
                s=LightSwitchRenderer.marker_size,
                marker="s",
                c="gold" if is_on else "lightgray",
                edgecolors="black",
            )
            ax.scatter(
                [robot_x],
                [0.5],
                s=LightSwitchRenderer.marker_size,
                marker="o",
                c="red",
                edgecolors="black",
            )
            fig.tight_layout()
            # matplotlib.use("Agg") above guarantees fig.canvas is really a
            # FigureCanvasAgg at runtime; its base type doesn't declare buffer_rgba.
            canvas = cast(FigureCanvasAgg, fig.canvas)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        finally:
            # Always close, even if drawing raised -- an unclosed Figure leaks in
            # pyplot's global registry for the rest of the process, and this runs
            # once per env step (many times per rendered episode).
            plt.close(fig)
