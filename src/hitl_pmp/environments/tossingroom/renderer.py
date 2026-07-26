from typing import ClassVar, cast

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from hitl_pmp.core.problem.environment.environment import Environment  # noqa: E402
from hitl_pmp.core.problem.environment.types import State  # noqa: E402
from hitl_pmp.core.renderer.renderer import Renderer  # noqa: E402

from .environment import TossingRoomEnvironment  # noqa: E402


class TossingRoomRenderer(Renderer):
    """Draws the 1-D hallway of rooms, the robot (colored by what it holds), the two
    bins (with their item counts), the button, and the one-way ledge as a wall. A
    static-method container, never instantiated, same as every other business-logic
    class in this project.

    Colors encode item kind consistently: recycling is green, trash is saddlebrown,
    an empty hand is gray. The bin the robot is throwing into fills solid; both bins
    always show their current count."""

    # 800x256 at the default 100 dpi -- both divisible by 16, avoiding an ffmpeg
    # macro_block_size resize warning when writing to mp4 (same reasoning as Light
    # Switch's renderer; taller here to fit two bin labels plus the action label).
    figure_size: ClassVar[tuple[float, float]] = (8.0, 2.56)
    marker_size: ClassVar[float] = 320.0
    recycling_color: ClassVar[str] = "forestgreen"
    trash_color: ClassVar[str] = "saddlebrown"
    empty_hand_color: ClassVar[str] = "gray"

    @staticmethod
    def _hand_color(*, holding: int) -> str:
        if holding == TossingRoomEnvironment.RECYCLING_KIND:
            return TossingRoomRenderer.recycling_color
        if holding == TossingRoomEnvironment.TRASH_KIND:
            return TossingRoomRenderer.trash_color
        return TossingRoomRenderer.empty_hand_color

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        # env is typed as the base Environment (matching the Renderer ABC signature,
        # so no Liskov violation); narrow it back before reading TossingRoom-specific
        # fields.
        assert isinstance(env, TossingRoomEnvironment)
        robot_room = int(round(state.get(obj=env.robot, feature_name="room")))
        holding = int(round(state.get(obj=env.robot, feature_name="holding")))
        recycling_count = int(round(state.get(obj=env.recycling_bin, feature_name="count")))
        trash_count = int(round(state.get(obj=env.trash_bin, feature_name="count")))

        fig, ax = plt.subplots(figsize=TossingRoomRenderer.figure_size)
        try:
            ax.set_xlim(0, env.num_rooms)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([i + 0.5 for i in range(env.num_rooms)])
            ax.set_xticklabels([str(i) for i in range(env.num_rooms)], fontsize=7)
            title = "Tossing Room"
            ax.set_title(f"{title}\n{label}" if label else title, fontsize=9)

            # Room boundaries.
            ax.vlines(
                np.arange(0, env.num_rooms + 1),
                ymin=0,
                ymax=1,
                colors="lightgray",
                linewidth=0.5,
                zorder=0,
            )
            # The one-way ledge: a thick wall between blocked_right_from and the room
            # above it (rightward blocked, leftward allowed).
            ledge_x = env.blocked_right_from + 1
            ax.vlines(ledge_x, ymin=0, ymax=1, colors="black", linewidth=4.0, zorder=1)
            ax.text(ledge_x, 0.92, "ledge", fontsize=6, ha="center", va="top")

            # Recycling bin.
            ax.scatter(
                [env.recycling_bin_room + 0.3],
                [0.3],
                s=TossingRoomRenderer.marker_size,
                marker="s",
                c=TossingRoomRenderer.recycling_color,
                edgecolors="black",
                zorder=2,
            )
            ax.text(
                env.recycling_bin_room + 0.3,
                0.05,
                f"R:{recycling_count}",
                fontsize=7,
                ha="center",
            )
            # Trash bin.
            ax.scatter(
                [env.trash_bin_room + 0.7],
                [0.3],
                s=TossingRoomRenderer.marker_size,
                marker="s",
                c=TossingRoomRenderer.trash_color,
                edgecolors="black",
                zorder=2,
            )
            ax.text(
                env.trash_bin_room + 0.7,
                0.05,
                f"T:{trash_count}",
                fontsize=7,
                ha="center",
            )
            # Button.
            ax.scatter(
                [env.button_room + 0.5],
                [0.75],
                s=TossingRoomRenderer.marker_size,
                marker="^",
                c="crimson",
                edgecolors="black",
                zorder=2,
            )
            # Robot, colored by what it holds.
            ax.scatter(
                [robot_room + 0.5],
                [0.55],
                s=TossingRoomRenderer.marker_size,
                marker="o",
                c=TossingRoomRenderer._hand_color(holding=holding),
                edgecolors="black",
                zorder=3,
            )
            fig.tight_layout()
            canvas = cast(FigureCanvasAgg, fig.canvas)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        finally:
            # Always close (even if drawing raised): an unclosed Figure leaks in
            # pyplot's global registry, and this runs once per env step.
            plt.close(fig)
