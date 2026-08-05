from typing import ClassVar, cast

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from hitl_pmp.core.problem.environment.environment import Environment  # noqa: E402
from hitl_pmp.core.problem.environment.types import State  # noqa: E402
from hitl_pmp.core.renderer.renderer import Renderer  # noqa: E402

from .environment import TossingRoomSplitEnvironment  # noqa: E402


class TossingRoomSplitRenderer(Renderer):
    """`TossingRoomRenderer` verbatim, retyped to this domain. Nothing about the *world*
    differs between the two -- the split is entirely in the symbolic layer -- so the
    picture is identical, and keeping it identical is deliberate: a reviewer diffing the
    two files should see only the type names change.

    Draws the hallway as a labeled floor-plan diagram: a row of "Room i" boxes, the Start
    room and both bin rooms tinted, the recycling and trash bins in their rooms holding
    either nothing or their single item, EACH BIN'S OWN empty/incinerate button beside it
    (labeled with the bin it empties, so which button is which is never ambiguous), the
    limitless trash+recycling piles at the start, the robot (outlined by what it holds),
    and the one-way ledge as a barrier with an arrow for the passable crossing and an X
    for the blocked return.

    A bin holds at most one item, so its contents are drawn as the item itself (present
    or absent) rather than as a running tally.

    Reads every position from the env instance, so it renders whatever layout the env
    is configured with (the ledge arrow always points the *passable* way -- leftward
    for the default V1 layout, since stepping right across the ledge is the blocked
    move). A static-method container, never instantiated."""

    # 1280x240 at the default 100 dpi -- both divisible by 16, avoiding ffmpeg's
    # macro_block_size resize warning; the wide, short aspect matches the hallway.
    figure_size: ClassVar[tuple[float, float]] = (12.8, 2.4)

    start_fill: ClassVar[str] = "#d7e8f7"
    button_fill: ClassVar[str] = "#fbe3d4"
    recycling_color: ClassVar[str] = "#2e8b57"
    trash_color: ClassVar[str] = "#8a8a8a"
    robot_color: ClassVar[str] = "#2b6cb0"
    ledge_color: ClassVar[str] = "#c0392b"

    @staticmethod
    def _hand_color(*, holding: int) -> str:
        if holding == TossingRoomSplitEnvironment.RECYCLING_KIND:
            return TossingRoomSplitRenderer.recycling_color
        if holding == TossingRoomSplitEnvironment.TRASH_KIND:
            return TossingRoomSplitRenderer.trash_color
        return "black"

    @staticmethod
    def _draw_robot(*, ax: Axes, x: float, y: float, holding: int) -> None:
        r = TossingRoomSplitRenderer
        edge = r._hand_color(holding=holding)
        # body: a rounded blue box with two "eyes" -- a simple, legible robot glyph,
        # outlined in the color of whatever it is carrying (black when empty-handed).
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.13, y - 0.12),
                0.26,
                0.24,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor=r.robot_color,
                edgecolor=edge,
                linewidth=2.5 if holding else 1.2,
                zorder=6,
            )
        )
        for dx in (-0.05, 0.05):
            ax.add_patch(Circle((x + dx, y + 0.02), 0.022, facecolor="white", zorder=7))
        ax.plot([x, x], [y + 0.12, y + 0.18], color=r.robot_color, lw=1.5, zorder=6)
        ax.add_patch(Circle((x, y + 0.19), 0.018, facecolor=r.robot_color, zorder=6))

    @staticmethod
    def _draw_bin(*, ax: Axes, x: float, y: float, color: str, count: int, letter: str) -> None:
        # a bin as a small open box with a thicker rim. Capacity is 1, so the badge
        # reads 0 or 1 and the item itself is drawn poking out of the bin when present.
        ax.add_patch(
            Rectangle((x - 0.11, y - 0.11), 0.22, 0.2, facecolor=color, edgecolor="black", zorder=5)
        )
        ax.add_patch(
            Rectangle(
                (x - 0.13, y + 0.07), 0.26, 0.05, facecolor=color, edgecolor="black", zorder=5
            )
        )
        if count:
            ax.add_patch(
                Polygon(
                    [[x - 0.045, y + 0.11], [x + 0.045, y + 0.11], [x, y + 0.21]],
                    facecolor=color,
                    edgecolor="black",
                    zorder=6,
                )
            )
        ax.text(
            x,
            y - 0.02,
            f"{letter}:{count}",
            color="white",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    @staticmethod
    def _draw_button(*, ax: Axes, x: float, label: str) -> None:
        """One bin's own empty/incinerate button, beside that bin, with a leader line to
        a label naming which bin it empties -- pressing it empties only that one."""
        ax.add_patch(Circle((x, 0.5), 0.055, facecolor="#e8862c", edgecolor="black", zorder=5))
        ax.plot([x, x], [0.44, -0.12], color="black", lw=0.8, zorder=1)
        ax.text(x, -0.16, label, ha="center", va="top", fontsize=7)

    @staticmethod
    def _draw_pile(*, ax: Axes, x: float, y: float, color: str, name: str, count: int = 3) -> None:
        # `count` small item-triangles in a tight row, a concrete depiction of the
        # start room's pile (the env pile itself is limitless for Pickup; this just
        # shows n of them). The label carries the count so it reads unambiguously.
        half = 0.028
        spacing = 0.062
        left = x - spacing * (count - 1) / 2.0
        for i in range(count):
            cx = left + i * spacing
            ax.add_patch(
                Polygon(
                    [[cx - half, y], [cx + half, y], [cx, y + 0.075]],
                    facecolor=color,
                    edgecolor="black",
                    zorder=4,
                )
            )
        ax.text(
            x,
            y - 0.05,
            f"{name} x{count}",
            color="black",
            fontsize=6.5,
            ha="center",
            va="top",
            zorder=4,
        )

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        assert isinstance(env, TossingRoomSplitEnvironment)
        r = TossingRoomSplitRenderer
        robot_room = int(round(state.get(obj=env.robot, feature_name="room")))
        holding = int(round(state.get(obj=env.robot, feature_name="holding")))
        recycling_count = int(round(state.get(obj=env.recycling_bin, feature_name="count")))
        trash_count = int(round(state.get(obj=env.trash_bin, feature_name="count")))
        n = env.num_rooms

        fig, ax = plt.subplots(figsize=r.figure_size)
        try:
            ax.set_xlim(-0.15, n + 0.15)
            ax.set_ylim(-0.45, 1.35)
            ax.axis("off")

            # Room boxes, with Start and both bin/button rooms tinted.
            button_rooms = {env.trash_bin_room, env.recycling_bin_room}
            for i in range(n):
                fill = "white"
                if i == env.start_room:
                    fill = r.start_fill
                elif i in button_rooms:
                    fill = r.button_fill
                ax.add_patch(
                    Rectangle(
                        (i, 0), 1, 1, facecolor=fill, edgecolor="black", linewidth=1.3, zorder=1
                    )
                )
                ax.text(i + 0.5, 0.9, f"Room {i}", ha="center", va="center", fontsize=8, zorder=2)

            ax.text(
                env.start_room + 0.5,
                1.12,
                "Start",
                color=r.robot_color,
                fontsize=10,
                weight="bold",
                ha="center",
                va="bottom",
            )

            # Bins (0 or 1 item each) and, beside each one, its OWN button. The button is
            # drawn in the same room as its bin and labeled with that bin, because
            # pressing it empties only that bin -- a viewer must never have to guess which
            # button belongs to which.
            r._draw_bin(
                ax=ax,
                x=env.recycling_bin_room + 0.35,
                y=0.28,
                color=r.recycling_color,
                count=recycling_count,
                letter="R",
            )
            r._draw_button(x=env.recycling_bin_room + 0.75, ax=ax, label="Empty\nRecycling")
            r._draw_bin(
                ax=ax,
                x=env.trash_bin_room + 0.35,
                y=0.28,
                color=r.trash_color,
                count=trash_count,
                letter="T",
            )
            r._draw_button(x=env.trash_bin_room + 0.75, ax=ax, label="Empty\nTrash")
            r._draw_pile(ax=ax, x=env.start_room + 0.45, y=0.12, color=r.trash_color, name="Trash")
            r._draw_pile(
                ax=ax, x=env.start_room + 0.78, y=0.12, color=r.recycling_color, name="Recycling"
            )

            # The one-way ledge: a bold barrier between blocked_right_from and the room
            # to its right. Stepping right across it is blocked; leftward is allowed --
            # so the arrow points the passable (left) way and the X marks the block.
            lx = float(env.blocked_right_from + 1)
            ax.add_patch(
                Rectangle(
                    (lx - 0.025, 0),
                    0.05,
                    1.0,
                    facecolor=r.ledge_color,
                    edgecolor=r.ledge_color,
                    zorder=4,
                )
            )
            ax.text(
                lx,
                1.12,
                "One-way\nLedge/Barrier",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
            # passable crossing (leftward): arrow pointing away from the ledge, left.
            ax.annotate(
                "",
                xy=(lx - 0.42, 0.62),
                xytext=(lx + 0.02, 0.62),
                arrowprops={"arrowstyle": "-|>", "color": r.ledge_color, "linewidth": 3},
                zorder=6,
            )
            # blocked return (rightward): a red X on the right side of the ledge.
            xx = lx + 0.22
            ax.plot([xx - 0.06, xx + 0.06], [0.24, 0.36], color=r.ledge_color, lw=3, zorder=6)
            ax.plot([xx - 0.06, xx + 0.06], [0.36, 0.24], color=r.ledge_color, lw=3, zorder=6)

            r._draw_robot(ax=ax, x=robot_room + 0.5, y=0.55, holding=holding)

            if label:
                ax.text((n) / 2.0, -0.34, label, ha="center", va="top", fontsize=8, color="#555555")

            fig.tight_layout()
            canvas = cast(FigureCanvasAgg, fig.canvas)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        finally:
            # Always close (even if drawing raised): an unclosed Figure leaks in
            # pyplot's global registry, and this runs once per env step.
            plt.close(fig)
