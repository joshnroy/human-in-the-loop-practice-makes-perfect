"""Tossing3D's 2-D (standoff x commanded release speed) success surface, as a heatmap.

Post-run analysis only: this reads the committed grid back in and never builds an
environment or drives a skill. It draws the surface and nothing else -- no fitting, no
inference. Every existing grid on this domain is a 1-D slice of this one (PR #221 swept 11
standoffs at 6 low-end speeds; PR #226/#227 swept 37 speeds at a single standoff), and the
two parameters are coupled by one piece of arithmetic:

    base_x    = bin_x - standoff       (`MoveToThrowPose` parks the base a standoff back)
    landing_x = base_x + range(speed)  (the toss is a ballistic throw of some range)

so the throw solves when `landing_x` lands in the goal box, i.e. when
`range(speed) - standoff` is within half the box. That prediction is drawn over the cells as
a band, from **this grid's own measured ranges** rather than from a model of them, so the
only thing a reader is comparing the cells against is the `+/- 0.15` window.

**Palette.** Deliberately *not* the project's `#0072B2`/`#D55E00`. Those encode "an
assistance mechanism is available" versus "nothing intervenes" across every reset-policy
figure here, and there is no such contrast anywhere in this figure -- every cell is the same
robot doing the same thing with two dials moved. Borrowing them would import a distinction
that does not exist. The cells use a sequential purple ramp because the value is a **count**,
which has no meaningful midpoint and so must not be diverging; `0/1` is a pale neutral and
`1/1` a saturated purple, far enough apart to survive greyscale. Cells where the skill never
executed are hatched grey -- a third state, not a low count, and colouring them as `0/1`
would assert a failed throw where no throw happened.

**One seed.** Every cell is `0/1` or `1/1`; there are no partial cells, so there is no
per-seed spread to draw and a marginal cell is a coin flip. The figure shows the shape and
rough location of the solving region and says nothing trustworthy about where its edges
fall. That is stated on the figure itself rather than left to a caption.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# The goal box is 0.300 m in x, so a throw solves when it lands within half of that of the
# bin -- which, since `base_x = bin_x - standoff`, is `|range(speed) - standoff| <= 0.15`.
GOAL_HALF_WIDTH = 0.15

# Sequential, for a count. Pale neutral -> saturated purple; see the palette note above.
SURFACE_CMAP = LinearSegmentedColormap.from_list("solve", ["#EFEAF4", "#4A1D6A"])
NOT_EXECUTABLE_COLOUR = "#BFBFBF"
GEOMETRY_COLOUR = "#117733"


def geometric_band_edges(*, ranges: np.ndarray, half_width: float) -> tuple[list, list]:
    """The standoffs at which a throw of each range just reaches each edge of the goal box."""
    return list(np.asarray(ranges) - half_width), list(np.asarray(ranges) + half_width)


def load_grid(*, path: Path) -> dict[str, Any]:
    """The committed grid, indexed by `(standoff, speed)` with its axes."""
    payload = json.loads(path.read_text())
    return {
        "standoffs": payload["standoffs"],
        "speeds": payload["speeds"],
        "seeds": payload["seeds"],
        "cells": {(r["standoff"], r["commanded_speed_deg"]): r for r in payload["rows"]},
    }


def executable_standoffs(*, grid: dict[str, Any]) -> list[float]:
    """Standoffs at which the skill sequence actually ran, at every speed.

    A row where `MoveToThrowPose` or `Toss` raised is not a measurement of a throw. Drawing
    it as "did not solve" would assert a failed throw where no throw happened, so those
    rows are hatched instead, and they are excluded from the measured `range(speed)` the
    prediction band is drawn from.
    """
    out = []
    for standoff in grid["standoffs"]:
        cells = [grid["cells"][(standoff, s)] for s in grid["speeds"]]
        if all(c["move_error"] is None and c["toss_error"] is None for c in cells):
            out.append(standoff)
    return out


def range_by_speed(*, grid: dict[str, Any], standoffs: list[float]) -> dict[float, float]:
    """`range(speed)`: how far the cube flies, averaged over the executable standoff rows.

    Averaging across standoff rows is only meaningful if the range does not depend on the
    standoff. `main` prints the spread across those rows so a reader can check that rather
    than take it on trust.
    """
    out = {}
    for speed in grid["speeds"]:
        values = [
            grid["cells"][(s, speed)]["ballistic_impact_x"]
            - grid["cells"][(s, speed)]["base_x_before_toss"]
            for s in standoffs
        ]
        out[speed] = float(np.mean(values))
    return out


def build_figure(*, grid: dict[str, Any], output: Path) -> None:
    """Draw the heatmap, with the geometric prediction band over it."""
    standoffs, speeds = grid["standoffs"], grid["speeds"]
    n_seeds = len(grid["seeds"])
    runnable = executable_standoffs(grid=grid)
    ranges = range_by_speed(grid=grid, standoffs=runnable)

    counts = np.full((len(standoffs), len(speeds)), np.nan)
    for i, standoff in enumerate(standoffs):
        for j, speed in enumerate(speeds):
            if standoff in runnable:
                counts[i, j] = 1.0 if grid["cells"][(standoff, speed)]["solved"] else 0.0

    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    image = ax.imshow(
        counts,
        origin="lower",
        aspect="auto",
        cmap=SURFACE_CMAP,
        vmin=0.0,
        vmax=1.0,
        extent=(-0.5, len(speeds) - 0.5, -0.5, len(standoffs) - 0.5),
    )
    for i, standoff in enumerate(standoffs):
        for j, speed in enumerate(speeds):
            if standoff not in runnable:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=NOT_EXECUTABLE_COLOUR,
                        hatch="///",
                        edgecolor="white",
                        linewidth=0,
                    )
                )
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#444444")
            else:
                solved = 1 if grid["cells"][(standoff, speed)]["solved"] else 0
                ax.text(
                    j,
                    i,
                    f"{solved}/{n_seeds}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if solved else "#5A5A5A",
                    fontweight="bold" if solved else "normal",
                )

    xs = np.arange(len(speeds))
    lower, upper = geometric_band_edges(
        ranges=np.array([ranges[s] for s in speeds]), half_width=GOAL_HALF_WIDTH
    )
    step = standoffs[1] - standoffs[0]

    def to_cell(*, values: Any) -> np.ndarray:
        """Standoffs in metres -> row index, so the band can be drawn over `imshow` cells."""
        return (np.asarray(values) - standoffs[0]) / step

    lower_cells, upper_cells = to_cell(values=lower), to_cell(values=upper)
    ax.plot(xs, lower_cells, color=GEOMETRY_COLOUR, lw=2.2)
    ax.plot(xs, upper_cells, color=GEOMETRY_COLOUR, lw=2.2)
    ax.fill_between(xs, lower_cells, upper_cells, color=GEOMETRY_COLOUR, alpha=0.13)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=GEOMETRY_COLOUR, lw=2.2),
            mpatches.Patch(facecolor=NOT_EXECUTABLE_COLOUR, hatch="///", edgecolor="white"),
        ],
        labels=[
            "geometric prediction: |range(speed) − standoff| <= 0.15 m",
            "skill did not execute — no throw happened",
        ],
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.088),
        ncol=2,
        frameon=False,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax.set_yticks(np.arange(len(standoffs)))
    ax.set_yticklabels([f"{s:.2f}" for s in standoffs])
    ax.set_xlabel("commanded release speed (joint-path deg/s)")
    ax.set_ylabel("throw standoff (m)")
    ax.set_title(
        "Tossing3D: the (standoff x commanded release speed) success surface\n"
        f"{len(standoffs)} standoffs x {len(speeds)} speeds x {n_seeds} seed "
        f"(seed {grid['seeds'][0]}) = {len(standoffs) * len(speeds)} cells — "
        "every cell is 0/1 or 1/1, so a marginal cell is a coin flip",
        fontsize=10.5,
    )
    bar = fig.colorbar(image, ax=ax, ticks=[0.0, 1.0], fraction=0.035, pad=0.02)
    bar.ax.set_yticklabels(["0/1", "1/1"])
    bar.set_label("seeds solved", fontsize=9)
    fig.text(
        0.5,
        0.018,
        "Palette deviates from the project's blue/orange on purpose: those encode\n"
        "assistance-available vs nothing-intervenes, and no cell here is an arm of that\n"
        "comparison. Sequential purple, because the cell value is a count.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grid = load_grid(path=args.grid)
    build_figure(grid=grid, output=args.output)
    runnable = executable_standoffs(grid=grid)
    ranges = range_by_speed(grid=grid, standoffs=runnable)
    solved = sum(1 for s in runnable for sp in grid["speeds"] if grid["cells"][(s, sp)]["solved"])
    total = len(runnable) * len(grid["speeds"])
    print(f"wrote {args.output}")
    print(f"executable standoff rows: {runnable}")
    print(f"solved {solved}/{total} executable cells")
    print("measured range(speed), and its spread across the executable standoff rows:")
    for speed in grid["speeds"]:
        values = [
            grid["cells"][(s, speed)]["ballistic_impact_x"]
            - grid["cells"][(s, speed)]["base_x_before_toss"]
            for s in runnable
        ]
        print(
            f"  {speed:6.0f} deg/s: range {ranges[speed]:.4f} m, "
            f"spread across standoffs {max(values) - min(values):.4f} m"
        )


if __name__ == "__main__":
    main()
