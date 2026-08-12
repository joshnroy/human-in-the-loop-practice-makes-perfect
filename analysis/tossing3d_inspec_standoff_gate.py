"""Read the in-spec speed x standoff gate grid back in and answer one question:

**does a workable standoff window reproduce at a hardware-feasible release speed?**

The design doc predicts one at `[1.100, 1.205]` -- but that prediction is arithmetic
from a linear fit of `range` against commanded speed, extrapolated to a speed nobody had
run. This reads the measured grid instead.

Post-run analysis only: it reads the probe's JSON and never drives a simulator. One JSON
per standoff, as `scripts/tossing3d_toss_speed_probe.py --mode in-spec` writes them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GREY = "#666666"

# What the design doc predicted, so the figure shows the comparison rather than asserting
# it in prose. `[0.905, 1.205]` is the feasible standoff interval its linear range fit
# implies at the in-spec ceiling; the barrier floor clips it to `[1.100, 1.205]`.
PREDICTED_WINDOW = (1.100, 1.205)
# `WORST_BARRIER_COLLISION_STANDOFF (1.00) + BARRIER_COLLISION_MARGIN (0.10)`, which is
# also `THROW_STANDOFF_BOUNDS`'s lower edge -- the smallest standoff the sampler may draw.
BARRIER_FLOOR = 1.100


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="directory of standoff_*.json files written by the probe",
    )
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


def load_grid(*, raw_dir: Path) -> tuple[list[dict[str, Any]], tuple[float, float] | None]:
    """Every cell across every standoff file, plus the goal box's x extent."""
    cells: list[dict[str, Any]] = []
    goal_box: tuple[float, float] | None = None
    for path in sorted(raw_dir.glob("standoff_*.json")):
        payload = json.loads(path.read_text())
        for row in payload["cells"]:
            cells.append(row)
            region = row.get("goal_region")
            if goal_box is None and region is not None:
                goal_box = (float(region[0]), float(region[3]))
    return cells, goal_box


def _by(*, cells: list[dict[str, Any]], key: str) -> list[float]:
    return sorted({float(c[key]) for c in cells})


def solved_table(*, cells: list[dict[str, Any]]) -> dict[tuple[float, float], list[bool]]:
    table: dict[tuple[float, float], list[bool]] = defaultdict(list)
    for c in cells:
        if c.get("solved") is None:
            continue
        table[(float(c["standoff"]), float(c["commanded_speed_deg"]))].append(bool(c["solved"]))
    return table


def print_report(*, cells: list[dict[str, Any]], goal_box: tuple[float, float] | None) -> None:
    standoffs = _by(cells=cells, key="standoff")
    speeds = _by(cells=cells, key="commanded_speed_deg")
    table = solved_table(cells=cells)

    print(f"goal box x: {goal_box}")
    print(f"{len(cells)} cells; {len(standoffs)} standoffs x {len(speeds)} speeds")
    print()
    header = "standoff | " + " | ".join(f"{s:>7.2f}" for s in speeds) + " |   any"
    print(header)
    print("-" * len(header))
    window: list[float] = []
    for d in standoffs:
        parts = []
        any_solved = 0
        total = 0
        for v in speeds:
            got = table.get((d, v), [])
            parts.append(f"{sum(got):>3}/{len(got):<3}")
            any_solved += sum(got)
            total += len(got)
        if any_solved > 0:
            window.append(d)
        print(f"{d:>8.3f} | " + " | ".join(parts) + f" | {any_solved:>3}/{total}")
    print()
    if window:
        print(f"standoffs with at least one solve: {min(window):.3f} - {max(window):.3f}")
    else:
        print("standoffs with at least one solve: NONE -- null result, no window reproduced")


def make_figure(
    *,
    cells: list[dict[str, Any]],
    goal_box: tuple[float, float] | None,
    output_png: Path,
) -> None:
    standoffs = _by(cells=cells, key="standoff")
    speeds = _by(cells=cells, key="commanded_speed_deg")
    seeds = sorted({int(c["seed"]) for c in cells})
    # Speed is an ordered quantity, so it gets a sequential ramp rather than the
    # role-based blue/orange, which encodes "mechanism exists vs does not" and would be
    # meaningless here.
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(speeds)))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    lookup: dict[tuple[float, float, int], dict[str, Any]] = {
        (float(c["standoff"]), float(c["commanded_speed_deg"]), int(c["seed"])): c for c in cells
    }

    # Panel 1 -- where the cube comes to rest, against the box it has to land in.
    ax = axes[0]
    if goal_box is not None:
        ax.axhspan(goal_box[0], goal_box[1], color=GREY, alpha=0.18, zorder=0)
        ax.text(
            standoffs[0], goal_box[1], " goal box", va="bottom", ha="left", fontsize=9, color=GREY
        )
    for ci, v in enumerate(speeds):
        for sd in seeds:
            trace = [lookup.get((d, v, sd), {}).get("cube_x_final") for d in standoffs]
            ax.plot(standoffs, trace, color=colors[ci], alpha=0.16, linewidth=0.8)
        means = [
            float(
                np.mean([
                    x
                    for sd in seeds
                    if (x := lookup.get((d, v, sd), {}).get("cube_x_final")) is not None
                ])
            )
            for d in standoffs
        ]
        ax.plot(
            standoffs,
            means,
            color=colors[ci],
            linewidth=2.3,
            label=f"{v:.4g} deg/s — mean, n={len(seeds)}",
        )
    ax.set_xlabel("standoff (m)")
    ax.set_ylabel("cube resting x (m)")
    ax.set_title(f"Where the cube lands (seeds: {len(seeds)})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")

    # Panel 2 -- the label itself, per speed.
    ax = axes[1]
    for ci, v in enumerate(speeds):
        counts = [
            sum(bool(lookup.get((d, v, sd), {}).get("solved")) for sd in seeds) for d in standoffs
        ]
        ax.plot(
            standoffs,
            counts,
            color=colors[ci],
            linewidth=2.3,
            marker="o",
            markersize=4,
            label=f"{v:.4g} deg/s, n={len(seeds)}",
        )
    ax.axvspan(*PREDICTED_WINDOW, color="#D55E00", alpha=0.13, zorder=0)
    ax.axvline(BARRIER_FLOOR, color=GREY, linestyle=":", linewidth=1.4)
    ax.set_xlabel("standoff (m)")
    ax.set_ylabel("seeds solved")
    ax.set_title(f"Solves per standoff (of {len(seeds)} seeds)")
    ax.set_ylim(-0.4, len(seeds) + 0.4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")

    # Panel 3 -- the window: every in-spec (speed, seed) cell that solved.
    ax = axes[2]
    totals = [
        sum(bool(lookup.get((d, v, sd), {}).get("solved")) for v in speeds for sd in seeds)
        for d in standoffs
    ]
    denom = len(speeds) * len(seeds)
    ax.axvspan(
        *PREDICTED_WINDOW,
        color="#D55E00",
        alpha=0.13,
        zorder=0,
        label="predicted window [1.100, 1.205]",
    )
    ax.bar(standoffs, totals, width=0.018, color="#0072B2", label="measured, in-spec")
    ax.axvline(
        BARRIER_FLOOR,
        color=GREY,
        linestyle=":",
        linewidth=1.4,
        label="barrier floor / sampler lower edge",
    )
    ax.set_xlabel("standoff (m)")
    ax.set_ylabel("cells solved")
    ax.set_title(f"In-spec window (of {denom} speed x seed cells per standoff)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Tossing3D at a hardware-feasible release speed: does a workable standoff window exist?",
        fontsize=12,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    print(f"wrote {output_png}")


def main() -> None:
    args = _parse_args()
    cells, goal_box = load_grid(raw_dir=args.raw_dir)
    print_report(cells=cells, goal_box=goal_box)
    make_figure(cells=cells, goal_box=goal_box, output_png=args.output_png)


if __name__ == "__main__":
    main()
