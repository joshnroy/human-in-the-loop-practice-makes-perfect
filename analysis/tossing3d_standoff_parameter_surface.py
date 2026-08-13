"""Tossing3D's `(release_speed, gripper_release_ms) -> throw distance` surface.

Post-run analysis only: reads a committed grid back in, never builds an environment.

Distance is `ballistic_impact_x - base_x_before_toss`, which pools across standoffs.
Ballistic crossing, not resting position (#240): an obstructed cube stops being recorded
above the floor.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Per-seed traces sit underneath the bold means on the curve panel.
SEED_TRACE_ALPHA = 0.16
SEED_TRACE_WIDTH = 0.8
MEAN_TRACE_WIDTH = 2.3


def load_grid(*, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "standoffs": payload["standoffs"],
        "speeds": payload["speeds"],
        "release_ms": payload["release_ms"],
        "seeds": payload["seeds"],
        "jobs_done": payload.get("jobs_done"),
        "jobs_total": payload.get("jobs_total"),
        "cells_total": payload.get("cells_total"),
        "rows": payload["rows"],
    }


def throw_distance_table(*, grid: dict[str, Any]) -> dict[tuple[float, float, float, int], float]:
    """`(standoff, speed, ms, seed) -> distance` in m; unthrown cells absent, not 0."""
    table: dict[tuple[float, float, float, int], float] = {}
    for row in grid["rows"]:
        if not row.get("threw"):
            continue
        impact = row.get("ballistic_impact_x")
        base = row.get("base_x_before_toss")
        if impact is None or base is None:
            continue
        key = (
            round(float(row["standoff"]), 6),
            round(float(row["commanded_speed_deg"]), 6),
            round(float(row["commanded_release_ms"]), 6),
            int(row["seed"]),
        )
        table[key] = float(impact) - float(base)
    return table


def standoff_invariance(
    *, distances: dict[tuple[float, float, float, int], float]
) -> dict[str, Any]:
    """Spread of distance when only standoff changes, grouped by `(speed, ms, seed)`.

    Groups seen at a single standoff are skipped: averaging them in dilutes the spread.
    """
    groups: dict[tuple[float, float, int], list[float]] = {}
    for (_standoff, speed, release_ms, seed), value in distances.items():
        groups.setdefault((speed, release_ms, seed), []).append(value)
    spreads = [max(values) - min(values) for values in groups.values() if len(values) >= 2]
    if not spreads:
        return {"groups": 0, "max_spread_m": None, "median_spread_m": None}
    array = np.array(spreads)
    return {
        "groups": len(spreads),
        "max_spread_m": float(array.max()),
        "mean_spread_m": float(array.mean()),
        "median_spread_m": float(np.median(array)),
        "p95_spread_m": float(np.percentile(array, 95)),
        "over_10mm": int((array > 0.010).sum()),
        "over_20mm": int((array > 0.020).sum()),
    }


def distance_surface(
    *,
    distances: dict[tuple[float, float, float, int], float],
    speeds: list[float],
    release_ms_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean distance and count per `(speed, ms)` cell, pooling standoff and seed."""
    surface = np.full((len(speeds), len(release_ms_values)), np.nan)
    counts = np.zeros((len(speeds), len(release_ms_values)), dtype=int)
    buckets: dict[tuple[float, float], list[float]] = {}
    for (_standoff, speed, release_ms, _seed), value in distances.items():
        buckets.setdefault((speed, release_ms), []).append(value)
    for si, speed in enumerate(speeds):
        for mi, release_ms in enumerate(release_ms_values):
            values = buckets.get((speed, release_ms))
            if values:
                surface[si, mi] = float(np.mean(values))
                counts[si, mi] = len(values)
    return surface, counts


def build_distance_heatmap(
    *,
    grid: dict[str, Any],
    distances: dict[tuple[float, float, float, int], float],
    output: Path,
) -> None:
    """How far the cube flies, over the two dials that decide it.

    Sequential palette, not project blue/orange: neither axis is an assistance contrast.
    """
    speeds = grid["speeds"]
    release_ms_values = grid["release_ms"]
    surface, counts = distance_surface(
        distances=distances, speeds=speeds, release_ms_values=release_ms_values
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.6))
    image = axis.imshow(
        surface,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(release_ms_values[0], release_ms_values[-1], speeds[0], speeds[-1]),
    )
    bar = figure.colorbar(image, ax=axis, pad=0.02)
    bar.set_label("distance covered before ground contact, from the base (m)")

    if np.isfinite(surface).any():
        mesh_x, mesh_y = np.meshgrid(release_ms_values, speeds)
        finite = surface[np.isfinite(surface)]
        levels = [
            level for level in np.arange(0.2, 1.6, 0.2) if finite.min() < level < finite.max()
        ]
        if levels:
            contours = axis.contour(
                mesh_x, mesh_y, surface, levels=levels, colors="white", linewidths=1.0, alpha=0.75
            )
            axis.clabel(contours, fmt="%.1f m", fontsize=8, colors="white")

    measured = int(np.count_nonzero(np.isfinite(surface)))
    total = len(speeds) * len(release_ms_values)
    per_cell = int(np.median(counts[counts > 0])) if (counts > 0).any() else 0
    reach = (
        f"{np.nanmin(surface):.3f}-{np.nanmax(surface):.3f} m" if measured else "not yet measured"
    )
    axis.set_xlabel("gripper release (ms)")
    axis.set_ylabel("release speed (deg/s)")
    axis.set_title(
        f"How far the cube flies\n"
        f"{measured}/{total} dial settings measured, median {per_cell} throws per setting "
        f"(standoffs and seeds pooled); reach {reach}\n"
        f"Sequential palette, not the project's blue/orange: neither axis is an "
        f"assistance-mechanism contrast",
        fontsize=10,
    )
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_invariance_figure(
    *,
    grid: dict[str, Any],
    distances: dict[tuple[float, float, float, int], float],
    output: Path,
) -> None:
    """Standoff invariance: spread CDF, and the distance curves at one speed."""
    standoffs = grid["standoffs"]
    speeds = grid["speeds"]
    release_ms_values = grid["release_ms"]
    seeds = grid["seeds"]

    groups: dict[tuple[float, float, int], list[float]] = {}
    for (_standoff, speed, release_ms, seed), value in distances.items():
        groups.setdefault((speed, release_ms, seed), []).append(value)
    spreads_mm = sorted(
        (max(values) - min(values)) * 1e3 for values in groups.values() if len(values) >= 2
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.6))
    total = len(spreads_mm)
    if total:
        array = np.array(spreads_mm)
        left.step(
            array,
            np.arange(1, total + 1) / total,
            where="post",
            color="#333333",
            linewidth=MEAN_TRACE_WIDTH,
        )
        for threshold, style in ((10.0, "--"), (20.0, ":")):
            left.axvline(threshold, color="#8c1d04", linestyle=style, linewidth=1.4)
        left.text(
            0.03,
            0.82,
            f"{int((array > 10).sum())}/{total} groups above 10 mm\n"
            f"{int((array > 20).sum())}/{total} groups above 20 mm\n"
            f"median {np.median(array):.2f} mm, worst {array.max():.1f} mm",
            transform=left.transAxes,
            fontsize=9,
            color="#8c1d04",
            va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#8c1d04"},
        )
        left.set_xscale("log")
    left.set_xlabel("distance spread across standoff, within one (speed, ms, seed) group (mm)")
    left.set_ylabel("fraction of groups at or below")
    left.set_title(
        f"Distance is a property of the throw, not the pose\nn={total} matched groups",
        fontsize=10,
    )
    left.grid(alpha=0.25)

    target_speed = speeds[-1]
    colours = plt.get_cmap("cividis")(np.linspace(0.0, 0.85, len(standoffs)))
    for standoff, colour in zip(standoffs, colours, strict=True):
        for seed in seeds:
            xs = [m for m in release_ms_values if (standoff, target_speed, m, seed) in distances]
            ys = [distances[(standoff, target_speed, m, seed)] for m in xs]
            if xs:
                right.plot(xs, ys, color=colour, alpha=SEED_TRACE_ALPHA, linewidth=SEED_TRACE_WIDTH)
        xs, ys, counts = [], [], []
        for release_ms in release_ms_values:
            values = [
                distances[(standoff, target_speed, release_ms, seed)]
                for seed in seeds
                if (standoff, target_speed, release_ms, seed) in distances
            ]
            if values:
                xs.append(release_ms)
                ys.append(float(np.mean(values)))
                counts.append(len(values))
        if xs:
            right.plot(
                xs,
                ys,
                color=colour,
                linewidth=MEAN_TRACE_WIDTH,
                label=f"standoff {standoff:.2f} m -- mean, n={min(counts)}",
            )
    right.set_xlabel("gripper release (ms)")
    right.set_ylabel("distance covered before ground contact (m)")
    right.set_title(
        f"The same curve from every standoff\nrelease speed fixed at {target_speed:g} deg/s",
        fontsize=10,
    )
    right.legend(fontsize=8, loc="best")
    right.grid(alpha=0.25)

    done, total_jobs = grid.get("jobs_done"), grid.get("jobs_total")
    cells = len(grid["rows"])
    cells_total = grid.get("cells_total")
    figure.suptitle(
        f"Standoff invariance -- the supporting result that licenses pooling "
        f"({cells}/{cells_total} cells, {done}/{total_jobs} jobs)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--distance-heatmap", type=Path, required=True)
    parser.add_argument("--invariance-figure", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    grid = load_grid(path=args.grid)
    distances = throw_distance_table(grid=grid)
    print(
        f"jobs {grid['jobs_done']}/{grid['jobs_total']}, "
        f"cells {len(grid['rows'])}/{grid['cells_total']}, throws measured {len(distances)}"
    )

    invariance = standoff_invariance(distances=distances)
    print("\n== standoff invariance of throw distance ==")
    print(f"  groups compared: {invariance['groups']}")
    if invariance["median_spread_m"] is not None:
        print(f"  median spread: {invariance['median_spread_m'] * 1e3:.2f} mm")
        print(f"  p95 spread   : {invariance['p95_spread_m'] * 1e3:.2f} mm")
        print(f"  max spread   : {invariance['max_spread_m'] * 1e3:.2f} mm")
        print(f"  above 10 mm  : {invariance['over_10mm']}/{invariance['groups']}")
        print(f"  above 20 mm  : {invariance['over_20mm']}/{invariance['groups']}")

    surface, counts = distance_surface(
        distances=distances, speeds=grid["speeds"], release_ms_values=grid["release_ms"]
    )
    if np.isfinite(surface).any():
        print("\n== distance surface ==")
        print(f"  cells measured: {int(np.count_nonzero(np.isfinite(surface)))}/{surface.size}")
        print(f"  reach         : {np.nanmin(surface):.4f} - {np.nanmax(surface):.4f} m")

    build_distance_heatmap(grid=grid, distances=distances, output=args.distance_heatmap)
    build_invariance_figure(grid=grid, distances=distances, output=args.invariance_figure)
    print(f"\nwrote {args.distance_heatmap} and {args.invariance_figure}")

    if args.report is not None:
        args.report.write_text(json.dumps({"invariance": invariance}, indent=1))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
