"""Read `scripts/tossing3d_toss_speed_probe.py`'s JSON back and answer the four questions
the probe exists for: the ceiling, the dial's authority in metres, release-pose
invariance, and whether the dial can overshoot the goal box as well as undershoot it.

Post-run only. This never drives a simulator; it takes one JSON per scaling mode and
produces the report and the figure.

    scripts/with_env.sh python analysis/tossing3d_toss_speed_probe.py \\
        --probe vel=out/vel.json --probe vel-accel=out/vel-accel.json \\
        --probe vel-accel-decel=out/vel-accel-decel.json \\
        --output-png docs/experiment-logs/<name>.png

## Reading the figure

Colour is the arm's role, per the project palette. Orange is `vel` -- the arm with no
extra mechanism, the one where nothing lifts the ceiling. Blue is option D, which *has*
a mechanism, with linestyle separating the two variants of it: solid for
`vel-accel-decel`, dashed for `vel-accel`. Grey is reference geometry that is not an arm
at all -- the identity line, upstream's own 140 deg/s, and the goal box.
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

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#666666"

DASHED = (0, (4, 2))

# Orange is the arm nothing helps; blue is the arm that has the extra mechanism, with
# linestyle carrying which variant of it. See the module docstring.
MODE_STYLE = {
    "vel": {"color": ORANGE, "linestyle": "-", "label": "vel only"},
    "vel-accel": {"color": BLUE, "linestyle": DASHED, "label": "vel + accel"},
    "vel-accel-decel": {"color": BLUE, "linestyle": "-", "label": "vel + accel + decel"},
}

UPSTREAM_DEFAULT_DEG = 140.0


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="append",
        required=True,
        metavar="MODE=PATH",
        help="one per scaling mode, e.g. --probe vel=out/vel.json",
    )
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


def load_probe(*, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "cells" not in payload:
        raise ValueError(f"{path} is not a toss-speed probe payload")
    return payload


def cells_by_speed(*, payload: dict[str, Any]) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for cell in payload["cells"]:
        grouped[float(cell["commanded_speed_deg"])].append(cell)
    return dict(sorted(grouped.items()))


def _mean(*, values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else float("nan")


def _goal_box_x(*, payload: dict[str, Any]) -> tuple[float, float] | None:
    """The goal region's x extent, which every cell records identically."""
    for cell in payload["cells"]:
        box = cell.get("goal_region")
        if box:
            return float(box[0]), float(box[3])
    return None


def print_report(*, mode: str, payload: dict[str, Any]) -> None:
    grouped = cells_by_speed(payload=payload)
    seeds = sorted({int(c["seed"]) for c in payload["cells"]})
    print(f"\n{'=' * 78}")
    print(f"mode={mode}   seeds={len(seeds)}   standoff={payload['standoff']}")
    print(f"kinder_models: {payload['kinder_models_file']}")
    print(f"{'=' * 78}")
    print(
        "'achieved' is the arm's path speed when the gripper-open decision is made; "
        "'post' is\nthe same quantity one control step later. The cube separates between "
        "the two."
    )
    header = (
        f"{'cmd deg/s':>10} {'achieved':>10} {'sd':>6} {'post':>9} {'ratio':>6} "
        f"{'rel step':>9} {'frac':>6} {'range m':>9} {'sd':>6} "
        f"{'solved':>8} {'max torque':>11}"
    )
    print(header)
    for speed, cells in grouped.items():
        achieved = [c["achieved_release_speed_deg"] for c in cells]
        ranges = [c["range_m"] for c in cells]
        solved = [c for c in cells if c.get("solved")]
        present_ach = [a for a in achieved if a is not None]
        present_rng = [r for r in ranges if r is not None]
        mean_ach = _mean(values=achieved)
        print(
            f"{speed:>10.0f} {mean_ach:>10.2f} "
            f"{(np.std(present_ach) if present_ach else float('nan')):>6.2f} "
            f"{_mean(values=[c.get('achieved_release_speed_after_deg') for c in cells]):>9.2f} "
            f"{mean_ach / speed:>6.3f} "
            f"{_mean(values=[c['release_step'] for c in cells]):>9.1f} "
            f"{_mean(values=[c['release_fraction_covered'] for c in cells]):>6.3f} "
            f"{_mean(values=ranges):>9.3f} "
            f"{(np.std(present_rng) if present_rng else float('nan')):>6.3f} "
            f"{f'{len(solved)}/{len(cells)}':>8} "
            f"{_mean(values=[c['max_torque_fraction'] for c in cells]):>11.3f}"
        )

    # R5: how far the achieved release configuration moves across the whole grid.
    confs = [
        c["achieved_release_conf_deg"]
        for c in payload["cells"]
        if c.get("achieved_release_conf_deg")
    ]
    if confs:
        spread = np.ptp(np.array(confs), axis=0)
        print(
            f"R5 release-conf spread over the whole grid (deg, per joint): "
            f"{np.round(spread, 2).tolist()}"
        )
        print(f"R5 worst joint: {float(spread.max()):.2f} deg at joint {int(spread.argmax()) + 1}")

    saturated = sum(int(c.get("torque_saturated_steps") or 0) for c in payload["cells"])
    steps = sum(int(c.get("toss_control_steps") or 0) for c in payload["cells"])
    peak = max(
        (c["max_torque_fraction"] for c in payload["cells"] if c.get("max_torque_fraction")),
        default=float("nan"),
    )
    print(f"R4 torque: {saturated}/{steps} control steps saturated, peak fraction {peak:.3f}")


def make_figure(*, probes: dict[str, dict[str, Any]], output_png: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

    goal_box = next((b for b in (_goal_box_x(payload=p) for p in probes.values()) if b), None)

    # --- panel 1: the ceiling ------------------------------------------------
    ax = axes[0]
    all_speeds: list[float] = []
    for mode, payload in probes.items():
        style = MODE_STYLE[mode]
        grouped = cells_by_speed(payload=payload)
        speeds = list(grouped)
        all_speeds += speeds
        seeds = sorted({int(c["seed"]) for c in payload["cells"]})
        for seed in seeds:
            trace = [
                next(
                    (c["achieved_release_speed_deg"] for c in grouped[s] if int(c["seed"]) == seed),
                    np.nan,
                )
                for s in speeds
            ]
            ax.plot(speeds, trace, color=style["color"], alpha=0.16, linewidth=0.8)
        means = [
            _mean(values=[c["achieved_release_speed_deg"] for c in grouped[s]]) for s in speeds
        ]
        ax.plot(
            speeds,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            label=f"{style['label']} — mean, n={len(seeds)}",
        )
    lo, hi = min(all_speeds), max(all_speeds)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=GREY,
        linestyle=":",
        linewidth=1.6,
        label="commanded (identity)",
    )
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("achieved joint-path speed at release (deg/s)")
    ax.set_title("The ceiling\nachieved speed vs what was asked for")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # --- panel 2: the dial's authority in metres ------------------------------
    ax = axes[1]
    for mode, payload in probes.items():
        style = MODE_STYLE[mode]
        grouped = cells_by_speed(payload=payload)
        speeds = list(grouped)
        seeds = sorted({int(c["seed"]) for c in payload["cells"]})
        for seed in seeds:
            trace = [
                next((c["range_m"] for c in grouped[s] if int(c["seed"]) == seed), np.nan)
                for s in speeds
            ]
            ax.plot(speeds, trace, color=style["color"], alpha=0.16, linewidth=0.8)
        means = [_mean(values=[c["range_m"] for c in grouped[s]]) for s in speeds]
        ax.plot(
            speeds,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            label=f"{style['label']} — mean, n={len(seeds)}",
        )
    if goal_box:
        base = _mean(
            values=[
                c["base_x_at_release"]
                for p in probes.values()
                for c in p["cells"]
                if c.get("base_x_at_release")
            ]
        )
        ax.axhspan(
            goal_box[0] - base,
            goal_box[1] - base,
            color=GREY,
            alpha=0.18,
            label="goal box, as a range from the base",
        )
    ax.axvline(UPSTREAM_DEFAULT_DEG, color=GREY, linestyle=":", linewidth=1.4)
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("cube resting distance from base (m)")
    ax.set_title("Dial authority\nhow far the parameter moves the cube")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # --- panel 3: is the label missable in both directions? -------------------
    ax = axes[2]
    for mode, payload in probes.items():
        style = MODE_STYLE[mode]
        grouped = cells_by_speed(payload=payload)
        speeds = list(grouped)
        n_seeds = len(sorted({int(c["seed"]) for c in payload["cells"]}))
        solved = [sum(1 for c in grouped[s] if c.get("solved")) for s in speeds]
        ax.plot(
            speeds,
            solved,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            marker="o",
            markersize=4,
            label=f"{style['label']} — of {n_seeds} seeds",
        )
    ax.axvline(
        UPSTREAM_DEFAULT_DEG,
        color=GREY,
        linestyle=":",
        linewidth=1.4,
        label="upstream's 140",
    )
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("seeds solved")
    ax.set_title("Missable in both directions?\nsolves per commanded speed (of 10 seeds)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # --- panel 4: R5, release-pose invariance ---------------------------------
    ax = axes[3]
    joint = 5  # joint 6, the largest mover along toss_dir
    for mode, payload in probes.items():
        style = MODE_STYLE[mode]
        grouped = cells_by_speed(payload=payload)
        speeds = list(grouped)
        seeds = sorted({int(c["seed"]) for c in payload["cells"]})
        for seed in seeds:
            trace = [
                next(
                    (
                        (c["achieved_release_conf_deg"] or [np.nan] * 7)[joint]
                        for c in grouped[s]
                        if int(c["seed"]) == seed
                    ),
                    np.nan,
                )
                for s in speeds
            ]
            ax.plot(speeds, trace, color=style["color"], alpha=0.16, linewidth=0.8)
        means = [
            _mean(
                values=[(c["achieved_release_conf_deg"] or [None] * 7)[joint] for c in grouped[s]]
            )
            for s in speeds
        ]
        ax.plot(
            speeds,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            label=f"{style['label']} — mean, n={len(seeds)}",
        )
    ax.set_xlabel("commanded release speed (deg/s)")
    ax.set_ylabel("joint 6 angle at release (deg)")
    ax.set_title("R5: is the release pose speed-invariant?\na flat line would mean yes")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    print(f"\nwrote {output_png}")


def main() -> None:
    args = _parse_args()
    probes: dict[str, dict[str, Any]] = {}
    for spec in args.probe:
        mode, _, path = spec.partition("=")
        probes[mode] = load_probe(path=Path(path))
    for mode, payload in probes.items():
        print_report(mode=mode, payload=payload)
    make_figure(probes=probes, output_png=args.output_png)


if __name__ == "__main__":
    main()
