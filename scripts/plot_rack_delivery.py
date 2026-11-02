"""Annotated measured-pose diagram for a recorded rack delivery, without simulation."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    trial = json.loads(args.trial.read_text())
    state, config = trial["final"], trial["config"]
    tray = np.array(state["poses"]["box_0"][:3])
    edge = tray[0] + config["tray_depth"] / 2
    top = tray[2] + config["tray_thickness"]
    radii = (
        np.array([config["dumpling_depth"], config["dumpling_width"], config["dumpling_height"]])
        / 2
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].add_patch(
        Rectangle(
            (tray[0] - config["tray_depth"] / 2, tray[1] - config["tray_width"] / 2),
            config["tray_depth"],
            config["tray_width"],
            color="#e2e8f0",
        )
    )
    axes[1].add_patch(
        Rectangle(
            (tray[0] - config["tray_depth"] / 2, tray[2]),
            config["tray_depth"],
            config["tray_thickness"],
            color="#94a3b8",
        )
    )
    overhangs = []
    for name, location in state["locations"].items():
        pose = np.array(state["poses"][name])
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        extent = np.linalg.norm(rotation * radii, axis=1)
        overhangs.append((max(0, pose[0] + extent[0] - edge), name))
        color = "#168aad" if location == "OnTray" else "#e76f51"
        for axis, components in zip(axes, ((0, 1), (0, 2)), strict=True):
            projected = rotation[list(components)] * radii
            values, vectors = np.linalg.eigh(projected @ projected.T)
            direction = vectors[:, -1]
            axis.add_patch(
                Ellipse(
                    pose[list(components)],
                    2 * np.sqrt(values[-1]),
                    2 * np.sqrt(values[0]),
                    angle=np.rad2deg(np.arctan2(direction[1], direction[0])),
                    facecolor=color,
                    edgecolor="#334155",
                    alpha=0.7,
                )
            )
        axes[0].text(pose[0], pose[1], name.replace("cube_", "food "), ha="center", fontsize=8)
    for axis in axes:
        axis.axvline(edge, color="#dc2626", linestyle="--", label="Measured tray edge")
        axis.set_xlabel("World x (m)")
        axis.grid(alpha=0.15)
        axis.set_aspect("equal")
    axes[0].set(xlim=(1.23, 1.39), ylim=(-0.12, 0.12), ylabel="World y (m)", title="Top view")
    axes[1].set(
        xlim=(1.26, 1.38), ylim=(0.895, 0.95), ylabel="World z (m)", title="Side projection"
    )
    axes[1].axhline(top, color="#64748b", linestyle=":", label="Measured tray top")
    axes[1].legend(fontsize=8, loc="upper left")
    overhang, name = max(overhangs)
    supported = "box_0" in state["supports"][name]
    figure.suptitle(
        f"{name.replace('cube_', 'Food ')}: {overhang * 1000:.1f} mm overhang; "
        f"actual tray support = {supported}\n"
        "Strict full-footprint scoring rejects this — it is NOT a floor/counter spill",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
