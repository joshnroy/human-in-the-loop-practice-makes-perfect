"""Plot measured initial poses written by scripts/tossing3d_layout_demo.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_layouts(*, poses_path: Path, output_path: Path) -> None:
    records = json.loads(poses_path.read_text())
    fig, axes = plt.subplots(1, len(records), figsize=(10, 5), squeeze=False)
    for ax, record in zip(axes[0], records, strict=True):
        objects = record["objects"]
        for name, pose in objects.items():
            if name == "cuboid_barrier":
                ax.add_patch(
                    Rectangle(
                        (pose["x"] - 0.015, pose["y"] - 2.5), 0.03, 5, color="0.5", label="barrier"
                    )
                )
            elif name == "bin_0":
                ax.add_patch(
                    Rectangle(
                        (pose["x"] - 0.15, pose["y"] - 0.15),
                        0.3,
                        0.3,
                        fill=False,
                        edgecolor="#D55E00",
                        linewidth=2,
                        label="bin",
                    )
                )
            else:
                ax.scatter(
                    pose["x"], pose["y"], label=name, s=65, marker="s" if name == "cube_0" else "o"
                )
        ax.set(
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{record['layout']} — measured reset, seed {record['seed']}",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left")
    fig.suptitle("Tossing3D: original and same-side layouts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_layouts(poses_path=args.poses, output_path=args.output)
