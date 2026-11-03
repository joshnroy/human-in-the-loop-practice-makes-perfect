"""Plot recorded passive diagnostics; never start a simulator."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve earlier figures; choose a new output")
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    for path, label, color in (
        (args.before, "Before: rolling resistance inactive", "#e76f51"),
        (args.after, "After: rolling contacts enabled", "#168aad"),
    ):
        report = json.loads(path.read_text())
        for column, case in enumerate(report["cases"]):
            time = [sample["seconds"] for sample in case["trace"]]
            velocities = np.array([sample["velocity"] for sample in case["trace"]])
            for row, components in enumerate((velocities[:, :3], velocities[:, 3:])):
                axes[row, column].semilogy(
                    time,
                    np.maximum(np.linalg.norm(components, axis=1), 1e-8),
                    label=label,
                    color=color,
                )
                axes[row, column].grid(alpha=0.2)
            axes[0, column].set_title(
                f"{case['tilt_degrees']}° tilt / {1000 * case['drop_m']:g} mm drop"
            )
            axes[1, column].set_xlabel("Simulated seconds")
    axes[0, 0].set_ylabel("Linear speed (m/s, log scale)")
    axes[1, 0].set_ylabel("Angular speed (rad/s, log scale)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Same rigid food proxy, tray and initial velocity: passive settling A/B\n"
        "Privileged drop setup — not robot delivery or calibrated cooked-food physics",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
