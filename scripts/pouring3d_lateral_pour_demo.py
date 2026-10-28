"""Try shallow held tilts with the approved sideways Pour macro and render it."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class LateralPourDemo:
    @staticmethod
    def worker(*, root: Path, material: str) -> None:
        directory = root / material / "0"
        angles = [95.5, 97, 99] if material == "wood" else [105, 107, 109]
        rows: list[dict[str, Any]] = []
        for angle in angles:
            rows.append(
                WiggleDemo.run(
                    directory=directory / f"trial-{len(rows)}",
                    material=material,
                    angle=angle,
                    duration=12,
                    mode="lateral",
                    requested_count=2,
                    place=False,
                )
            )
        # Only refine an empty-to-full jump if none of the initial pours is partial.
        for _ in range(2):
            if any(row["eligible_count"] in {1, 2, 3, 4} for row in rows):
                break
            ordered = sorted(rows, key=lambda row: row["parameters"]["tilt_angle"])
            intervals = [
                (a, b)
                for a, b in zip(ordered, ordered[1:], strict=False)
                if a["eligible_count"] == 0 and b["eligible_count"] == 5
            ]
            if not intervals:
                break
            a, b = min(
                intervals,
                key=lambda pair: (
                    pair[1]["parameters"]["tilt_angle"] - pair[0]["parameters"]["tilt_angle"]
                ),
            )
            angle = (a["parameters"]["tilt_angle"] + b["parameters"]["tilt_angle"]) / 2
            rows.append(
                WiggleDemo.run(
                    directory=directory / f"trial-{len(rows)}",
                    material=material,
                    angle=angle,
                    duration=12,
                    mode="lateral",
                    requested_count=2,
                    place=False,
                )
            )
        contained = [row for row in rows if row["eligible_count"] is not None]
        nonempty = [row for row in contained if row["eligible_count"] > 0]
        selected = min(
            nonempty or contained,
            key=lambda row: (abs(row["eligible_count"] - 2), row["parameters"]["tilt_angle"]),
        )
        params = selected["parameters"]
        rendered = WiggleDemo.run(
            directory=directory / "video",
            material=material,
            angle=params["tilt_angle"],
            duration=params["wiggle_seconds"],
            mode="lateral",
            requested_count=2,
            video=True,
        )
        for point in ("initial", "after_pour"):
            for field in ("qpos", "qvel", "ctrl", "locations", "settled", "goal"):
                assert rendered[point][field] == selected[point][field], (point, field)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "material": material,
                    "trials": rows,
                    "selected_parameters": params,
                    "rendered": rendered,
                    "video_replay_exact_through_pour": True,
                },
                indent=2,
            )
        )

    @staticmethod
    def launch(*, root: Path) -> None:
        if root.exists():
            raise ValueError("Use a new result directory")
        root.mkdir(parents=True)
        (root / "protocol.json").write_text(
            json.dumps(
                {
                    "requested_count": 2,
                    "seed": 0,
                    "materials": ["wood", "foam"],
                    "initial_angles": {"wood": [95.5, 97, 99], "foam": [105, 107, 109]},
                    "wiggle_seconds": 12,
                    "maximum_refinements_per_material": 2,
                    "refinement": "Bisect empty/full jumps when no partial count was found",
                    "render_selection": "Closest nonempty count to two; lower tilt on ties",
                    "note": "Bounded feeding diagnostic; full count feasibility is untested",
                },
                indent=2,
            )
        )
        repo = Path(__file__).resolve().parents[1]
        results = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=material,
                    seed=0,
                    output_dir=root / material / "0",
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_lateral_pour_demo",
                        "--root",
                        str(root),
                        "--worker",
                        material,
                    ],
                )
                for material in ("wood", "foam")
            ],
            max_workers=2,
        )
        if not all(result.succeeded for result in results):
            raise SystemExit(1)

    @staticmethod
    def report(*, root: Path) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        summaries = {}
        for col, material in enumerate(("wood", "foam")):
            summary = json.loads((root / material / "0/summary.json").read_text())
            summaries[material] = summary
            shades = plt.get_cmap("PuRd")(np.linspace(0.4, 0.9, len(summary["trials"])))
            for trial, color in zip(summary["trials"], shades, strict=True):
                event = trial["events"][1]
                wave = event["wiggle_trace"]
                label = f"{trial['parameters']['tilt_angle']:g} deg"
                axes[0, col].plot(
                    [r["elapsed"] for r in wave],
                    [r["locations"].count("tray") for r in wave],
                    label=label,
                    color=color,
                )
            axes[0, col].set(
                title=f"{material.title()} | seed 0 | five cubes total",
                xlabel="Wiggle time (s)",
                ylabel="Tray count",
                yticks=range(6),
            )
            axes[0, col].legend()
            rows = summary["trials"]
            axes[1, col].scatter(
                [r["parameters"]["tilt_angle"] for r in rows],
                [r["after_pour"]["locations"].count("tray") for r in rows],
                c=shades,
            )
            axes[1, col].set(xlabel="Held tilt (deg)", ylabel="Final tray count", yticks=range(6))
        figure.savefig(root / "feeding-results.png", dpi=150)
        plt.close(figure)
        (root / "summary.json").write_text(json.dumps(summaries, indent=2))

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        args = parser.parse_args()
        if args.worker:
            LateralPourDemo.worker(root=args.root, material=args.worker)
        else:
            LateralPourDemo.launch(root=args.root)
            LateralPourDemo.report(root=args.root)


if __name__ == "__main__":
    LateralPourDemo.main()
