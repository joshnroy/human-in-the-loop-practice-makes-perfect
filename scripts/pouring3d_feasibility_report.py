"""Validate selected physical count-pour recipes and plot the bounded search."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from scripts.pouring3d_feasibility import PourFeasibility
from scripts.pouring3d_refine import CountRefinement
from scripts.run_sweep import SweepRun, SweepRunner


class FeasibilityReport:
    @staticmethod
    def validate(*, directory: Path, root: Path, material: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        search = CountRefinement.rows(root=root, material=material)
        valid = [
            row
            for row in search
            if len(row["events"]) >= 2
            and row["events"][0]["success"]
            and row["events"][1]["skill"] == "Pour"
            and row["events"][1]["error"] is None
            and any(
                row[boundary]["settled"]
                and all(location in {"pitcher", "tray"} for location in row[boundary]["locations"])
                for boundary in ("after_pour", "after_place")
            )
        ]
        path = directory / "validation.jsonl"
        history = (
            [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        )
        for target in range(1, 6):
            # Choose using development outcomes only, freeze across validation seeds.
            chosen = min(
                valid,
                key=lambda row: (
                    abs(row["after_pour"]["locations"].count("tray") - target),
                    row["feasible_count"] != target,
                    not row["after_pour"]["settled"],
                    row["trial"],
                ),
            )
            for seed in (0, 1, 2):
                if any(row["target"] == target and row["seed"] == seed for row in history):
                    continue
                row = PourFeasibility.trial(
                    material=material,
                    seed=seed,
                    parameters=PourParameters(**chosen["parameters"]),
                    target=target,
                )
                row["search_trial"] = chosen["trial"]
                row["search_exact_candidate"] = chosen["feasible_count"] == target
                row["pour_success"] = (
                    len(row["events"]) >= 2
                    and row["events"][1]["skill"] == "Pour"
                    and row["events"][1]["success"]
                    and row["after_pour"]["goal"]
                )
                row["exact_success"] = (
                    row["feasible_count"] == target
                    and row["after_pour"]["goal"]
                    and row["after_place"]["goal"]
                )
                history.append(row)
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
        (directory / "validation-results.json").write_text(
            json.dumps(
                {
                    "material": material,
                    "seeds": [0, 1, 2],
                    "results": [
                        {
                            "target": target,
                            "parameters": next(
                                row["parameters"] for row in history if row["target"] == target
                            ),
                            "outcomes": [
                                {
                                    "seed": row["seed"],
                                    "success": row["exact_success"],
                                    "pour_success": row["pour_success"],
                                    "tray_after_pour": row["after_pour"]["locations"].count("tray"),
                                    "tray": row["after_place"]["locations"].count("tray"),
                                    "pitcher": row["after_place"]["locations"].count("pitcher"),
                                    "settled": row["after_place"]["settled"],
                                    "completed": row["completed"],
                                }
                                for row in history
                                if row["target"] == target
                            ],
                        }
                        for target in range(1, 6)
                    ],
                    "interpretation": (
                        "Seed 0 replays development; seeds 1 and 2 are new initial jitter. "
                        "These checks do not measure robustness to general reset arrangements."
                    ),
                },
                indent=2,
            )
        )

    @staticmethod
    def launch(*, root: Path, materials: tuple[str, ...] = ("wood", "foam")) -> None:
        repo = Path(__file__).resolve().parents[1]
        runs = [
            SweepRun(
                method=f"validate-{material}",
                seed=0,
                output_dir=root / f"validate-{material}" / "0",
                command=[
                    "run-heavy",
                    str(repo / "scripts/with_env.sh"),
                    sys.executable,
                    "-m",
                    "scripts.pouring3d_feasibility_report",
                    "--worker",
                    material,
                    "--root",
                    str(root),
                ],
            )
            for material in materials
        ]
        outcomes = SweepRunner.execute(runs=runs, max_workers=2)
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def confirm_wood_four(*, root: Path) -> None:
        """Repeat the seed-2 discovery with an explicit four-cube request."""
        repo = Path(__file__).resolve().parents[1]
        directory = root / "confirm-wood-four" / "2"
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method="confirm-wood-four",
                    seed=2,
                    output_dir=directory,
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_native_demo",
                        "--output-dir",
                        str(directory),
                        "--seed",
                        "2",
                        "--requested-count",
                        "4",
                        "--angle",
                        "102.5",
                        "--tilt-seconds",
                        "10",
                        "--hold-seconds",
                        "0.2",
                        "--cube-material",
                        "wood",
                        "--pitcher-reference",
                        "winco",
                        "--reset",
                        "none",
                        "--no-video",
                    ],
                )
            ],
            max_workers=1,
        )
        if not outcomes[0].succeeded:
            raise SystemExit(1)

    @staticmethod
    def plot(*, root: Path) -> dict[str, Any]:
        result: dict[str, Any] = {}
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for column, material in enumerate(("wood", "foam")):
            search = CountRefinement.rows(root=root, material=material)
            validation = json.loads(
                (root / f"validate-{material}/0/validation-results.json").read_text()
            )
            ax = axes[0, column]
            rows = [row for row in search if CountRefinement.pour_count(row=row) is not None]
            scatter = ax.scatter(
                [row["parameters"]["angle"] for row in rows],
                [CountRefinement.pour_count(row=row) for row in rows],
                c=[row["parameters"]["tilt_seconds"] for row in rows],
                cmap="viridis",
                vmin=2,
                vmax=15,
                s=45,
                alpha=0.8,
            )
            rejected = [row for row in search if CountRefinement.pour_count(row=row) is None]
            if rejected:
                ax.scatter(
                    [row["parameters"]["angle"] for row in rejected],
                    [row["after_pour"]["locations"].count("tray") for row in rejected],
                    marker="x",
                    color="black",
                    s=50,
                    label="Handling / spill / settling failure",
                )
            ax.set(
                title=f"{material.title()}: {len(search)} development trials",
                xlabel="Maximum commanded tilt (degrees)",
                ylabel="Tray count after Pour",
                yticks=range(6),
                ylim=(-0.3, 5.3),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="center right")
            ax = axes[1, column]
            counts = [
                [
                    str(outcome["tray_after_pour"])
                    + ("*" if outcome["pour_success"] and not outcome["success"] else "")
                    for outcome in row["outcomes"]
                ]
                for row in validation["results"]
            ]
            colors = [
                ["#8bd1a5" if outcome["pour_success"] else "#f4b087" for outcome in row["outcomes"]]
                for row in validation["results"]
            ]
            table = ax.table(
                cellText=counts,
                cellColours=colors,
                rowLabels=[f"Target {count}" for count in range(1, 6)],
                colLabels=["Seed 0 (replay)", "Seed 1", "Seed 2"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            ax.set_title(
                "Frozen settings: tray counts after Pour\n"
                "Green = successful Pour; * = placement/completion issue"
            )
            ax.axis("off")
            result[material] = {
                "search_trials": len(search),
                "found_counts": sorted({
                    row["feasible_count"] for row in search if row["feasible_count"] in range(1, 6)
                }),
                "found_pour_counts": sorted({
                    row["after_pour"]["locations"].count("tray")
                    for row in search
                    if len(row["events"]) >= 2
                    and row["events"][1]["skill"] == "Pour"
                    and row["events"][1]["error"] is None
                    and row["after_pour"]["settled"]
                    and all(
                        location in {"pitcher", "tray"}
                        for location in row["after_pour"]["locations"]
                    )
                    and row["after_pour"]["locations"].count("tray") in range(1, 6)
                }),
                "validation": validation,
            }
            settling_path = root / f"{material}-four-settling/0/settling-results.jsonl"
            if settling_path.exists():
                settling = [json.loads(line) for line in settling_path.read_text().splitlines()]
                result[material]["settling_supplement"] = [
                    {
                        "seed": row["seed"],
                        "wait_seconds": row["wait_seconds"],
                        "pour_success": row["pour_success"],
                        "pour_plus_wait_success": row["pour_plus_wait_success"],
                        "full_completion_success": row["full_completion_success"],
                        "tray_after_wait": row["after_wait"]["locations"].count("tray"),
                        "pitcher_after_wait": row["after_wait"]["locations"].count("pitcher"),
                    }
                    for row in settling
                ]
                successes = sum(row["pour_plus_wait_success"] for row in settling)
                waits = ", ".join(f"{row['wait_seconds']:.0f}" for row in settling)
                note = (
                    f"Separate four-cube + Wait: {successes}/{len(settling)} exact; waits {waits} s"
                )
                confirmation_path = root / "confirm-wood-four/2/native-results.json"
                if material == "wood" and confirmation_path.exists():
                    confirmation = json.loads(confirmation_path.read_text())
                    assert confirmation["completed"] and confirmation["final"]["goal"]
                    result[material]["four_cube_confirmation"] = {
                        "seed": 2,
                        "parameters": confirmation["pour_parameters"],
                        "native_pour_success": confirmation["events"][1]["success"],
                        "full_completion_success": confirmation["completed"],
                        "extra_wait_seconds": 0,
                    }
                    note += "\nSeed 2 also confirmed four with native Pour, no extra Wait."
                ax.text(
                    0.5,
                    0.02,
                    note,
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
        figure.colorbar(
            scatter,
            ax=axes[0, :].tolist(),
            label="Outward / return duration (s)",
            shrink=0.8,
            pad=0.02,
        )
        figure.suptitle(
            "Winco pitcher: requested-count feasibility with approved physics", fontsize=15
        )
        figure.savefig(root / "feasibility-results.png", dpi=170)
        plt.close(figure)
        (root / "feasibility-results.json").write_text(json.dumps(result, indent=2))
        return result

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--material", choices=("wood", "foam"))
        parser.add_argument("--plot-only", action="store_true")
        parser.add_argument("--confirm-wood-four", action="store_true")
        args = parser.parse_args()
        if args.confirm_wood_four:
            FeasibilityReport.confirm_wood_four(root=args.root)
        elif args.worker:
            FeasibilityReport.validate(
                directory=args.root / f"validate-{args.worker}" / "0",
                root=args.root,
                material=args.worker,
            )
        else:
            if not args.plot_only:
                FeasibilityReport.launch(
                    root=args.root,
                    materials=(args.material,) if args.material else ("wood", "foam"),
                )
            if all(
                (args.root / f"validate-{material}/0/validation-results.json").exists()
                for material in ("wood", "foam")
            ):
                FeasibilityReport.plot(root=args.root)


if __name__ == "__main__":
    FeasibilityReport.main()
