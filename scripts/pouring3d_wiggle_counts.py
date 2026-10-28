"""Bounded recipe discovery and explicit 1–5 checks for centered tilt/wiggle."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class WiggleCounts:
    @staticmethod
    def probes(*, root: Path, material: str | None = None) -> None:
        plans: dict[str, list[tuple[int, float, float]]] = {
            "wood": [(4, 98.875, 2)],
            "foam": [(2, 107, 1.6), (2, 107, 2.8), (4, 107.375, 2)],
        }
        if material is not None:
            for index, (target, angle, duration) in enumerate(plans[material]):
                directory = root / material / "0/probes" / str(index)
                if directory.exists():
                    raise ValueError("Never overwrite an earlier probe")
                WiggleDemo.run(
                    directory=directory,
                    material=material,
                    angle=angle,
                    duration=duration,
                    requested_count=target,
                )
            return
        protocol = root / "targeted-probes-protocol.json"
        if protocol.exists():
            raise ValueError("Probe phase already planned")
        protocol.write_text(
            json.dumps(
                {
                    "reason": (
                        "Additional explicit requests for gaps after the eight-trial discovery"
                    ),
                    "seed": 0,
                    "plans": plans,
                    "separate_from_frozen_validation": True,
                },
                indent=2,
            )
        )
        repo = Path(__file__).resolve().parents[1]
        runs = [
            SweepRun(
                method=f"probe-{name}",
                seed=0,
                output_dir=root / f"probe-{name}/0",
                command=[
                    "run-heavy",
                    str(repo / "scripts/with_env.sh"),
                    sys.executable,
                    "-m",
                    "scripts.pouring3d_wiggle_counts",
                    "--root",
                    str(root),
                    "--probe-worker",
                    name,
                ],
            )
            for name in plans
        ]
        outcomes = SweepRunner.execute(runs=runs, max_workers=2)
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def select(*, rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
        eligible = [row for row in rows if row["eligible_count"] is not None]
        if not eligible:
            raise ValueError("No settled, contained, error-free candidate to replay")
        return min(
            eligible,
            key=lambda row: (
                abs(row["eligible_count"] - target),
                row["parameters"]["wiggle_seconds"],
                row["parameters"]["tilt_angle"],
            ),
        )

    @staticmethod
    def proposal(*, rows: list[dict[str, Any]], base: float, step: int) -> tuple[int, float, float]:
        eligible = [row for row in rows if row["eligible_count"] is not None]
        found = {row["eligible_count"] for row in eligible}
        missing = sorted(set(range(1, 6)) - found)
        target = missing[step % len(missing)] if missing else 2
        used = {
            (row["parameters"]["tilt_angle"], row["parameters"]["wiggle_seconds"]) for row in rows
        }
        intervals = []
        for left in eligible:
            for right in eligible:
                lp, rp = left["parameters"], right["parameters"]
                duration = round((lp["wiggle_seconds"] + rp["wiggle_seconds"]) / 2, 1)
                if (
                    lp["tilt_angle"] == rp["tilt_angle"]
                    and lp["wiggle_seconds"] < rp["wiggle_seconds"] - 0.1
                    and min(left["eligible_count"], right["eligible_count"])
                    < target
                    < max(left["eligible_count"], right["eligible_count"])
                    and (lp["tilt_angle"], duration) not in used
                ):
                    intervals.append((
                        rp["wiggle_seconds"] - lp["wiggle_seconds"],
                        lp["tilt_angle"],
                        duration,
                    ))
        if intervals:
            _, angle, duration = max(intervals)
            return target, angle, duration
        direction = -1 if target <= 2 else 1
        for delta, duration in ((0.75, 2), (1.5, 2), (0.75, 1), (0, 1.5), (0, 2.5)):
            angle = base + direction * delta
            if (angle, duration) not in used:
                return target, angle, duration
        raise ValueError("Bounded proposal list exhausted")

    @staticmethod
    def worker(*, root: Path, material: str) -> None:
        directory = root / material / "0"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "discovery").exists():
            raise ValueError("Use a new output root; never overwrite an earlier experiment")
        base = 97 if material == "wood" else 107
        rows: list[dict[str, Any]] = []
        angle: float
        # First explicitly attempt every requested count. Prior diagnostic
        # outcomes suggest the release transition, without assuming a monotone map.
        for target, angle, duration in (
            (1, base, 0.5),
            (2, base, 1),
            (3, base, 2),
            (4, base, 3),
            (5, 130, 0),
        ):
            rows.append(
                WiggleDemo.run(
                    directory=directory / "discovery" / str(len(rows)),
                    material=material,
                    angle=angle,
                    duration=duration,
                    requested_count=target,
                    place=False,
                )
            )
        # At most three refinements per material. These are between-trial
        # choices; the physical Pour itself has no count feedback or early exit.
        for step in range(3):
            if set(range(1, 6)) <= {row["eligible_count"] for row in rows}:
                break
            target, angle, duration = WiggleCounts.proposal(rows=rows, base=base, step=step)
            rows.append(
                WiggleDemo.run(
                    directory=directory / "discovery" / str(len(rows)),
                    material=material,
                    angle=angle,
                    duration=duration,
                    requested_count=target,
                    place=False,
                )
            )
        recipes = {
            str(target): {
                "parameters": WiggleCounts.select(rows=rows, target=target)["parameters"],
                "discovery_count": WiggleCounts.select(rows=rows, target=target)["eligible_count"],
                "exact_candidate_found": any(row["eligible_count"] == target for row in rows),
            }
            for target in range(1, 6)
        }
        (directory / "recipes.json").write_text(json.dumps(recipes, indent=2))
        for seed in range(3):
            for target in range(1, 6):
                parameters = recipes[str(target)]["parameters"]
                WiggleDemo.run(
                    directory=root / material / str(seed) / "validation" / str(target),
                    material=material,
                    angle=parameters["tilt_angle"],
                    duration=parameters["wiggle_seconds"],
                    seed=seed,
                    requested_count=target,
                )

    @staticmethod
    def launch(*, root: Path) -> None:
        repo = Path(__file__).resolve().parents[1]
        root.mkdir(parents=True, exist_ok=False)
        source = root / "source"
        source.mkdir()
        sources = [
            Path(__file__).resolve(),
            repo / "scripts/pouring3d_wiggle_demo.py",
            repo / "src/hitl_pmp/environments/pouring3d/wiggle.py",
        ]
        for path in sources:
            shutil.copyfile(path, source / path.name)
        (root / "protocol.json").write_text(
            json.dumps(
                {
                    "requested_counts": list(range(1, 6)),
                    "materials": ["wood", "foam"],
                    "discovery_seed": 0,
                    "discovery_budget_per_material": 8,
                    "validation_seeds": [0, 1, 2],
                    "validation_trials": 30,
                    "recipe_selection": (
                        "Closest settled count; ties use shorter wiggle, then lower tilt"
                    ),
                    "missing_recipe": "Replay closest observed candidate; report as unresolved",
                    "fixed_settings": (
                        "Centered; amplitude 4 degrees; 1 Hz; 10 s outward/return; 2–10 s settling"
                    ),
                    "discovery_placement": False,
                    "validation_placement": True,
                    "source_sha256": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
                    },
                },
                indent=2,
            )
        )
        runs = [
            SweepRun(
                method=material,
                seed=0,
                output_dir=root / material / "0",
                command=[
                    "run-heavy",
                    str(repo / "scripts/with_env.sh"),
                    sys.executable,
                    "-m",
                    "scripts.pouring3d_wiggle_counts",
                    "--root",
                    str(root),
                    "--worker",
                    material,
                ],
            )
            for material in ("wood", "foam")
        ]
        outcomes = SweepRunner.execute(runs=runs, max_workers=2)
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def report(*, root: Path) -> None:
        rows: list[dict[str, Any]] = []
        discoveries: list[dict[str, Any]] = []
        probes = [
            json.loads(path.read_text()) for path in sorted(root.glob("*/0/probes/*/result.json"))
        ]
        recipes = {}
        for material in ("wood", "foam"):
            recipes[material] = json.loads((root / material / "0/recipes.json").read_text())
            discoveries.extend(
                json.loads(path.read_text())
                for path in sorted((root / material / "0/discovery").glob("*/result.json"))
            )
            for seed in range(3):
                for target in range(1, 6):
                    row = json.loads(
                        (
                            root / material / str(seed) / "validation" / str(target) / "result.json"
                        ).read_text()
                    )
                    assert row["requested_count"] == target and row["seed"] == seed
                    assert row["parameters"] == recipes[material][str(target)]["parameters"]
                    rows.append(row)
        checked = 0
        for row in [*discoveries, *rows, *probes]:
            assert row["initial"]["locations"] == ["pitcher"] * 5
            assert row["source_sha256"] == discoveries[0]["source_sha256"]
            for event in row["events"]:
                for snapshot in (event["before"], event["after"]):
                    target = row["requested_count"]
                    assert snapshot["requested_count"] == target
                    expected = (
                        snapshot["settled"]
                        and snapshot["locations"].count("tray") == target
                        and snapshot["locations"].count("pitcher") == 5 - target
                    )
                    assert snapshot["goal"] == expected
                    checked += 1
        summary = {
            "recipes": recipes,
            "discovery_trials": len(discoveries),
            "validation_trials": len(rows),
            "targeted_probe_trials": len(probes),
            "goal_snapshots_checked": checked,
            "source_hashes_identical": True,
            "discovery": [
                {
                    key: row[key]
                    for key in (
                        "material",
                        "seed",
                        "requested_count",
                        "parameters",
                        "eligible_count",
                        "pour_success",
                    )
                }
                for row in discoveries
            ],
            "targeted_probes": [
                {
                    key: row[key]
                    for key in (
                        "material",
                        "seed",
                        "requested_count",
                        "parameters",
                        "eligible_count",
                        "pour_success",
                        "full_success",
                    )
                }
                for row in probes
            ],
            "validation": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "material",
                            "seed",
                            "requested_count",
                            "parameters",
                            "eligible_count",
                            "pour_success",
                            "full_success",
                            "source_sha256",
                        )
                    },
                    "after_pour_locations": row["after_pour"]["locations"],
                    "after_pour_settled": row["after_pour"]["settled"],
                    "events": [
                        {key: event[key] for key in ("skill", "success", "error")}
                        for event in row["events"]
                    ],
                }
                for row in rows
            ],
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
        for axis, material in zip(axes, ("wood", "foam"), strict=True):
            axis.plot([1, 5], [1, 5], color="#999999", linestyle=":", label="Requested count")
            for seed, marker in enumerate(("o", "s", "^")):
                subset = [
                    row for row in rows if row["material"] == material and row["seed"] == seed
                ]
                axis.scatter(
                    [row["requested_count"] + (seed - 1) * 0.12 for row in subset],
                    [row["after_pour"]["locations"].count("tray") for row in subset],
                    color="#CC79A7",
                    marker=marker,
                    label=f"Start {seed}",
                    s=50,
                )
            for target in range(1, 6):
                subset = [
                    row
                    for row in rows
                    if row["material"] == material and row["requested_count"] == target
                ]
                axis.text(
                    target, 5.6, f"{sum(row['pour_success'] for row in subset)}/3", ha="center"
                )
            extra = [row for row in probes if row["material"] == material]
            if extra:
                axis.scatter(
                    [row["requested_count"] for row in extra],
                    [row["after_pour"]["locations"].count("tray") for row in extra],
                    marker="*",
                    s=150,
                    color="#CC79A7",
                    edgecolor="black",
                    label="Extra seed-0 probes",
                )
            axis.set(
                title=f"{material.title()} | exact Pour successes shown above",
                xlabel="Requested count",
                ylabel="Delivered tray count",
                xticks=range(1, 6),
                yticks=range(6),
                ylim=(-0.3, 6),
                xlim=(0.6, 5.4),
            )
            axis.grid(alpha=0.2)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
        figure.suptitle("Centered tilt/wiggle: frozen recipes on three starts per material")
        figure.savefig(root / "counts-results.png", dpi=160)
        plt.close(figure)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--probe-worker", choices=("wood", "foam"))
        parser.add_argument("--probes", action="store_true")
        parser.add_argument("--report-only", action="store_true")
        args = parser.parse_args()
        if args.probes or args.probe_worker:
            WiggleCounts.probes(root=args.root, material=args.probe_worker)
        elif args.worker:
            WiggleCounts.worker(root=args.root, material=args.worker)
        else:
            if not args.report_only:
                WiggleCounts.launch(root=args.root)
            WiggleCounts.report(root=args.root)


if __name__ == "__main__":
    WiggleCounts.main()
