"""Bounded 1–5 discovery, frozen-recipe checks, and native sideways-pour videos."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pouring3d_wiggle_counts import WiggleCounts
from scripts.pouring3d_wiggle_demo import WiggleDemo
from scripts.run_sweep import SweepRun, SweepRunner


class LateralCounts:
    DISCOVERY_BUDGET: ClassVar[int] = 18

    @staticmethod
    def proposal(
        *, rows: list[dict[str, Any]], material: str, step: int
    ) -> tuple[int, float, float]:
        eligible = [row for row in rows if row["eligible_count"] is not None]
        missing = sorted(set(range(1, 6)) - {row["eligible_count"] for row in eligible})
        used = {
            (row["parameters"]["tilt_angle"], row["parameters"]["wiggle_seconds"]) for row in rows
        }
        candidates = []
        for target in missing:
            for left in eligible:
                for right in eligible:
                    if not left["eligible_count"] < target < right["eligible_count"]:
                        continue
                    a, b = left["parameters"], right["parameters"]
                    if a["wiggle_seconds"] == b["wiggle_seconds"]:
                        angle = round((a["tilt_angle"] + b["tilt_angle"]) / 2, 4)
                        duration = a["wiggle_seconds"]
                        width = abs(a["tilt_angle"] - b["tilt_angle"])
                    elif a["tilt_angle"] == b["tilt_angle"]:
                        angle = a["tilt_angle"]
                        duration = round((a["wiggle_seconds"] + b["wiggle_seconds"]) / 2, 1)
                        width = abs(a["wiggle_seconds"] - b["wiggle_seconds"])
                    else:
                        continue
                    if width < 0.15 or (angle, duration) in used:
                        continue
                    candidates.append((
                        right["eligible_count"] - left["eligible_count"],
                        width,
                        (missing.index(target) - step) % len(missing),
                        target,
                        angle,
                        duration,
                    ))
        if candidates:
            *_, target, angle, duration = min(candidates)
            return target, angle, duration
        base = 95.5 if material == "wood" else 105
        # A finite fallback grid covers the release boundary and longer feeding.
        for delta in (0, -0.5, 0.5, 1, -1, 1.5, 2):
            for duration in (0, 4, 8, 16, 20, 2, 12):
                if (base + delta, duration) not in used:
                    return missing[step % len(missing)], base + delta, duration
        raise ValueError("Bounded sideways proposal grid exhausted")

    @staticmethod
    def render_choice(*, rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
        return min(
            rows,
            key=lambda row: (
                not row["pour_success"],
                row["eligible_count"] is None,
                abs((row["eligible_count"] if row["eligible_count"] is not None else -5) - target),
                not row["full_success"],
                row["seed"],
            ),
        )

    @staticmethod
    def worker(*, root: Path, material: str) -> None:
        directory = root / material / "0"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "discovery").exists():
            raise ValueError("Use a new output root")
        plans = (
            [(1, 95.5, 2.5), (2, 95.5, 12), (3, 97, 2), (4, 97, 6), (5, 97, 12)]
            if material == "wood"
            else [(1, 104.5, 12), (2, 105, 12), (3, 106, 12), (4, 106.5, 12), (5, 107, 12)]
        )
        rows: list[dict[str, Any]] = []
        for index in range(LateralCounts.DISCOVERY_BUDGET):
            if index < len(plans):
                target, angle, duration = plans[index]
            else:
                if set(range(1, 6)) <= {row["eligible_count"] for row in rows}:
                    break
                target, angle, duration = LateralCounts.proposal(
                    rows=rows, material=material, step=index - len(plans)
                )
            rows.append(
                WiggleDemo.run(
                    directory=directory / "discovery" / str(index),
                    material=material,
                    angle=angle,
                    duration=duration,
                    mode="lateral",
                    requested_count=target,
                    place=False,
                )
            )
        recipes = {}
        for target in range(1, 6):
            selected = WiggleCounts.select(rows=rows, target=target)
            recipes[str(target)] = {
                "parameters": selected["parameters"],
                "discovery_count": selected["eligible_count"],
                "exact_candidate_found": any(row["eligible_count"] == target for row in rows),
            }
        (directory / "recipes.json").write_text(json.dumps(recipes, indent=2))
        for seed in range(3):
            for target in range(1, 6):
                params = recipes[str(target)]["parameters"]
                WiggleDemo.run(
                    directory=root / material / str(seed) / "validation" / str(target),
                    material=material,
                    angle=params["tilt_angle"],
                    duration=params["wiggle_seconds"],
                    mode="lateral",
                    requested_count=target,
                    seed=seed,
                )

    @staticmethod
    def render(*, root: Path, material: str, include_probes: bool = False) -> None:
        for target in range(1, 6):
            paths = [
                root / material / str(seed) / "validation" / str(target) / "result.json"
                for seed in range(3)
            ]
            if include_probes:
                paths.extend(sorted((root / material / "0/probes").glob("*/result.json")))
            choices = []
            for path in paths:
                row = json.loads(path.read_text())
                if row["requested_count"] == target:
                    choices.append({**row, "reference_path": str(path)})
            selected = LateralCounts.render_choice(rows=choices, target=target)
            params = selected["parameters"]
            directory = root / f"video-{material}" / "0" / str(target)
            if directory.exists():
                raise ValueError("Never overwrite a native video replay")
            result = WiggleDemo.run(
                directory=directory,
                material=material,
                angle=params["tilt_angle"],
                duration=params["wiggle_seconds"],
                mode="lateral",
                requested_count=target,
                seed=selected["seed"],
                video=True,
            )
            for name in ("initial", "after_pour", "final"):
                for field in ("qpos", "qvel", "ctrl", "locations", "settled", "goal"):
                    assert result[name][field] == selected[name][field], (name, field)
            original = Path(selected["reference_path"])
            trace = json.loads(original.with_name("trace.json").read_text())
            assert json.loads((directory / "trace.json").read_text()) == trace
            (directory / "replay-check.json").write_text(
                json.dumps({
                    "exact": True,
                    "motor_trace_rows": len(trace),
                    "seed": selected["seed"],
                    "reference_path": str(original),
                    "targeted_probe": original.parent.parent.name == "probes",
                })
            )

    @staticmethod
    def launch(*, root: Path, render: bool = False) -> None:
        repo = Path(__file__).resolve().parents[1]
        if not render:
            root.mkdir(parents=True, exist_ok=False)
            source = root / "source"
            source.mkdir()
            paths = [
                Path(__file__).resolve(),
                repo / "scripts/pouring3d_wiggle_demo.py",
                repo / "scripts/pouring3d_wiggle_counts.py",
                repo / "src/hitl_pmp/environments/pouring3d/lateral_wiggle.py",
                repo / "src/hitl_pmp/environments/pouring3d/lateral_environment.py",
            ]
            for path in paths:
                shutil.copy2(path, source / path.name)
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "controller": "horizontal_lateral_wiggle_v1",
                        "requested_counts": list(range(1, 6)),
                        "materials": ["wood", "foam"],
                        "discovery_seed": 0,
                        "discovery_budget_per_material": LateralCounts.DISCOVERY_BUDGET,
                        "validation_seeds": [0, 1, 2],
                        "validation_trials": 30,
                        "discovery_placement": False,
                        "validation_placement": True,
                        "selection": "Closest contained settled count, shorter duration/lower tilt",
                        "missing_recipe": "Validate closest candidate and report unresolved",
                        "refinement": "Bisect observed count gaps; finite boundary grid otherwise",
                        "adaptive_count_feedback_during_pour": False,
                        "seed_zero_overlap": "Seed 0 is the discovery start; seeds 1/2 are new",
                        "source_sha256": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                        },
                    },
                    indent=2,
                )
            )
        outcomes = SweepRunner.execute(
            runs=[
                SweepRun(
                    method=(f"video-{m}" if render else m),
                    seed=0,
                    output_dir=root / (f"video-{m}" if render else m) / "0",
                    command=[
                        "run-heavy",
                        str(repo / "scripts/with_env.sh"),
                        sys.executable,
                        "-m",
                        "scripts.pouring3d_lateral_counts",
                        "--root",
                        str(root),
                        "--render-worker" if render else "--worker",
                        m,
                    ],
                )
                for m in ("wood", "foam")
            ],
            max_workers=2,
        )
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def report(*, root: Path) -> None:
        # Reuse the independently audited scoring/summary pipeline. Replace only
        # its presentation, which names the historical pitch-wiggle controller.
        WiggleCounts.report(root=root)
        summary = json.loads((root / "summary.json").read_text())
        figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for axis, material in zip(axes, ("wood", "foam"), strict=True):
            axis.plot([1, 5], [1, 5], color="#999999", linestyle=":", label="Requested count")
            for seed, marker in enumerate(("o", "s", "^")):
                subset = [
                    r
                    for r in summary["validation"]
                    if r["material"] == material and r["seed"] == seed
                ]
                axis.scatter(
                    [r["requested_count"] + (seed - 1) * 0.12 for r in subset],
                    [r["after_pour_locations"].count("tray") for r in subset],
                    color="#CC79A7",
                    marker=marker,
                    label=f"Start {seed} (5 trials)",
                )
            for target in range(1, 6):
                rows = [
                    r
                    for r in summary["validation"]
                    if r["material"] == material and r["requested_count"] == target
                ]
                axis.text(target, 5.6, f"{sum(r['pour_success'] for r in rows)}/3", ha="center")
            axis.set(
                title=f"{material.title()} | exact Pour successes above",
                xlabel="Requested count",
                ylabel="Delivered tray count",
                xticks=range(1, 6),
                yticks=range(6),
                ylim=(-0.3, 6),
                xlim=(0.6, 5.4),
            )
            axis.grid(alpha=0.2)
            axis.legend(loc="lower right", fontsize=8)
        figure.suptitle("Horizontal sideways Pour | frozen recipes | three starts")
        figure.savefig(root / "counts-results.png", dpi=160)
        plt.close(figure)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--render-worker", choices=("wood", "foam"))
        parser.add_argument("--render", action="store_true")
        parser.add_argument("--report-only", action="store_true")
        args = parser.parse_args()
        if args.worker:
            LateralCounts.worker(root=args.root, material=args.worker)
        elif args.render_worker:
            LateralCounts.render(root=args.root, material=args.render_worker)
        elif args.render:
            LateralCounts.launch(root=args.root, render=True)
        else:
            if not args.report_only:
                LateralCounts.launch(root=args.root)
            LateralCounts.report(root=args.root)


if __name__ == "__main__":
    LateralCounts.main()
