"""Refine observed count transitions without changing approved pouring physics."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from scripts.pouring3d_feasibility import PourFeasibility
from scripts.run_sweep import SweepRun, SweepRunner


class CountRefinement:
    @staticmethod
    def rows(*, root: Path, material: str) -> list[dict[str, Any]]:
        paths = [root / material / "0/trials.jsonl", root / f"refine-{material}" / "0/trials.jsonl"]
        return [
            json.loads(line)
            for path in paths
            if path.exists()
            for line in path.read_text().splitlines()
        ]

    @staticmethod
    def pour_count(*, row: dict[str, Any]) -> int | None:
        if (
            len(row["events"]) < 2
            or not row["events"][0]["success"]
            or row["events"][1]["skill"] != "Pour"
            or row["events"][1]["error"] is not None
            or not row["after_pour"]["settled"]
            or any(
                location not in {"pitcher", "tray"} for location in row["after_pour"]["locations"]
            )
        ):
            return None
        return int(row["after_pour"]["locations"].count("tray"))

    @staticmethod
    def run(*, root: Path, material: str, budget: int) -> None:
        assert (root / material / "0/search-results.json").exists()
        directory = root / f"refine-{material}" / "0"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trials.jsonl"
        prior = len(path.read_text().splitlines()) if path.exists() else 0
        for _ in range(prior, budget):
            rows = CountRefinement.rows(root=root, material=material)
            missing = set(range(1, 6)) - {CountRefinement.pour_count(row=row) for row in rows}
            if not missing:
                break
            valid = [row for row in rows if CountRefinement.pour_count(row=row) is not None]
            options = []
            axes = ("angle", "tilt_seconds", "hold_seconds")
            for axis, scale, resolution in zip(
                axes, (150, 13, 7.8), (0.04, 0.04, 0.025), strict=True
            ):
                groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
                for row in valid:
                    key = tuple(
                        value for name, value in sorted(row["parameters"].items()) if name != axis
                    )
                    groups.setdefault(key, []).append(row)
                for group in groups.values():
                    ordered = sorted(group, key=lambda row: row["parameters"][axis])
                    for left, right in zip(ordered, ordered[1:], strict=False):
                        a, b = (CountRefinement.pour_count(row=row) for row in (left, right))
                        assert a is not None and b is not None
                        gap = right["parameters"][axis] - left["parameters"][axis]
                        if gap <= resolution or not any(min(a, b) < k < max(a, b) for k in missing):
                            continue
                        parameters = dict(left["parameters"])
                        parameters[axis] = (
                            left["parameters"][axis] + right["parameters"][axis]
                        ) / 2
                        if any(row["parameters"] == parameters for row in rows):
                            continue
                        options.append((gap / scale / abs(b - a), parameters))
            if not options:
                break
            _, parameters = max(options, key=lambda option: option[0])
            row = PourFeasibility.trial(
                material=material,
                seed=0,
                parameters=PourParameters(**parameters),
            )
            row["trial"] = len(rows)
            row["refinement"] = True
            with path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        rows = CountRefinement.rows(root=root, material=material)
        (directory / "refinement-results.json").write_text(
            json.dumps(
                {
                    "material": material,
                    "budget": budget,
                    "total_trials": len(rows),
                    "found_pour_counts": sorted({
                        CountRefinement.pour_count(row=row)
                        for row in rows
                        if CountRefinement.pour_count(row=row) in range(1, 6)
                    }),
                    "method": (
                        "Midpoints between settled count outcomes, fixing other controls; "
                        "no monotonicity or infeasibility claim."
                    ),
                },
                indent=2,
            )
        )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--material", choices=("wood", "foam"), required=True)
        parser.add_argument("--budget", type=int, default=16)
        parser.add_argument("--worker", action="store_true")
        args = parser.parse_args()
        if not 1 <= args.budget <= 16:
            parser.error("refinement budget must be between 1 and 16")
        if args.worker:
            CountRefinement.run(root=args.root, material=args.material, budget=args.budget)
        else:
            repo = Path(__file__).resolve().parents[1]
            outcomes = SweepRunner.execute(
                runs=[
                    SweepRun(
                        method=f"refine-{args.material}",
                        seed=0,
                        output_dir=args.root / f"refine-{args.material}" / "0",
                        command=[
                            "run-heavy",
                            str(repo / "scripts/with_env.sh"),
                            sys.executable,
                            "-m",
                            "scripts.pouring3d_refine",
                            "--root",
                            str(args.root),
                            "--material",
                            args.material,
                            "--budget",
                            str(args.budget),
                            "--worker",
                        ],
                    )
                ],
                max_workers=1,
            )
            if not outcomes[0].succeeded:
                raise SystemExit(1)


if __name__ == "__main__":
    CountRefinement.main()
