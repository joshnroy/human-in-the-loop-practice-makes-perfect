"""Bounded physical feasibility search with the approved Winco material profiles.

This is a controller diagnostic, not a learning-method benchmark. Every trial
starts from a fresh seeded environment and physically picks, pours, and places.
SweepRunner supplies process isolation, logs, and timing; run-heavy limits jobs.
"""

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from hitl_pmp.environments.pouring3d.simulation import PouringSimulation
from scripts.run_sweep import SweepRun, SweepRunner


class PourFeasibility:
    @staticmethod
    def trial(
        *, material: str, seed: int, parameters: PourParameters, target: int = 5
    ) -> dict[str, Any]:
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        start = time.monotonic()
        native_path = Path(inspect.getfile(PitcherPour3DEnv))
        task_path = native_path.parent / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
        sim = PouringSimulation.create(
            seed=seed,
            scene_bg=True,
            requested_count=target,
            cube_material=material,
            task_config_path=str(task_path),
        )
        try:
            initial = sim.snapshot()
            assert initial["locations"] == ["pitcher"] * 5
            events = [sim.execute(skill="PickPitcher")]
            before_pour = sim.snapshot()
            if events[-1]["success"]:
                assert before_pour["locations"] == ["pitcher"] * 5
                events.append(sim.execute(skill="Pour", parameters=parameters.model_dump()))
            after_pour = sim.snapshot()
            if events[-1]["skill"] == "Pour" and events[-1]["error"] is None:
                events.append(sim.execute(skill="PlacePitcherAtHome"))
            after_place = sim.snapshot()
            completed = (
                len(events) == 3
                and events[0]["success"]
                and events[1]["error"] is None
                and events[2]["success"]
            )
            count = after_pour["locations"].count("tray")
            feasible_count = (
                count
                if completed
                and after_pour["settled"]
                and after_place["settled"]
                and after_pour["locations"].count("pitcher") == 5 - count
                and after_place["locations"] == after_pour["locations"]
                else None
            )
            return {
                "material": material,
                "seed": seed,
                "target": target,
                "parameters": parameters.model_dump(),
                "feasible_count": feasible_count,
                "completed": completed,
                "initial": initial,
                "before_pour": before_pour,
                "after_pour": after_pour,
                "after_place": after_place,
                "events": events,
                "elapsed_seconds": time.monotonic() - start,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        native_path,
                        task_path,
                        Path(inspect.getfile(type(sim.controller))),
                    )
                },
            }
        finally:
            sim.env.close()
            del sim
            gc.collect()

    @staticmethod
    def search(*, directory: Path, material: str, budget: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        history_path = directory / "trials.jsonl"
        history = (
            [json.loads(line) for line in history_path.read_text().splitlines()]
            if history_path.exists()
            else []
        )

        def evaluate(*, parameters: PourParameters) -> None:
            if len(history) >= budget or any(
                row["parameters"] == parameters.model_dump() for row in history
            ):
                return
            row = PourFeasibility.trial(material=material, seed=0, parameters=parameters)
            row["trial"] = len(history)
            history.append(row)
            with history_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                json.dumps({
                    key: row[key]
                    for key in (
                        "trial",
                        "material",
                        "parameters",
                        "feasible_count",
                        "elapsed_seconds",
                    )
                }),
                flush=True,
            )

        def found() -> set[int]:
            return {
                row["feasible_count"] for row in history if row["feasible_count"] in range(1, 6)
            }

        # Anchor the known all-five setting, then bracket the release transition.
        for angle in [130, *range(90 if material == "wood" else 100, 130, 5)]:
            evaluate(parameters=PourParameters(angle=angle, tilt_seconds=10, hold_seconds=0.2))
            if len(found()) == 5:
                break
        # Refine gaps that skip a requested count at the slow, short-hold setting.
        for _ in range(18):
            if len(found()) == 5 or len(history) >= budget:
                break
            rows = sorted(
                (
                    {**row, "bracket_count": row["after_pour"]["locations"].count("tray")}
                    for row in history
                    if len(row["events"]) >= 2
                    and row["events"][1]["skill"] == "Pour"
                    and row["events"][1]["error"] is None
                    and row["after_pour"]["settled"]
                    and all(
                        location in {"pitcher", "tray"}
                        for location in row["after_pour"]["locations"]
                    )
                    and row["parameters"]["tilt_seconds"] == 10
                    and row["parameters"]["hold_seconds"] == 0.2
                ),
                key=lambda row: row["parameters"]["angle"],
            )
            intervals = [
                (left, right)
                for left, right in zip(rows, rows[1:], strict=False)
                if right["parameters"]["angle"] - left["parameters"]["angle"] > 0.08
                and any(
                    min(left["bracket_count"], right["bracket_count"])
                    < count
                    < max(left["bracket_count"], right["bracket_count"])
                    for count in set(range(1, 6)) - found()
                )
            ]
            if not intervals:
                break
            left, right = max(
                intervals,
                key=lambda pair: pair[1]["parameters"]["angle"] - pair[0]["parameters"]["angle"],
            )
            evaluate(
                parameters=PourParameters(
                    angle=(left["parameters"]["angle"] + right["parameters"]["angle"]) / 2,
                    tilt_seconds=10,
                    hold_seconds=0.2,
                )
            )
        # If counts remain missing, explore the existing timing parameters.
        # Deterministic coarse-to-fine coverage; never change physics or geometry.
        for tilt_seconds in (5, 2, 15, 7, 3, 12):
            for angle in (
                (100, 105, 110, 115, 120) if material == "wood" else (110, 115, 120, 125, 130)
            ):
                for hold_seconds in (0.2, 1.0):
                    if len(found()) == 5 or len(history) >= budget:
                        break
                    evaluate(
                        parameters=PourParameters(
                            angle=angle,
                            tilt_seconds=tilt_seconds,
                            hold_seconds=hold_seconds,
                        )
                    )
        candidates = {
            str(count): next(row for row in history if row["feasible_count"] == count)["parameters"]
            for count in sorted(found())
        }
        (directory / "search-results.json").write_text(
            json.dumps(
                {
                    "material": material,
                    "seed": 0,
                    "budget": budget,
                    "trials": len(history),
                    "found_counts": sorted(found()),
                    "not_found_counts": sorted(set(range(1, 6)) - found()),
                    "candidates": candidates,
                    "success_definition": (
                        "Exact tray count, remainder retained, no spills, "
                        "all settled after Pour and unchanged after physical Place."
                    ),
                    "interpretation": (
                        "Bounded diagnostic search, not a learned policy or proof of infeasibility."
                    ),
                },
                indent=2,
            )
        )

    @staticmethod
    def launch(
        *, directory: Path, budget: int, materials: tuple[str, ...] = ("wood", "foam")
    ) -> None:
        repo = Path(__file__).resolve().parents[1]
        runs = [
            SweepRun(
                method=material,
                seed=0,
                output_dir=directory / material / "0",
                command=[
                    "run-heavy",
                    str(repo / "scripts/with_env.sh"),
                    sys.executable,
                    "-m",
                    "scripts.pouring3d_feasibility",
                    "--worker",
                    material,
                    "--output-dir",
                    str(directory / material / "0"),
                    "--budget",
                    str(budget),
                ],
            )
            for material in materials
        ]
        outcomes = SweepRunner.execute(runs=runs, max_workers=2)
        if not all(outcome.succeeded for outcome in outcomes):
            raise SystemExit(1)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--budget", type=int, default=60)
        parser.add_argument("--worker", choices=("wood", "foam"))
        parser.add_argument("--material", choices=("wood", "foam"))
        args = parser.parse_args()
        if not 1 <= args.budget <= 100:
            parser.error("budget must be between 1 and 100 trials per material")
        if args.worker:
            PourFeasibility.search(
                directory=args.output_dir, material=args.worker, budget=args.budget
            )
        else:
            PourFeasibility.launch(
                directory=args.output_dir,
                budget=args.budget,
                materials=(args.material,) if args.material else ("wood", "foam"),
            )


if __name__ == "__main__":
    PourFeasibility.main()
