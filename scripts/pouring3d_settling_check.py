"""Check whether existing stationary Wait resolves the observed four-cube splits.

This supplements, rather than replaces, the unchanged Pour completion check.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

from hitl_pmp.environments.pouring3d.native_controller import PourParameters
from hitl_pmp.environments.pouring3d.simulation import PouringSimulation
from scripts.run_sweep import SweepRun, SweepRunner


class SettlingCheck:
    @staticmethod
    def run(*, root: Path, material: str) -> None:
        from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

        task = (
            Path(inspect.getfile(PitcherPour3DEnv)).parent
            / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
        )
        directory = root / f"{material}-four-settling" / "0"
        directory.mkdir(parents=True, exist_ok=True)
        parameters = PourParameters(
            angle=105 if material == "wood" else 115, tilt_seconds=5, hold_seconds=1
        )
        for seed in (0, 1, 2):
            sim = PouringSimulation.create(
                seed=seed,
                requested_count=4,
                cube_material=material,
                scene_bg=True,
                task_config_path=str(task),
            )
            try:
                initial = sim.snapshot()
                assert initial["locations"] == ["pitcher"] * 5
                events = [sim.execute(skill="PickPitcher")]
                assert events[-1]["success"] and sim.locations() == ["pitcher"] * 5
                events.append(sim.execute(skill="Pour", parameters=parameters.model_dump()))
                after_pour = sim.snapshot()
                if events[-1]["error"] is None:
                    for _ in range(10):
                        if sim.settled():
                            break
                        events.append(sim.execute(skill="Wait"))
                        if not events[-1]["success"]:
                            break
                after_wait = sim.snapshot()
                if events[-1]["error"] is None:
                    events.append(sim.execute(skill="PlacePitcherAtHome"))
                after_place = sim.snapshot()
                row = {
                    "material": material,
                    "target": 4,
                    "seed": seed,
                    "parameters": parameters.model_dump(),
                    "initial": initial,
                    "after_pour": after_pour,
                    "after_wait": after_wait,
                    "after_place": after_place,
                    "events": events,
                    "wait_seconds": sum(
                        event["seconds"] for event in events if event["skill"] == "Wait"
                    ),
                    "pour_success": events[1]["success"],
                    "pour_plus_wait_success": (
                        events[1]["error"] is None
                        and after_wait["goal"]
                        and all(event["success"] for event in events if event["skill"] == "Wait")
                    ),
                    "full_completion_success": (
                        after_wait["goal"]
                        and after_place["goal"]
                        and events[-1]["skill"] == "PlacePitcherAtHome"
                        and events[-1]["success"]
                    ),
                }
                with (directory / "settling-results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            finally:
                sim.close()

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--material", choices=("wood", "foam"), default="wood")
        args = parser.parse_args()
        if args.worker:
            SettlingCheck.run(root=args.root, material=args.material)
        else:
            repo = Path(__file__).resolve().parents[1]
            outcomes = SweepRunner.execute(
                runs=[
                    SweepRun(
                        method=f"{args.material}-four-settling",
                        seed=0,
                        output_dir=args.root / f"{args.material}-four-settling" / "0",
                        command=[
                            "run-heavy",
                            str(repo / "scripts/with_env.sh"),
                            sys.executable,
                            "-m",
                            "scripts.pouring3d_settling_check",
                            "--root",
                            str(args.root),
                            "--material",
                            args.material,
                            "--worker",
                        ],
                    )
                ],
                max_workers=1,
            )
            if not outcomes[0].succeeded:
                raise SystemExit(1)


if __name__ == "__main__":
    SettlingCheck.main()
