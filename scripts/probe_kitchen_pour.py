"""Measure one fixed-rate kitchen pour; run identical cases in two frozen checkouts."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES


class PourProbe:
    @staticmethod
    def run(*, seed: int, requested: int, rate: float, setup: str) -> dict:
        env = KitchenEnvironment(requested_count=requested)
        started = time.monotonic()
        record: dict = {"seed": seed, "requested": requested, "rate": rate, "setup": setup}
        actions = []
        stage = "initialization"

        def action(*, skill: str, count: int = requested, index: int = 0, ramp: float = rate):
            skill_id = SKILL_NAMES.index(skill)
            env.take_action(action=np.array([skill_id, index, count, ramp, 0, 0], dtype=float))
            info = dict(env.last_info)
            actions.append(info)
            return info

        def require(*, info: dict):
            if not info["success"]:
                raise ValueError(f"Setup action failed: {info}")

        try:
            env.reset_to_seed(seed=seed, requested=requested)
            c = env.controller()
            stage = "robot_setup" if setup == "robot" else "initialization"
            if setup == "robot":
                require(info=action(skill="PickPitcher"))
                # A spill is allowed here; the robot must recover from its measured location.
                action(skill="Pour", count=3, ramp=1.0)
                require(info=action(skill="PlacePitcherAtHome"))
                for index in range(len(c.scene.cube_names)):
                    location = c.scene.cube_locations()[index]
                    if location == "pitcher":
                        continue
                    skill = {
                        "tray": "PickShrimpFromTray",
                        "floor": "PickShrimpFromFloor",
                        "surface": "PickShrimpFromSurface",
                    }[location]
                    require(info=action(skill=skill, index=index))
                    require(info=action(skill="DropShrimpIntoPitcher", index=index))
                if not c.ready():
                    raise ValueError("Robot reset did not leave a ready scene")
            record["scene_before_pick"] = c.scene.physical_snapshot()
            record["setup_actions"] = list(actions)
            actions.clear()
            stage = "pick_pitcher"
            require(info=action(skill="PickPitcher"))
            record["setup_success"] = True
            stage = "pour"
            hand_before = c.hand_pose()[1]
            jug_before = c.body_pose(name="pitcher")[1]
            grasp = hand_before.inv() * jug_before
            relative_before = hand_before.inv().apply(
                c.body_pose(name="pitcher")[0] - c.hand_pose()[0]
            )
            record["jug_before_pour"] = c.body_pose(name="pitcher")[0].tolist()
            pour = action(skill="Pour")
            jug = c.body_pose(name="pitcher")[1]
            hand = c.hand_pose()[1]
            record["grasp_rotation_change_deg"] = float(
                np.rad2deg((grasp.inv() * hand.inv() * jug).magnitude())
            )
            relative_after = hand.inv().apply(c.body_pose(name="pitcher")[0] - c.hand_pose()[0])
            record["grasp_translation_change_m"] = float(
                np.linalg.norm(relative_after - relative_before)
            )
            record["pour"] = pour
            record["pour_success"] = bool(pour["success"])
            record["locations_after_pour"] = c.scene.cube_locations()
            record["settled_after_pour"] = bool(c.outcome_settled())
            stage = "placement"
            place = action(skill="PlacePitcherAtHome")
            record["place"] = place
            record["home_after_place"] = bool(c.vessel_home(name="pitcher"))
            record["success"] = bool(pour["success"] and place["success"])
            stage = "complete"
        except Exception as exc:
            record.setdefault("setup_success", False)
            record["success"] = False
            record["exception"] = str(exc)
            record["traceback"] = traceback.format_exc()
        finally:
            record["stage"] = stage
            record["actions"] = actions
            record["elapsed_seconds"] = time.monotonic() - started
            env.close()
        return record

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--requested", type=int, choices=(1, 2, 3), required=True)
        parser.add_argument("--rate", type=float, required=True)
        parser.add_argument("--setup", choices=("fresh", "robot"), default="fresh")
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        import kinder
        import kinder_models

        import hitl_pmp

        root = Path(hitl_pmp.__file__).resolve().parents[2]
        identity = {
            name: subprocess.check_output(
                ["git", "-C", str(root / rel), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, rel in [
                ("main", "."),
                ("kindergarden", "reference/kindergarden"),
                ("kinder-baselines", "reference/kinder-baselines"),
            ]
        }
        for module in [hitl_pmp, kinder, kinder_models]:
            assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
        result = PourProbe.run(
            seed=args.seed, requested=args.requested, rate=args.rate, setup=args.setup
        )
        result["identity"] = identity
        result["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["module_paths"] = {
            module.__name__: str(Path(module.__file__).resolve())
            for module in [hitl_pmp, kinder, kinder_models]
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, default=lambda value: value.tolist())
            f.write("\n")
        print(json.dumps({k: result[k] for k in ["seed", "requested", "rate", "setup", "success"]}))


if __name__ == "__main__":
    PourProbe.main()
