"""Scratch-kinematic stance diagnostic after real scraper pickup; no learning run."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-restarts", type=int, default=0)
    parser.add_argument("--grasp-roll", type=float, default=0)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--tilt", type=float, default=45)
    parser.add_argument("--end", action="store_true")
    parser.add_argument("--stance-grid", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("use a fresh evidence file")
    rows = []

    def record(*, row):
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")

    env = RackServingEnvironment(config=RackServingConfig(cooling_seconds=0))
    try:
        env.reset_to_seed(seed=0)
        tool = env.rack_controllers().tool
        tool.hand_in_body = (
            tool.hand_in_body @ Rotation.from_euler("z", args.grasp_roll, degrees=True).as_matrix()
        )
        env.take_action(action=np.array([0.0, 0, 0, 0.12]))
        record(row={"pickup_error": env.last_skill_error()})
        backend = env.backend()
        controller = env.rack_controllers()
        shelf, tool = controller.shelf, controller.tool
        if not backend.held(name="wiper_0"):
            return
        stroke = GatherAndSweepPlanner.sweep(
            food_corners=np.stack([backend.corners(name=name) for name in backend.cube_names]),
            d=0.15,
            phi_deg=0,
            **controller.stroke_geometry(),
        )
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ tool.wiper_pose()
        body = np.eye(4)
        body[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
        if args.push:
            body[:3, :3] = (
                Rotation.from_euler("y", args.tilt, degrees=True).as_matrix()
                @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
            )
        edge = stroke.end if args.end else stroke.start
        body[:3, 3] = edge - tool.blade_edge(pose=body).mean(axis=0)
        target = body @ np.linalg.inv(shelf.in_hand)
        record(row={"hand_target": target.tolist(), "measured_grasp": shelf.in_hand.tolist()})
        candidates = [
            *[[distance, 0, 0] for distance in (0.65, 0.75, 0.85, 0.95, 1.05)],
            *[[1.55, -distance, np.pi / 2] for distance in (0.6, 0.7, 0.8)],
            *[[distance, 0, np.pi] for distance in (1.95, 2.05, 2.15)],
        ]
        if args.push:
            candidates = [[distance, 0, np.pi] for distance in (1.95, 2.05, 2.15, 2.25)]
        if args.stance_grid:
            candidates = [
                [distance, lateral, np.pi + yaw]
                for distance in (1.95, 2.05, 2.15)
                for lateral in (-0.15, 0, 0.15)
                for yaw in (-0.3, 0, 0.3)
            ]
        rng = np.random.default_rng(42)
        for base in candidates:
            errors = []
            solutions = []
            for height in (0.06, 0):
                try:
                    arm, _ = shelf.ik(
                        position=target[:3, 3] + [0, 0, height],
                        rotation=target[:3, :3],
                        base=np.asarray(base),
                        holding=True,
                        attempts=12,
                    )
                    tool.check_configuration(base=np.asarray(base), arm=arm, holding=True)
                    achieved = shelf.site_at(base=np.asarray(base), arm=arm)
                    residual = np.linalg.norm(achieved[:3, 3] - target[:3, 3] - [0, 0, height])
                    angular = tool.rotation_error(actual=achieved[:3, :3], target=target[:3, :3])
                    if residual > 0.002 or angular > 0.5:
                        raise SkillFailed(f"IK residual {residual:.6f} m, {angular:.3f} deg")
                    errors.append(None)
                    solutions.append(arm.tolist())
                except SkillFailed as error:
                    errors.append(str(error))
                    solutions.append(None)
                if errors[-1] is not None:
                    for attempt in range(args.random_restarts):
                        try:
                            seed = rng.uniform(-np.pi, np.pi, size=7)
                            arm, _ = shelf.ik(
                                position=target[:3, 3] + [0, 0, height],
                                rotation=target[:3, :3],
                                base=np.asarray(base),
                                holding=True,
                                attempts=1,
                                seed=seed,
                            )
                            tool.check_configuration(base=np.asarray(base), arm=arm, holding=True)
                            errors[-1] = None
                            solutions[-1] = arm.tolist()
                            record(row={"base": base, "height": height, "restart": attempt})
                            break
                        except SkillFailed:
                            continue
            record(row={"base": base, "hover_lower_errors": errors, "solutions": solutions})
    finally:
        env.close()


if __name__ == "__main__":
    main()
