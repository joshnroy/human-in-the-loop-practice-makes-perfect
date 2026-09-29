"""Read-only geometric diagnosis from an observed development probe state."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPlanningScene
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


class ApproachDiagnosis:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("tag")
        args = parser.parse_args()
        root = Path("scratchpad/sweepsimple3d") / args.tag
        records = [json.loads(x) for x in (root / "replay.jsonl").read_text().splitlines()]
        frame = next(x for x in reversed(records) if "qpos" in x)
        records = [json.loads(x) for x in (root / "state.jsonl").read_text().splitlines()]
        state = next(x["state"] for x in reversed(records) if "state" in x)
        session = SweepSimpleSession(seed=0)
        session.mj_data.qpos[:] = frame["qpos"]
        mujoco.mj_forward(session.mj_model, session.mj_data)
        for name, values in state.items():
            obj = session.state.get_object_from_name(name)
            for key, value in zip(session.state.type_features[obj.type], values, strict=True):
                session.state.set(obj, key, value)
        scene = FloorPlanningScene(session=session)
        held = multiply_poses(
            scene.ee_now().invert(),
            Pose(tuple(session.position(name="wiper_0")), session.quaternion(name="wiper_0")),
        )
        cube = session.position(name="cube_0")
        behind = max(
            [0.0]
            + [
                -float(session.position(name=f"cube_{i}")[1] - cube[1])
                for i in range(5)
                if abs(session.position(name=f"cube_{i}")[0] - cube[0]) <= 0.16
            ]
        )
        point = cube[:2] - np.array([0.0, behind + 0.06])
        bodies = scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
        import pybullet

        scene.in_collision(
            joints=scene.planning_fingers(arm=session.arm(), state=0.5),
            bodies=bodies,
            held=scene.wiper_body,
            held_tf=held,
        )
        print(
            "self_contacts",
            [
                (a, b)
                for a, b in scene.robot.self_collision_link_ids
                if pybullet.getClosestPoints(
                    scene.robot.robot_id,
                    scene.robot.robot_id,
                    distance=0.0,
                    linkIndexA=a,
                    linkIndexB=b,
                    physicsClientId=scene.cid,
                )
            ],
            flush=True,
        )
        labels = {v: k for k, v in scene._sim._static_colliders.items()}
        labels[scene.chassis_body] = "chassis"
        print(
            "current_contacts",
            json.dumps({
                labels.get(b, str(b)): [
                    (x[3], x[4], x[8])
                    for x in pybullet.getClosestPoints(
                        scene.robot.robot_id, b, distance=0.0, physicsClientId=scene.cid
                    )
                ]
                for b in bodies
            }),
            flush=True,
        )
        print(
            "current_tool_contacts",
            json.dumps({
                labels.get(b, str(b)): [
                    (x[3], x[4], x[8])
                    for x in pybullet.getClosestPoints(
                        scene.wiper_body, b, distance=0.005, physicsClientId=scene.cid
                    )
                ]
                for b in [*bodies, scene.robot.robot_id]
            }),
            flush=True,
        )
        output = []
        for yaw in (0.0, np.pi):
            floor = multiply_poses(Pose.from_rpy((*point, 0.001), (0.0, 0.0, yaw)), held.invert())
            for height in (0.0, 0.14, 0.24):
                hover = Pose(tuple(np.asarray(floor.position) + [0, 0, height]), floor.orientation)
                scene.sync()
                solutions = ikfast_closest_inverse_kinematics(scene.robot, world_from_target=hover)
                row = {
                    "yaw": yaw,
                    "hover_height": height,
                    "solutions": len(solutions),
                    "in_limits": 0,
                    "robot_clear": 0,
                    "tool_clear": 0,
                    "descent_clear": 0,
                }
                for solution in solutions:
                    q = np.asarray(solution[:7])
                    if not scene.within_arm_limits(arm=q):
                        continue
                    row["in_limits"] += 1
                    if scene.in_collision(
                        joints=scene.planning_fingers(arm=q, state=0.5), bodies=bodies
                    ):
                        continue
                    row["robot_clear"] += 1
                    if scene.in_collision(
                        joints=scene.planning_fingers(arm=q, state=0.5),
                        bodies=bodies,
                        held=scene.wiper_body,
                        held_tf=held,
                    ):
                        continue
                    row["tool_clear"] += 1
                    path = scene.linear_path(
                        start=q, target=floor, bodies=bodies, finger_state=0.5, max_jump=0.6
                    )
                    if path is not None and not any(
                        scene.in_collision(
                            joints=scene.planning_fingers(arm=x, state=0.5),
                            bodies=bodies,
                            held=scene.wiper_body,
                            held_tf=held,
                        )
                        for x in path
                    ):
                        row["descent_clear"] += 1
                output.append(row)
                print(json.dumps(row), flush=True)
        (
            root
            / (
                "approach-diagnosis-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + ".json"
            )
        ).write_text(json.dumps(output, indent=2))
        scene._sim.close()
        session.close()


if __name__ == "__main__":
    ApproachDiagnosis.run()
