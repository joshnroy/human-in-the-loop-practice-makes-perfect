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


class FloorSupportDiagnosis:
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
        cubes = np.array([session.position(name=f"cube_{i}") for i in range(5)])
        point = np.array([(cubes[:, 0].min() + cubes[:, 0].max()) / 2, cubes[:, 1].min() - 0.025])
        output = []
        for distance in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            for heading in (-np.pi / 12, 0.0, np.pi / 12):
                bearing = -np.pi / 2 + heading
                base = (
                    float(point[0] + distance * np.cos(bearing)),
                    float(point[1] + distance * np.sin(bearing)),
                    float(np.pi / 2 + heading),
                )
                for yaw in (0.0, np.pi):
                    scene.sync(base=base)
                    bodies = scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
                    target = multiply_poses(
                        Pose.from_rpy((*point, 0.001), (0.0, 0.0, yaw)), held.invert()
                    )
                    sols = ikfast_closest_inverse_kinematics(scene.robot, world_from_target=target)
                    limits = [q for q in sols if scene.within_arm_limits(arm=q[:7])]
                    clear = [
                        q
                        for q in limits
                        if not scene.in_collision(
                            joints=scene.planning_fingers(arm=q, state=0.5),
                            bodies=bodies,
                            held=scene.wiper_body,
                            held_tf=held,
                        )
                    ]
                    row = dict(
                        distance=distance,
                        heading=float(heading),
                        yaw=yaw,
                        solutions=len(sols),
                        native_limits=len(limits),
                        clear=len(clear),
                        ee=target.position,
                    )
                    output.append(row)
                    print(json.dumps(row), flush=True)
        path = root / (
            "floor-support-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        path.write_text(json.dumps(output, indent=2))
        scene._sim.close()
        session.close()


if __name__ == "__main__":
    FloorSupportDiagnosis.run()
