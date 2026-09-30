"""Read-only grasp-aperture feasibility on an actual recorded shelf cube."""

import json
import pickle
from pathlib import Path
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.utils import PyBulletSim

root = Path("scratchpad/shelf-forward-reset")
state = pickle.loads((root / "native-seed1-v1/action-02.pkl").read_bytes())["state"]
sim = PyBulletSim(state)
robot = state.get_object_from_name("robot")
cube = state.get_object_from_name("cube1")
rows = []
for offset in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    for yaw in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        quat = (Rotation.from_euler("z", yaw) * Rotation.from_quat([0.707, 0.707, 0, 0])).as_quat()
        for grip in (0.0, 0.15, 0.3, 0.45):
            x = state.copy()
            x.set(robot, "pos_base_x", 0.8)
            x.set(robot, "pos_base_y", state.get(cube, "y"))
            x.set(robot, "pos_base_rot", 0.0)
            sim.set_state(x)
            pose = Pose(
                (state.get(cube, "x") - 0.005, state.get(cube, "y"), state.get(cube, "z") + offset),
                tuple(quat),
            )
            row = {"offset": offset, "yaw": float(yaw), "grip": grip}
            try:
                joints = inverse_kinematics(sim.robot, pose, set_joints=False)
                joints[7:] = sim.robot.finger_state_to_joints(grip)
                sim.robot.set_joints(joints)
                bodies = {
                    **sim._cubes,
                    **sim._static_colliders,
                    "shelf_proxy": sim._cupboard1_shelf_id,
                }
                row["collisions"] = []
                for name, body in bodies.items():
                    for c in p.getClosestPoints(
                        sim.robot.robot_id, body, 0.0, physicsClientId=sim.physics_client_id
                    ):
                        row["collisions"].append({"body": name, "link": c[3], "distance": c[8]})
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)
(root / "finger-aperture-audit.json").write_text(json.dumps(rows, indent=2))
print(json.dumps([row for row in rows if row.get("collisions") == []], indent=2))
