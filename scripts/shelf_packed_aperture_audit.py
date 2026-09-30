"""Static aperture/orientation candidates for the actual packed shelf state."""

import json, pickle
from pathlib import Path
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.utils import PyBulletSim

root = Path("scratchpad/shelf-forward-reset")
state = pickle.loads((root / "native-seed1-v1/final.pkl").read_bytes())["state"]
sim = PyBulletSim(state)
robot = state.get_object_from_name("robot")
rows = []
for i in range(1, 9):
    cube = state.get_object_from_name(f"cube{i}")
    for offset in (0.005, 0.015, 0.025):
        for roll in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            quat = (
                Rotation.from_euler("x", roll) * Rotation.from_quat([0.5, 0.5, 0.5, 0.5])
            ).as_quat()
            for grip in (0.15, 0.3, 0.4):
                x = state.copy()
                x.set(robot, "pos_base_x", 0.75)
                x.set(robot, "pos_base_y", state.get(cube, "y"))
                x.set(robot, "pos_base_rot", 0.0)
                sim.set_state(x)
                pose = Pose(
                    (
                        state.get(cube, "x") - 0.005,
                        state.get(cube, "y"),
                        state.get(cube, "z") + offset,
                    ),
                    tuple(quat),
                )
                row = {
                    "cube": cube.name,
                    "offset": offset,
                    "roll": float(roll),
                    "grip": grip,
                    "pose": [*pose.position, *pose.orientation],
                }
                try:
                    q = inverse_kinematics(sim.robot, pose, set_joints=False)
                    q[7:] = sim.robot.finger_state_to_joints(grip)
                    sim.robot.set_joints(q)
                    row["joints"] = q
                    row["collisions"] = sum(
                        len(
                            p.getClosestPoints(
                                sim.robot.robot_id, b, 0.0, physicsClientId=sim.physics_client_id
                            )
                        )
                        for b in sim.get_collision_bodies()
                    )
                except Exception as error:
                    row["error"] = str(error)
                rows.append(row)
(root / "packed-aperture-audit.json").write_text(json.dumps(rows, indent=2))
print(json.dumps([r for r in rows if r.get("collisions") == 0], indent=2))
