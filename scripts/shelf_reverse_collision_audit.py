"""Exact native PyBullet target-collision diagnosis, no physical state changes."""

import json
import pickle
from pathlib import Path
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.utils import PyBulletSim, GRASP_TRANSFORM_TO_OBJECT
from shelf_reverse_controllers import ReversePlanning

root = Path("scratchpad/shelf-forward-reset")
state = pickle.loads((root / "native-seed1-v1/final.pkl").read_bytes())["state"]
sim = PyBulletSim(state)
robot = state.get_object_from_name("robot")
rows = []
for i in range(8, 0, -1):
    cube = state.get_object_from_name(f"cube{i}")
    x = state.copy()
    x.set(robot, "pos_base_x", 0.475)
    x.set(robot, "pos_base_y", state.get(cube, "y"))
    x.set(robot, "pos_base_rot", 0.0)
    for label, pose in [
        (
            "native_down",
            multiply_poses(ReversePlanning.pose(state, cube), GRASP_TRANSFORM_TO_OBJECT),
        ),
        (
            "horizontal",
            Pose(
                (state.get(cube, "x") - 0.005, state.get(cube, "y"), state.get(cube, "z") + 0.005),
                (0.5, 0.5, 0.5, 0.5),
            ),
        ),
    ]:
        sim.set_state(x)
        row = {"cube": cube.name, "grasp": label}
        try:
            joints = inverse_kinematics(sim.robot, pose, set_joints=False)
            sim.robot.set_joints(joints)
            bodies = {
                **sim._cubes,
                **sim._static_colliders,
                "native_shelf_proxy": sim._cupboard1_shelf_id,
            }
            row["collisions"] = []
            for name, body in bodies.items():
                contacts = p.getClosestPoints(
                    sim.robot.robot_id, body, distance=0.0, physicsClientId=sim.physics_client_id
                )
                for contact in contacts:
                    row["collisions"].append({
                        "body": name,
                        "robot_link": contact[3],
                        "other_link": contact[4],
                        "distance": contact[8],
                    })
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
(root / "reverse-collision-audit.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
