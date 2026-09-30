"""Diagnose native held-object collision at the start of shelf retrieval."""

import json
import pickle
from pathlib import Path
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    set_robot_joints_with_held_object,
    check_collisions_with_held_object,
)
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.utils import PyBulletSim
from shelf_reverse_controllers import ReversePlanning

root = Path("scratchpad/shelf-forward-reset")
x = pickle.loads((root / "native-seed1-v1/action-02.pkl").read_bytes())["state"]
robot = x.get_object_from_name("robot")
cube = x.get_object_from_name("cube1")
x.set(robot, "pos_base_x", 0.8)
x.set(robot, "pos_base_y", x.get(cube, "y"))
x.set(robot, "pos_base_rot", 0.0)
sim = PyBulletSim(x)
sim.set_state(x)
target = Pose(
    (x.get(cube, "x") - 0.005, x.get(cube, "y"), x.get(cube, "z") + 0.01), (0.5, 0.5, 0.5, 0.5)
)
joints = inverse_kinematics(sim.robot, target, set_joints=False)
transform = multiply_poses(target.invert(), ReversePlanning.pose(x, cube))
held = sim._cubes[cube.name]
hit = check_collisions_with_held_object(
    sim.robot, sim.get_collision_bodies(held), sim.physics_client_id, held, transform, joints
)
bodies = {**sim._cubes, **sim._static_colliders, "shelf_proxy": sim._cupboard1_shelf_id}
contacts = []
for name, body in bodies.items():
    if body == held:
        continue
    for c in p.getClosestPoints(held, body, distance=0.001, physicsClientId=sim.physics_client_id):
        contacts.append({"body": name, "distance": c[8], "held_point": c[5], "other_point": c[6]})
out = {"native_collision": hit, "held_contacts": contacts}
lift = []
for dz in (0.0, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03):
    pose = Pose(
        (target.position[0], target.position[1], target.position[2] + dz), target.orientation
    )
    q = inverse_kinematics(sim.robot, pose, set_joints=True)
    held_collision = check_collisions_with_held_object(
        sim.robot, sim.get_collision_bodies(held), sim.physics_client_id, held, transform, q
    )
    cube_contacts = []
    for name, body in bodies.items():
        if body == held:
            continue
        for c in p.getClosestPoints(
            held, body, distance=0.001, physicsClientId=sim.physics_client_id
        ):
            cube_contacts.append({"body": name, "distance": c[8]})
    robot_collision = check_collisions_with_held_object(
        sim.robot, sim.get_collision_bodies(held), sim.physics_client_id, None, None, q
    )
    lift.append({
        "dz": dz,
        "native_held_collision": held_collision,
        "native_robot_collision": robot_collision,
        "cube_contacts": cube_contacts,
    })
out["static_upward_lift"] = lift
(root / "support-contact-audit.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
