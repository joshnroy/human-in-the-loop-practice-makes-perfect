"""Search collision-checked reverse grasps; no forward ranges or physics changed."""

import argparse
import json
import pickle
import traceback
from pathlib import Path
import numpy as np
import pybullet as p
import mujoco
from scipy.spatial.transform import Rotation
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from shelf_physical_demo import Demo
from shelf_reverse_controllers import RetrieveShelfController

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-state", default="final.pkl")
parser.add_argument("--grasp-style", choices=["down", "horizontal"], default="down")
args = parser.parse_args()
saved = pickle.loads((args.source / args.source_state).read_bytes())
demo = Demo(output=args.output, seed=1, max_actions=1)
demo.env.set_state(saved["state"])
demo.state = saved["state"]
demo.initial = saved["initial"]
demo.records = saved["actions"]
demo.ticks = saved["ticks"]
model = demo.env._robot_env.sim.model.mj_model
limits = []
for index in range(1, 8):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{demo.robot.name}_joint_{index}")
    limits.append(tuple(model.jnt_range[jid]) if model.jnt_limited[jid] else None)
trials = []
try:
    cubes = [demo.state.get_object_from_name(f"cube{i}") for i in range(1, 9)]
    cubes = [cube for cube in cubes if demo.state.get(cube, "z") > 0.3]
    cubes.sort(key=lambda c: -demo.state.get(c, "z"))
    for cube in cubes:
        for offset in (0.01, 0.02, 0.03):
            for yaw in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                quat = (
                    (
                        Rotation.from_euler("z", yaw) * Rotation.from_quat([0.707, 0.707, 0, 0])
                    ).as_quat()
                    if args.grasp_style == "down"
                    else (
                        Rotation.from_euler("x", yaw) * Rotation.from_quat([0.5, 0.5, 0.5, 0.5])
                    ).as_quat()
                )
                for distance in (0.65, 0.7, 0.75):
                    base = [1.5 - distance, demo.state.get(cube, "y"), 0.0]
                    pose = Pose(
                        (
                            demo.state.get(cube, "x") - 0.005,
                            demo.state.get(cube, "y"),
                            demo.state.get(cube, "z") + offset,
                        ),
                        tuple(quat),
                    )
                    x = demo.state.copy()
                    for key, value in zip(("pos_base_x", "pos_base_y", "pos_base_rot"), base):
                        x.set(demo.robot, key, value)
                    demo.sim.set_state(x)
                    row = {"cube": cube.name, "yaw": float(yaw), "offset": offset, "base": base}
                    try:
                        joints = inverse_kinematics(demo.sim.robot, pose, set_joints=False)
                        if any(
                            lim is not None and not lim[0] <= q <= lim[1]
                            for q, lim in zip(joints[:7], limits)
                        ):
                            row["error"] = "Target exceeds actual MuJoCo joint limits"
                            trials.append(row)
                            continue
                        demo.sim.robot.set_joints(joints)
                        collisions = sum(
                            len(
                                p.getClosestPoints(
                                    demo.sim.robot.robot_id,
                                    b,
                                    0.0,
                                    physicsClientId=demo.sim.physics_client_id,
                                )
                            )
                            for b in demo.sim.get_collision_bodies()
                        )
                        row["endpoint_collisions"] = collisions
                        if collisions:
                            trials.append(row)
                            continue
                        params = np.array([*base, *pose.position, *pose.orientation])
                        controller = RetrieveShelfController(
                            objects=[demo.robot, cube], pybullet_sim=demo.sim
                        )
                        controller.physical_joint_limits = limits
                        controller.reset(demo.state, params)
                        row["planned"] = True
                        trials.append(row)
                        demo.execute("retrieve_shelf", cube, controller, params)
                        raise SystemExit(0)
                    except Exception as error:
                        row["error"] = type(error).__name__ + ": " + str(error)
                        trials.append(row)
finally:
    (args.output / "grasp-candidates.json").write_text(json.dumps(trials, indent=2))
    demo.save()
    demo.snapshot("reverse-final")
    demo.writer.close()
    demo.env.close()
