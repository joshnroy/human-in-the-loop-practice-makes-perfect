"""Observe the stage of a native Sweep refusal without changing planner results."""

import json

import numpy as np
import pybullet as p
from kinder_models.dynamic3d.sweep3D import parameterized_skills as native

from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession
from hitl_pmp.environments.sweep_drawer3d.stock_skills import StockSweepSkills

original_base = native.run_base_motion_planning
original_arm = native.run_motion_planning


def base(*args, **kwargs):
    result = original_base(*args, **kwargs)
    print(json.dumps({"stage": "base", "plan": result is not None}), flush=True)
    return result


def arm(*args, **kwargs):
    result = original_arm(*args, **kwargs)
    print(json.dumps({"stage": "arm", "plan": result is not None}), flush=True)
    if result is None:
        robot, start, end = args[:3]
        saved = robot.get_joint_positions()
        try:
            for label, joints in (("start", start), ("target", end)):
                robot.set_joints(joints)
                p.performCollisionDetection(physicsClientId=kwargs["physics_client_id"])
                for body in kwargs["collision_bodies"]:
                    contacts = p.getClosestPoints(
                        robot.robot_id, body, 0.001, physicsClientId=kwargs["physics_client_id"]
                    )
                    if contacts:
                        print(
                            json.dumps({
                                "stage": label,
                                "body": body,
                                "pose": p.getBasePositionAndOrientation(
                                    body, physicsClientId=kwargs["physics_client_id"]
                                ),
                                "contacts": [{"link": c[3], "distance": c[8]} for c in contacts],
                            }),
                            flush=True,
                        )
        finally:
            robot.set_joints(saved)
    return result


session = SweepDrawerSession(seed=10001)
try:
    for skill, distance in (("open_drawer", 0.8), ("pick_wiper", 0.7)):
        result = StockSweepSkills.run(
            session=session,
            skill=skill,
            label=skill,
            phase="diagnostic",
            params=np.array([distance, -np.pi]),
        )
        print(skill, result, flush=True)
    native.run_base_motion_planning = base
    native.run_motion_planning = arm
    result = StockSweepSkills.run(
        session=session,
        skill="sweep",
        label="sweep",
        phase="diagnostic",
        params=np.array([0.55, -np.pi]),
    )
    print("Sweep", result, flush=True)
finally:
    native.run_base_motion_planning = original_base
    native.run_motion_planning = original_arm
    session.close()
