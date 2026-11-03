"""Planned kitchen motions, laid out like `kinder_models.dynamic3d.*.parameterized_skills`.

Each controller's `reset` does all of its planning, in a PyBullet scene built fresh
from the observed state, before the robot moves (upstream's pattern). Its `plans` are
the executed form of each motion: BiRRT's path remapped to constant spacing and
densely re-checked (`plan_checked_arm_motion`, Tossing3D's pick), so the straight
joint-space segments between waypoints are collision-free and not only the waypoints.
The kitchen's controller then drives MuJoCo through every waypoint
(`KitchenController.follow_arm_path` / `follow_base_path`).

The one deliberate API difference from upstream: execution is not an 11-D action from
`step()` but the waypoints themselves, because the kitchen's motor layer adds a payload
feed-forward and per-tick observers that upstream's kp=2 action loop does not have.
"""

from typing import Any

import numpy as np
from kinder_models.dynamic3d.tossing.parameterized_skills import plan_checked_arm_motion
from kinder_models.dynamic3d.utils import (
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    inverse_kinematics_fixed_effort,
    run_base_motion_planning,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from scipy.spatial.transform import Rotation
from spatialmath import SE2

from .utils import PLANNING_MARGIN_M, KitchenPyBulletSim


class PlanningRefused(ValueError):
    """A planner found no plan; nothing was executed."""


class KitchenMotionPlanner:
    """A static-method container, never instantiated."""

    @staticmethod
    def with_fingers(*, sim: KitchenPyBulletSim, arm: Any) -> list[float]:
        joints = list(sim.robot.get_joint_positions())
        joints[:7] = [float(v) for v in arm[:7]]
        return joints

    @staticmethod
    def plan_arm(
        *,
        sim: KitchenPyBulletSim,
        start: Any,
        position: np.ndarray,
        rotation: Rotation,
        what: str,
        margin: float = PLANNING_MARGIN_M,
    ) -> list[np.ndarray]:
        """A checked joint path from `start` to an IK solution for the hand pose."""
        start_joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=start)
        sim.robot.set_joints(start_joints)
        px, py, pz = (float(v) for v in position)
        qx, qy, qz, qw = (float(v) for v in rotation.as_quat())
        target = Pose((px, py, pz), (qx, qy, qz, qw))
        try:
            goal = inverse_kinematics_fixed_effort(sim.robot, target, set_joints=False)
        except InverseKinematicsError as exc:
            raise PlanningRefused(f"No IK solution for {what}") from exc
        planned = plan_checked_arm_motion(
            sim.robot,
            start_joints,
            KitchenMotionPlanner.with_fingers(sim=sim, arm=goal),
            collision_bodies=sim.get_collision_bodies(),
            physics_client_id=sim.physics_client_id,
            distance_threshold=margin,
        )
        if planned is None:
            raise PlanningRefused(f"No collision-free arm path to {what}")
        return [np.asarray(q[:7], dtype=float) for q in planned[1]]

    @staticmethod
    def plan_base(
        *, sim: KitchenPyBulletSim, state: Any, target: np.ndarray, arm: Any, what: str
    ) -> list[np.ndarray]:
        """A base path to `target` (x, y, yaw), the arm carried as it is: the 2-D
        footprint against obstacle footprints, the 3-D arm against the scene."""
        joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=arm)
        bodies = sim.get_collision_bodies()

        def body_collides(pose: SE2) -> bool:  # noqa: PLR0917 -- upstream callback
            return sim.robot_collides_at_base(
                pose, joints, collision_bodies=bodies, distance_threshold=PLANNING_MARGIN_M
            )

        path = run_base_motion_planning(
            state=state,
            target_base_pose=SE2(float(target[0]), float(target[1]), float(target[2])),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            body_collision_fn=body_collides,
        )
        if path is None:
            raise PlanningRefused(f"No collision-free base path to {what}")
        return [np.array([pose.x, pose.y, pose.theta()]) for pose in path]


class PickPitcherController:
    """The free-space part of PickPitcher: the base to the pick pose, then the arm to a
    hover above the handle. The descent, grasp and lift that follow are contact
    motions and stay scripted (`grasp_and_lift_pitcher`)."""

    BASE_MOTION = "planned base to pitcher"
    MOVE_ARM_TO_HOVER = "planned arm to pitcher hover"

    def __init__(self, *, sim: KitchenPyBulletSim) -> None:
        self.sim = sim
        self.plans: dict[str, list[np.ndarray]] = {}

    def reset(
        self,
        *,
        x: Any,
        arm: np.ndarray,
        base: np.ndarray,
        base_target: np.ndarray,
        hover: np.ndarray,
        rotation: Rotation,
    ) -> None:
        sim = self.sim
        sim.set_state(x)
        if (
            np.linalg.norm(base_target[:2] - base[:2]) < 0.01
            and abs(base_target[2] - base[2]) < 0.01
        ):
            base_path = [base.copy(), base_target.copy()]
        else:
            base_path = KitchenMotionPlanner.plan_base(
                sim=sim, state=x, target=base_target, arm=arm, what="the pitcher pick pose"
            )
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        self.plans = {
            self.BASE_MOTION: base_path,
            self.MOVE_ARM_TO_HOVER: KitchenMotionPlanner.plan_arm(
                sim=sim, start=arm, position=hover, rotation=rotation, what="pitcher hover"
            ),
        }
