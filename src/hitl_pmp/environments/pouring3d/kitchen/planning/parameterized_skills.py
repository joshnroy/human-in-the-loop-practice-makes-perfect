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

from collections.abc import Callable
from typing import Any

import numpy as np
import pybullet as p
from kinder_models.dynamic3d.tossing.parameterized_skills import plan_checked_arm_motion
from kinder_models.dynamic3d.utils import (
    IKFAST_MAX_ATTEMPTS,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    run_base_motion_planning,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast.utils import IKFastHyperparameters, ikfast_closest_inverse_kinematics
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
)
from pybullet_helpers.link import get_link_state
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from spatialmath import SE2

from .utils import PLANNING_MARGIN_M, UPRIGHT_TILT_LIMIT_DEG, KitchenPyBulletSim


class PlanningRefused(ValueError):
    """A planner found no plan; nothing was executed."""


class HeldObject(BaseModel):
    """A grasped body carried rigidly by the hand, and what rides inside it."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    body: int
    base_link_to_held_obj: Pose
    # Bodies that move with the held one (dumplings in the jug); never obstacles to it.
    riders: tuple[int, ...] = ()

    @staticmethod
    def grasped(
        *, sim: KitchenPyBulletSim, arm: Any, name: str, riders: tuple[str, ...]
    ) -> "HeldObject":
        """Freeze the current hand-to-object transform, as upstream's pick does.

        Measured from the end-effector link's frame exactly as
        `set_robot_joints_with_held_object` applies it, so the planned object sits
        where MuJoCo has it at the start.
        """
        sim.robot.set_joints(KitchenMotionPlanner.with_fingers(sim=sim, arm=arm))
        world_to_hand = get_link_state(
            sim.robot.robot_id, sim.robot.end_effector_id, physics_client_id=sim.physics_client_id
        ).com_pose
        position, orientation = p.getBasePositionAndOrientation(
            sim.mirrored_id(name=name), physicsClientId=sim.physics_client_id
        )
        inverse = p.invertTransform(world_to_hand[0], world_to_hand[1])
        relative = p.multiplyTransforms(inverse[0], inverse[1], position, orientation)
        x, y, z = (float(v) for v in relative[0])
        qx, qy, qz, qw = (float(v) for v in relative[1])
        return HeldObject(
            body=sim.mirrored_id(name=name),
            base_link_to_held_obj=Pose((x, y, z), (qx, qy, qz, qw)),
            riders=tuple(sim.mirrored_id(name=r) for r in riders),
        )

    def obstacles(self, *, sim: KitchenPyBulletSim) -> set[int]:
        return sim.get_collision_bodies(held_object=self.body) - set(self.riders)

    def upright(self, *, sim: KitchenPyBulletSim, limit_deg: float) -> Callable[[Any], bool]:
        """True while the held object's axis stays within `limit_deg` of vertical.

        Upstream calls this right after its collision check has posed the robot and
        the held object, so it reads the pose the check just set.
        """
        cosine = float(np.cos(np.deg2rad(limit_deg)))
        body, client = self.body, sim.physics_client_id

        def within(_: Any) -> bool:  # noqa: PLR0917 -- upstream callback signature
            _, xyzw = p.getBasePositionAndOrientation(body, physicsClientId=client)
            return float(Rotation.from_quat(xyzw).apply([0.0, 0.0, 1.0])[2]) >= cosine

        return within


class KitchenMotionPlanner:
    """A static-method container, never instantiated."""

    tilt_limit_deg: float = UPRIGHT_TILT_LIMIT_DEG

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
        held: HeldObject | None = None,
    ) -> list[np.ndarray]:
        """A checked joint path from `start` to a valid IK solution for the hand pose.

        Holding something, it is carried rigidly, checked against the scene, and kept
        within `tilt_limit_deg` of vertical at every checked configuration.
        """
        start_joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=start)
        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        constraint = (
            None
            if held is None
            else held.upright(sim=sim, limit_deg=KitchenMotionPlanner.tilt_limit_deg)
        )

        def collides(q: Any) -> bool:  # noqa: PLR0917 -- upstream's collision callback
            if check_collisions_with_held_object(
                sim.robot,
                bodies,
                sim.physics_client_id,
                None if held is None else held.body,
                None if held is None else held.base_link_to_held_obj,
                list(q),
                distance_threshold=margin,
            ):
                return True
            return constraint is not None and not constraint(q)

        sim.robot.set_joints(start_joints)
        goal = KitchenMotionPlanner.goal(
            sim=sim, position=position, rotation=rotation, valid=collides
        )
        if goal is None:
            upright = " upright" if held is not None else ""
            raise PlanningRefused(f"No collision-free{upright} IK solution for {what}")
        planned = plan_checked_arm_motion(
            sim.robot,
            start_joints,
            goal,
            collision_bodies=bodies,
            physics_client_id=sim.physics_client_id,
            held_object=None if held is None else held.body,
            base_link_to_held_obj=None if held is None else held.base_link_to_held_obj,
            distance_threshold=margin,
            additional_state_constraint_fn=constraint,
        )
        if planned is None:
            upright = " upright" if held is not None else ""
            raise PlanningRefused(f"No collision-free{upright} arm path to {what}")
        return [np.asarray(q[:7], dtype=float) for q in planned[1]]

    @staticmethod
    def goal(
        *, sim: KitchenPyBulletSim, position: np.ndarray, rotation: Rotation, valid: Any
    ) -> list[float] | None:
        """The IK solution closest to the robot's current joints that is collision-free
        (and keeps any constraint); None if IKFast offers none.

        Upstream takes IKFast's closest solution outright. Near the stove that can be an
        elbow-down branch through the worktop when a clear one exists, so the candidates
        IKFast returns (sorted by closeness, a fixed sample budget) are tried in order.
        """
        px, py, pz = (float(v) for v in position)
        qx, qy, qz, qw = (float(v) for v in rotation.as_quat())
        target = Pose((px, py, pz), (qx, qy, qz, qw))
        current = list(sim.robot.get_joint_positions())
        solutions = ikfast_closest_inverse_kinematics(
            sim.robot,
            world_from_target=target,
            hyperparameters=IKFastHyperparameters(
                max_time=np.inf, max_attempts=IKFAST_MAX_ATTEMPTS
            ),
        )
        for solution in solutions:
            # The fingers keep their current opening.
            joints = [*(float(v) for v in solution[:7]), *current[7:]]
            sim.robot.set_joints(joints)
            reached = np.asarray(sim.robot.get_end_effector_pose().position)
            sim.robot.set_joints(current)
            if np.linalg.norm(reached - np.array([px, py, pz])) > 1e-3:
                continue
            if not valid(joints):
                return joints
        return None

    @staticmethod
    def plan_base(
        *,
        sim: KitchenPyBulletSim,
        state: Any,
        target: np.ndarray,
        arm: Any,
        what: str,
        held: HeldObject | None = None,
    ) -> list[np.ndarray]:
        """A base path to `target` (x, y, yaw), the arm (and anything held) carried as
        it is: the 2-D footprint against obstacle footprints, the 3-D body against the
        scene."""
        joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=arm)
        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        if held is not None:
            sim.base_link_to_held_obj = held.base_link_to_held_obj

        def body_collides(pose: SE2) -> bool:  # noqa: PLR0917 -- upstream callback
            return sim.robot_collides_at_base(
                pose,
                joints,
                held_object=None if held is None else held.body,
                collision_bodies=bodies,
                distance_threshold=PLANNING_MARGIN_M,
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


class CarryToPourController:
    """Carry the held jug from where it was lifted to the start of the pour.

    Three planned motions, all with the jug upright: the arm tucks the jug into a carry
    pose close to the body; the base drives to the pour pose with the arm and jug carried
    as tucked; the arm moves the jug to the pour start above the pan. The pour itself
    (tilt, wiggle, return) stays the scripted closed-loop controller.
    """

    TUCK = "planned tuck jug for carry"
    BASE_MOTION = "planned carry jug to pour pose"
    MOVE_ARM_TO_POUR = "planned arm to pour start"
    # The carry pose, in the mobile base's frame: hand in front of the base, high.
    CARRY_HAND_IN_BASE = (0.45, 0.0, 1.10)

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
        hand_rotation: Rotation,
        start: np.ndarray,
        start_rotation: Rotation,
        riders: tuple[str, ...],
    ) -> None:
        sim = self.sim
        sim.set_state(x)
        sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        held = HeldObject.grasped(sim=sim, arm=arm, name="pitcher", riders=riders)
        carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler("z", float(base[2])).apply(
            self.CARRY_HAND_IN_BASE
        )
        tuck = KitchenMotionPlanner.plan_arm(
            sim=sim,
            start=arm,
            position=carry,
            rotation=hand_rotation,
            what="the carry pose",
            held=held,
        )
        base_path = KitchenMotionPlanner.plan_base(
            sim=sim, state=x, target=base_target, arm=tuck[-1], what="the pour pose", held=held
        )
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        self.plans = {
            self.TUCK: tuck,
            self.BASE_MOTION: base_path,
            self.MOVE_ARM_TO_POUR: KitchenMotionPlanner.plan_arm(
                sim=sim,
                start=tuck[-1],
                position=start,
                rotation=start_rotation,
                what="the pour start",
                held=held,
            ),
        }


class PlacePitcherController:
    """Return the held jug from the pour to a hover just above its home on the counter.

    Planned, upright, before anything moves: the arm tucks the jug into the carry pose;
    the base drives to the pick pose, from which the arm demonstrably reaches the handle
    at its seated height; the arm brings the jug to a hover `HOVER_M` above home. The seat
    itself is a contact motion and stays scripted, as do release and the landing checks.
    """

    TUCK = "planned tuck jug for return"
    BASE_MOTION = "planned carry jug to placement pose"
    MOVE_ARM_TO_HOVER = "planned arm to jug hover above home"
    # The scripted seat's own alignment height (`move_pitcher`, "align pitcher above table").
    HOVER_M = 0.04

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
        hand_rotation: Rotation,
        hand_offset: np.ndarray,
        rotation_offset: Rotation,
        home: np.ndarray,
        riders: tuple[str, ...],
    ) -> None:
        sim = self.sim
        sim.set_state(x)
        sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        held = HeldObject.grasped(sim=sim, arm=arm, name="pitcher", riders=riders)
        carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler("z", float(base[2])).apply(
            CarryToPourController.CARRY_HAND_IN_BASE
        )
        tuck = KitchenMotionPlanner.plan_arm(
            sim=sim,
            start=arm,
            position=carry,
            rotation=hand_rotation,
            what="the carry pose",
            held=held,
        )
        base_path = KitchenMotionPlanner.plan_base(
            sim=sim, state=x, target=base_target, arm=tuck[-1], what="placement", held=held
        )
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        # The jug upright over home; the hand where the current grasp puts it.
        jug = home + np.array([0.0, 0.0, self.HOVER_M])
        self.plans = {
            self.TUCK: tuck,
            self.BASE_MOTION: base_path,
            self.MOVE_ARM_TO_HOVER: KitchenMotionPlanner.plan_arm(
                sim=sim,
                start=tuck[-1],
                position=jug + hand_offset,
                rotation=rotation_offset,
                what="the jug hover above home",
                held=held,
            ),
        }


class ClearWorkspaceController:
    """After the release and the scripted withdrawal of the hand from the handle (the
    mirror of the scripted descent onto it): back the base off, arm carried as is."""

    BASE_MOTION = "planned clear workspace"

    def __init__(self, *, sim: KitchenPyBulletSim) -> None:
        self.sim = sim
        self.plans: dict[str, list[np.ndarray]] = {}

    def reset(self, *, x: Any, arm: np.ndarray, base_target: np.ndarray) -> None:
        self.sim.set_state(x)
        self.plans = {
            self.BASE_MOTION: KitchenMotionPlanner.plan_base(
                sim=self.sim, state=x, target=base_target, arm=arm, what="the cleared workspace"
            )
        }
