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
from typing import Any, ClassVar

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.object_types import (
    MujocoDrawerObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder_models.dynamic3d.tossing.parameterized_skills import plan_checked_arm_motion
from kinder_models.dynamic3d.utils import (
    IKFAST_MAX_ATTEMPTS,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    get_bounding_box,
    get_overhead_kinematic2ds,
    get_overhead_robot_se2_pose,
    run_base_motion_planning,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast.utils import IKFastHyperparameters, ikfast_closest_inverse_kinematics
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.link import get_link_state
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from spatialmath import SE2
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from .utils import PLANNING_MARGIN_M, UPRIGHT_TILT_LIMIT_DEG, KitchenPyBulletSim

# An arm motion whose start lies inside the planning margin, but at least this far from
# everything, first escapes the margin at this margin, then continues at the full one.
# MuJoCo's IK keeps the same 15 mm in its own robot model, which is under a millimetre
# off PyBullet's, so a skill can end 14.4 mm from the scene as PyBullet measures it
# (ees/ees/0 of Della job 16949734); without the escape, BiRRT refuses its own start.
ESCAPE_MARGIN_M = 0.005

# The arm's continuous joints (1, 3, 5, 7) and MuJoCo's range for them. PyBullet models
# them as unlimited, so the range is enforced here, on the path as MuJoCo follows it.
CONTINUOUS = (0, 2, 4, 6)
CONTINUOUS_LIMIT_RAD = 12.0
# A held skill's own IK refuses within 20 deg of the range (`held_bound_margin`), so a
# turn chosen for a planned end keeps that much room too.
CONTINUOUS_LIMIT_MARGIN_RAD = float(np.deg2rad(20.0))
# Joint-space spacing at which a re-turned path is collision-checked, and the largest
# per-joint step between the waypoints it is returned as (the remap distance, 0.2).
RETURN_CHECK_RAD = 0.02
RETURN_WAYPOINT_RAD = 0.2


# MuJoCo's joint limits are soft: an arm resting at one sits slightly past it.
EXECUTED_LIMIT_SLACK_RAD = 0.02


class PlanningRefused(ValueError):
    """A planner found no plan; nothing was executed."""


class HeldObject(BaseModel):
    """A grasped body carried rigidly by the hand, and what rides inside it."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    body: int
    base_link_to_held_obj: Pose
    # Bodies that move with the held one (dumplings in the jug); never obstacles to it.
    riders: tuple[int, ...] = ()
    # The jug must stay upright while carried; a welded dumpling may take any orientation.
    keep_upright: bool = True

    @staticmethod
    def grasped(
        *,
        sim: KitchenPyBulletSim,
        arm: Any,
        name: str,
        riders: tuple[str, ...],
        keep_upright: bool = True,
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
            keep_upright=keep_upright,
        )

    def obstacles(self, *, sim: KitchenPyBulletSim) -> set[int]:
        return sim.get_collision_bodies(held_object=self.body) - set(self.riders)

    def tilt_deg(self, *, sim: KitchenPyBulletSim, arm: Any) -> float:
        """The held object's tilt from vertical with the arm at `arm`."""
        set_robot_joints_with_held_object(
            sim.robot, sim.physics_client_id, self.body, self.base_link_to_held_obj, list(arm)
        )
        _, xyzw = p.getBasePositionAndOrientation(self.body, physicsClientId=sim.physics_client_id)
        axis_z = float(Rotation.from_quat(xyzw).apply([0.0, 0.0, 1.0])[2])
        return float(np.rad2deg(np.arccos(np.clip(axis_z, -1.0, 1.0))))

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
    # The Gen3's continuous joints (1, 3, 5, 7), as MuJoCo models them.
    CONTINUOUS_JOINTS: ClassVar[list[int]] = [0, 2, 4, 6]

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
        reference: Any | None = None,
        neutral: Any | None = None,
    ) -> list[np.ndarray]:
        """A checked joint path from `start` to a valid IK solution for the hand pose.

        Holding something, it is carried rigidly, checked against the scene, and kept
        within `tilt_limit_deg` of vertical at every checked configuration. The goal is
        the valid solution closest to `reference` if given, else to `start`.

        A `start` inside `margin` but clear of `ESCAPE_MARGIN_M` is not refused: the
        motion escapes at `ESCAPE_MARGIN_M` until it is first outside `margin`, and is
        planned at `margin` from there. A start outside `margin` plans as it always has.
        """
        return KitchenMotionPlanner.plan_arm_motion(
            sim=sim,
            start=start,
            what=what,
            margin=margin,
            held=held,
            target=(position, rotation, reference),
        )

    @staticmethod
    def plan_arm_to_joints(
        *,
        sim: KitchenPyBulletSim,
        start: Any,
        goal: Any,
        what: str,
        margin: float = PLANNING_MARGIN_M,
    ) -> list[np.ndarray]:
        """`plan_arm` to a given arm configuration rather than a hand pose: the goal is a
        configuration another model has already checked (the pan pick's hover, from which
        MuJoCo's IK reaches the grasp). Continuous joints are taken the short way round,
        as `follow_arm_path` executes them; the goal must be clear at `margin`."""
        return KitchenMotionPlanner.plan_arm_motion(
            sim=sim, start=start, what=what, margin=margin, held=None, joints=goal
        )

    @staticmethod
    def plan_arm_motion(
        *,
        sim: KitchenPyBulletSim,
        start: Any,
        what: str,
        margin: float,
        held: HeldObject | None,
        target: tuple[np.ndarray, Rotation, Any | None] | None = None,
        joints: Any | None = None,
    ) -> list[np.ndarray]:
        """`plan_arm` and `plan_arm_to_joints`: to the IK solution for `target` (hand
        position, rotation, reference), or to the arm configuration `joints`."""
        if sim.executed_arm_limits is not None:
            # An arm resting at a soft MuJoCo limit sits slightly past it; plan from the
            # limit itself, never further out.
            lower, upper = sim.executed_arm_limits
            arm_now = np.asarray(start[:7], dtype=float)
            inside = np.clip(arm_now, lower, upper)
            if np.max(np.abs(inside - arm_now)) <= EXECUTED_LIMIT_SLACK_RAD:
                start = inside
        start_joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=start)
        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        constraint = None
        # Named in a refusal only when it binds: a welded dumpling has no upright limit.
        upright = " upright" if held is not None and held.keep_upright else ""
        if held is not None and held.keep_upright:
            # A compliant grasp can leave the jug hanging past the limit when a skill
            # starts (measured 10.2 deg after a pour's return to upright). Never tilt it
            # further than it starts; plan towards upright.
            limit = max(
                KitchenMotionPlanner.tilt_limit_deg,
                held.tilt_deg(sim=sim, arm=start_joints) + 0.1,
            )
            constraint = held.upright(sim=sim, limit_deg=limit)

        limits = sim.executed_arm_limits
        if limits is not None:
            upright_only = constraint

            def executable(q: Any) -> bool:  # noqa: PLR0917 -- upstream callback signature
                arm_q = np.asarray(q[:7], dtype=float)
                if np.any(arm_q < limits[0]) or np.any(arm_q > limits[1]):
                    return False
                return upright_only is None or upright_only(q)

            constraint = executable

        def within(q: Any, threshold: float) -> bool:  # noqa: PLR0917 -- local helper
            if limits is not None:
                arm_q = np.asarray(q[:7], dtype=float)
                if np.any(arm_q < limits[0]) or np.any(arm_q > limits[1]):
                    return True
            if check_collisions_with_held_object(
                sim.robot,
                bodies,
                sim.physics_client_id,
                None if held is None else held.body,
                None if held is None else held.base_link_to_held_obj,
                list(q),
                distance_threshold=threshold,
            ):
                return True
            return constraint is not None and not constraint(q)

        def collides(q: Any) -> bool:  # noqa: PLR0917 -- upstream's collision callback
            return within(q, margin)

        def checked(a: Any, b: Any, threshold: float) -> list[Any] | None:  # noqa: PLR0917 -- local helper
            plan = plan_checked_arm_motion(
                sim.robot,
                list(a),
                list(b),
                collision_bodies=bodies,
                physics_client_id=sim.physics_client_id,
                held_object=None if held is None else held.body,
                base_link_to_held_obj=None if held is None else held.base_link_to_held_obj,
                distance_threshold=threshold,
                additional_state_constraint_fn=constraint,
            )
            return None if plan is None else list(plan[1])

        sim.robot.set_joints(start_joints)
        if target is not None:
            position, rotation, reference = target
            goal = KitchenMotionPlanner.goal(
                sim=sim,
                position=position,
                rotation=rotation,
                valid=collides,
                reference=None
                if reference is None
                else KitchenMotionPlanner.with_fingers(sim=sim, arm=reference),
            )
        else:
            arm_goal = np.asarray(joints, dtype=float)[:7].copy()
            continuous = KitchenMotionPlanner.CONTINUOUS_JOINTS
            turn = arm_goal[continuous] - np.asarray(start_joints, dtype=float)[continuous]
            arm_goal[continuous] -= turn - np.arctan2(np.sin(turn), np.cos(turn))
            goal = KitchenMotionPlanner.with_fingers(sim=sim, arm=arm_goal)
            if collides(goal):
                raise PlanningRefused(f"No collision-free{upright} configuration for {what}")
        if goal is None:
            raise PlanningRefused(f"No collision-free{upright} IK solution for {what}")
        planned = checked(start_joints, goal, margin)
        if planned is None and collides(start_joints) and not within(start_joints, ESCAPE_MARGIN_M):
            escape = checked(start_joints, goal, ESCAPE_MARGIN_M)
            if escape is not None:
                # The goal itself is outside the margin, so the escape always leaves it.
                out = next(i for i, q in enumerate(escape) if not collides(q))
                rest = checked(escape[out], goal, margin) if out < len(escape) - 1 else [goal]
                planned = None if rest is None else [*escape[: out + 1], *rest[1:]]
        if planned is None:
            raise PlanningRefused(f"No collision-free{upright} arm path to {what}")
        path = KitchenMotionPlanner.unwrapped(path=planned)
        if neutral is not None:
            turns = KitchenMotionPlanner.turns_toward(
                end=path[-1], neutral=np.asarray(neutral, dtype=float)
            )
            if turns.any():
                returned = KitchenMotionPlanner.returned(
                    path=path, turns=turns, collides=lambda q: collides([*q, *start_joints[7:]])
                )
                if returned is not None:
                    path = returned
        KitchenMotionPlanner.check_range(path=path, what=what)
        return path

    @staticmethod
    def returned(
        *, path: list[np.ndarray], turns: np.ndarray, collides: Callable[[list[float]], bool]
    ) -> list[np.ndarray] | None:
        """`path` ending on the turn `turns` whole turns away, the most preferred clear
        version of it; None if every one collides.

        A whole turn leaves every configuration's hand pose unchanged, but not those on
        the way, so each candidate is checked again as MuJoCo will follow it, every
        `RETURN_CHECK_RAD` along its straight segments. Candidates, in order: all of
        `turns` first, then one turn fewer per joint down to one; for each, the turns
        made at the start (where the unwind before a pick makes them), then spread
        along the path in proportion to its joint-space length.
        """
        steps = np.array([np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:], strict=True)])
        total = float(steps.sum())
        spread = (
            np.concatenate([[0.0], np.cumsum(steps) / total])
            if total > 0
            else np.linspace(0.0, 1.0, len(path))
        )
        for most in range(int(np.max(np.abs(turns))), 0, -1):
            chosen = np.clip(turns, -most, most)
            at_start = [path[0], *(q + 2 * np.pi * chosen for q in path)]
            along = [q + 2 * np.pi * chosen * u for q, u in zip(path, spread, strict=True)]
            for candidate in (at_start, along):
                out = KitchenMotionPlanner.densified(path=candidate, collides=collides)
                if out is not None:
                    return out
        return None

    @staticmethod
    def densified(
        *, path: list[np.ndarray], collides: Callable[[list[float]], bool]
    ) -> list[np.ndarray] | None:
        """`path` with no per-joint step over `RETURN_WAYPOINT_RAD`, or None if a
        configuration on it, checked every `RETURN_CHECK_RAD`, collides."""
        out = [path[0]]
        for a, b in zip(path[:-1], path[1:], strict=True):
            span = float(np.max(np.abs(b - a)))
            for u in np.linspace(0, 1, 1 + max(1, int(np.ceil(span / RETURN_CHECK_RAD))))[1:]:
                if collides([float(v) for v in a + u * (b - a)]):
                    return None
            count = max(1, int(np.ceil(span / RETURN_WAYPOINT_RAD)))
            out.extend(a + (b - a) * (i / count) for i in range(1, count + 1))
        return out

    @staticmethod
    def unwrapped(*, path: list[Any]) -> list[np.ndarray]:
        """`path` as MuJoCo follows it from `path[0]`: each continuous joint the short
        way round between waypoints (`KitchenController.follow_arm_path`)."""
        out = [np.asarray(path[0], dtype=float)[:7].copy()]
        index = list(CONTINUOUS)
        for q in path[1:]:
            delta = np.asarray(q, dtype=float)[:7] - out[-1]
            delta[index] = np.arctan2(np.sin(delta[index]), np.cos(delta[index]))
            out.append(out[-1] + delta)
        return out

    @staticmethod
    def turns_toward(*, end: np.ndarray, neutral: np.ndarray) -> np.ndarray:
        """Whole turns to add to each continuous joint of `end` to bring it nearest
        `neutral`, kept inside MuJoCo's range less its margin; zero elsewhere."""
        turns = np.zeros(7, dtype=int)
        bound = CONTINUOUS_LIMIT_RAD - CONTINUOUS_LIMIT_MARGIN_RAD
        for j in CONTINUOUS:
            k = round((float(neutral[j]) - float(end[j])) / (2 * np.pi))
            candidates = [k, k - 1, k + 1]
            inside = [c for c in candidates if abs(end[j] + 2 * np.pi * c) <= bound]
            if inside:
                turns[j] = min(inside, key=lambda c: abs(end[j] + 2 * np.pi * c - neutral[j]))
        return turns

    @staticmethod
    def check_range(*, path: list[np.ndarray], what: str) -> None:
        """Refuse a path that carries a continuous joint past MuJoCo's range, or further
        past it than it starts."""
        for j in CONTINUOUS:
            reach = max(abs(float(q[j])) for q in path)
            if reach > max(CONTINUOUS_LIMIT_RAD, abs(float(path[0][j]))):
                raise PlanningRefused(
                    f"The arm path to {what} carries continuous joint {j + 1} to "
                    f"{np.rad2deg(reach):.0f} deg, past its range "
                    f"({np.rad2deg(CONTINUOUS_LIMIT_RAD):.0f} deg)"
                )

    @staticmethod
    def goal(
        *,
        sim: KitchenPyBulletSim,
        position: np.ndarray,
        rotation: Rotation,
        valid: Any,
        reference: list[float] | None = None,
    ) -> list[float] | None:
        """The IK solution closest to `reference` (default: the robot's current joints)
        that is collision-free (and keeps any constraint); None if IKFast offers none.

        Upstream takes IKFast's closest solution outright. Near the stove that can be an
        elbow-down branch through the worktop when a clear one exists, so the candidates
        IKFast returns (sorted by closeness, a fixed sample budget) are tried in order.
        Closeness is upstream's own measure (continuous joints the short way round),
        taken from `reference` by posing the robot there for the sort.
        """
        px, py, pz = (float(v) for v in position)
        qx, qy, qz, qw = (float(v) for v in rotation.as_quat())
        target = Pose((px, py, pz), (qx, qy, qz, qw))
        current = list(sim.robot.get_joint_positions())
        if reference is not None:
            sim.robot.set_joints(list(reference))
        solutions = ikfast_closest_inverse_kinematics(
            sim.robot,
            world_from_target=target,
            hyperparameters=IKFastHyperparameters(
                max_time=np.inf, max_attempts=IKFAST_MAX_ATTEMPTS
            ),
        )
        sim.robot.set_joints(current)
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
        ignore: tuple[str, ...] = (),
        also_collides: Callable[[SE2], bool] | None = None,
    ) -> list[np.ndarray]:
        """A base path to `target` (x, y, yaw), the arm (and anything held) carried as
        it is: the 2-D footprint against obstacle footprints, the 3-D body against the
        scene. `ignore` names objects with no 2-D footprint, such as one the hand
        carries above the base. `also_collides` is any further check of a base pose."""
        body_collides = KitchenMotionPlanner.body_check(
            sim=sim, arm=arm, held=held, also_collides=also_collides
        )
        path = run_base_motion_planning(
            state=state,
            target_base_pose=SE2(float(target[0]), float(target[1]), float(target[2])),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=list(ignore) or None,
            body_collision_fn=body_collides,
        )
        if path is None:
            raise PlanningRefused(f"No collision-free base path to {what}")
        return [np.array([pose.x, pose.y, pose.theta()]) for pose in path]

    @staticmethod
    def body_check(
        *,
        sim: KitchenPyBulletSim,
        arm: Any,
        held: HeldObject | None = None,
        also_collides: Callable[[SE2], bool] | None = None,
    ) -> Callable[[SE2], bool]:
        """Whether the robot's 3-D body, the arm (and anything held) carried as it is,
        collides with the scene at a base pose, at the full planning margin."""
        joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=arm)
        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        if held is not None:
            sim.base_link_to_held_obj = held.base_link_to_held_obj

        def body_collides(pose: SE2) -> bool:  # noqa: PLR0917 -- upstream callback
            if also_collides is not None and also_collides(pose):
                return True
            return sim.robot_collides_at_base(
                pose,
                joints,
                held_object=None if held is None else held.body,
                collision_bodies=bodies,
                distance_threshold=PLANNING_MARGIN_M,
            )

        return body_collides

    @staticmethod
    def base_pose_blocked(
        *,
        sim: KitchenPyBulletSim,
        state: Any,
        target: np.ndarray,
        arm: Any,
        held: HeldObject | None = None,
    ) -> bool:
        """Whether the base pose `target` is itself in collision, by the check
        `run_base_motion_planning` applies to every pose: the 2-D footprint against the
        obstacle footprints, then the 3-D body. No path can end at a blocked pose."""
        (robot,) = state.get_objects(MujocoTidyBotRobotObjectType)
        width, height, _ = get_bounding_box(state, robot)
        geoms = get_overhead_kinematic2ds(state, None)
        obstacles = [
            geoms[o.name]
            for o in state.get_objects(MujocoObjectType)
            if o.name in geoms and not o.is_instance(MujocoDrawerObjectType)
        ]
        moved = state.copy()
        for feature, value in zip(
            ("pos_base_x", "pos_base_y", "pos_base_rot"), target, strict=True
        ):
            moved.set(robot, feature, float(value))
        pose = get_overhead_robot_se2_pose(moved, robot)
        footprint = Rectangle.from_center(
            pose.x, pose.y, width, height, rotation_about_center=pose.theta()
        )
        if any(geom2ds_intersect(footprint, obstacle) for obstacle in obstacles):
            return True
        se2 = SE2(float(target[0]), float(target[1]), float(target[2]))
        return KitchenMotionPlanner.body_check(sim=sim, arm=arm, held=held)(se2)


class PickPitcherController:
    """The free-space part of PickPitcher: the base to the pick pose, then the arm to a
    hover above the handle. The descent, grasp and lift that follow are contact
    motions and stay scripted (`grasp_and_lift_pitcher`).

    The hover's IK solution is the one closest to `home` (the episode-initial arm), not
    to the arm as it is. The grasp carries the hover's branch through the whole cycle,
    and after dumpling drops the arm can sit nearer the mirror (shoulder-flipped)
    branch: the same hand pose, but no upright carry path out of it. A fresh episode
    starts at home, so its pick is unchanged.
    """

    TUCK = "planned tuck arm before pitcher pick"
    BASE_MOTION = "planned base to pitcher"
    MOVE_ARM_TO_HOVER = "planned arm to pitcher hover"
    # Within this of `home` (the episode-initial arm) on every joint, continuous joints
    # mod a turn, the arm is already tucked: a fresh episode's pick is unchanged.
    TUCKED_RAD = 0.01

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
        home: np.ndarray,
    ) -> None:
        sim = self.sim
        sim.set_state(x)
        tuck: list[np.ndarray] = []
        if (
            np.linalg.norm(base_target[:2] - base[:2]) < 0.01
            and abs(base_target[2] - base[2]) < 0.01
        ):
            base_path = [base.copy(), base_target.copy()]
        else:
            # A failed dumpling skill can leave the arm reaching out over the stove,
            # where no base path exists for it as it is: tuck it into the carry pose
            # (the jug carries' own, hand oriented for the hover) at the full margin.
            if not self.tucked(arm=arm, home=home):
                carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler(
                    "z", float(base[2])
                ).apply(CarryToPourController.CARRY_HAND_IN_BASE)
                tuck = KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=arm,
                    position=carry,
                    rotation=rotation,
                    what="the tucked arm",
                    reference=home,
                    neutral=home,
                )
                arm = tuck[-1]
            base_path = KitchenMotionPlanner.plan_base(
                sim=sim, state=x, target=base_target, arm=arm, what="the pitcher pick pose"
            )
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        self.plans = {
            **({self.TUCK: tuck} if tuck else {}),
            self.BASE_MOTION: base_path,
            self.MOVE_ARM_TO_HOVER: KitchenMotionPlanner.plan_arm(
                sim=sim,
                start=arm,
                position=hover,
                rotation=rotation,
                what="pitcher hover",
                reference=home,
                neutral=home,
            ),
        }

    @staticmethod
    def nearest(*, home: np.ndarray, arm: np.ndarray) -> np.ndarray:
        """`home`, its continuous joints (1, 3, 5, 7) on the turn nearest `arm`."""
        goal = np.asarray(home, dtype=float).copy()
        turns = np.round((np.asarray(arm)[[0, 2, 4, 6]] - goal[[0, 2, 4, 6]]) / (2 * np.pi))
        goal[[0, 2, 4, 6]] += 2 * np.pi * turns
        return goal

    @classmethod
    def tucked(cls, *, arm: np.ndarray, home: np.ndarray) -> bool:
        delta = np.asarray(arm, dtype=float) - cls.nearest(home=home, arm=arm)
        return bool(np.max(np.abs(delta)) < cls.TUCKED_RAD)


class PickShrimpController:
    """The free-space part of a dumpling pick from the pan: tuck the empty arm if the
    base cannot move with it as it is, drive the base to the pick pose, then move the arm
    to a hover above the dumpling. The vertical descent, pinch and lift that follow stay
    the pick's own contact motions (`KitchenController.pick_from_pan`).

    The hover is a configuration, not a hand pose: the one MuJoCo's IK reached from the
    grasp going straight up, so the descent from it is known to reach the grasp. The
    caller offers hovers for each grasp orientation in turn (`plan_hover`); the tuck and
    base path are planned once (`plan_base`), since the pick pose depends only on where
    the dumpling lies along the counter.
    """

    TUCK = "planned tuck arm before shrimp pick"
    BASE_MOTION = "planned base to shrimp pick pose"
    MOVE_ARM_TO_HOVER = "planned arm to shrimp hover"
    # Tuck orientations after the caller's own: the hand down, then tilted 30 and 60 deg,
    # turned about vertical in 45 deg steps. The hand is empty, so any of them will do.
    TUCK_PITCHES_DEG = (0.0, 30.0, 60.0)
    TUCK_YAWS_DEG = (-90.0, 0.0, 90.0, 180.0, -45.0, 45.0, 135.0, -135.0)

    def __init__(self, *, sim: KitchenPyBulletSim) -> None:
        self.sim = sim
        self.plans: dict[str, list[np.ndarray]] = {}
        # The arm as the base motion carries it: as it was, or tucked.
        self.carried = np.zeros(7)

    def plan_base(
        self,
        *,
        x: Any,
        arm: np.ndarray,
        base: np.ndarray,
        base_target: np.ndarray,
        rotations: list[Rotation],
        home: np.ndarray,
    ) -> None:
        """Plan the base to `base_target` with the arm as it is, else after a tuck into
        the carry pose at the first of `rotations` (then `TUCK_*`) from which it plans."""
        sim = self.sim
        sim.set_state(x)
        sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        self.plans = {}
        self.carried = np.asarray(arm, dtype=float).copy()
        if (
            np.linalg.norm(base_target[:2] - base[:2]) < 0.01
            and abs(base_target[2] - base[2]) < 0.01
        ):
            self.plans[self.BASE_MOTION] = [base.copy(), base_target.copy()]
            return
        try:
            self.plans[self.BASE_MOTION] = KitchenMotionPlanner.plan_base(
                sim=sim, state=x, target=base_target, arm=arm, what="the shrimp pick pose"
            )
            return
        except PlanningRefused as exc:
            refusals = [f"arm as it is: {exc}"]
        carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler("z", float(base[2])).apply(
            CarryToPourController.CARRY_HAND_IN_BASE
        )
        down = Rotation.from_euler("y", np.pi)
        options = [
            *rotations,
            *(
                Rotation.from_euler("z", yaw, degrees=True)
                * Rotation.from_euler("x", pitch, degrees=True)
                * down
                for pitch in self.TUCK_PITCHES_DEG
                for yaw in self.TUCK_YAWS_DEG
            ),
        ]
        for rotation in options:
            sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
            try:
                tuck = KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=arm,
                    position=carry,
                    rotation=rotation,
                    what="the tucked arm",
                    reference=home,
                )
                path = KitchenMotionPlanner.plan_base(
                    sim=sim, state=x, target=base_target, arm=tuck[-1], what="the shrimp pick pose"
                )
            except PlanningRefused as exc:
                refusals.append(str(exc))
                continue
            self.plans = {self.TUCK: tuck, self.BASE_MOTION: path}
            self.carried = tuck[-1]
            return
        raise PlanningRefused(
            f"No base path to the shrimp pick pose with the arm tucked in any of "
            f"{len(options)} orientations ({refusals[0]}; then {refusals[-1]})"
        )

    def plan_hover(self, *, hover: np.ndarray, base_target: np.ndarray) -> None:
        """Plan the arm, from where the base motion leaves it, to the configuration
        `hover` at `base_target`."""
        sim = self.sim
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        self.plans[self.MOVE_ARM_TO_HOVER] = KitchenMotionPlanner.plan_arm_to_joints(
            sim=sim, start=self.carried, goal=hover, what="the shrimp hover"
        )


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
        neutral: np.ndarray,
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
            neutral=neutral,
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
                neutral=neutral,
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
    # Alternatives to an occupied nominal base pose: offsets along the counter and yaw
    # changes, alone and combined. At most `MAX_PLANNED_ALTERNATIVES` unoccupied ones
    # are planned before the refusal stands.
    ALONG_COUNTER_M = (0.05, 0.10, 0.15, 0.20)
    YAW_RAD = (0.10, 0.20)
    MAX_PLANNED_ALTERNATIVES = 6

    def __init__(self, *, sim: KitchenPyBulletSim) -> None:
        self.sim = sim
        self.plans: dict[str, list[np.ndarray]] = {}
        # Where the base stands for the placement: the nominal pose unless it is occupied.
        self.base_target = np.zeros(3)

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
        neutral: np.ndarray,
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
            neutral=neutral,
        )
        # The jug upright over home; the hand where the current grasp puts it.
        jug = home + np.array([0.0, 0.0, self.HOVER_M])

        def plan_from(target: np.ndarray) -> dict[str, list[np.ndarray]]:  # noqa: PLR0917
            base_path = KitchenMotionPlanner.plan_base(
                sim=sim, state=x, target=target, arm=tuck[-1], what="placement", held=held
            )
            sim.set_robot_base(SE2(float(target[0]), float(target[1]), float(target[2])))
            return {
                self.TUCK: tuck,
                self.BASE_MOTION: base_path,
                self.MOVE_ARM_TO_HOVER: KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=tuck[-1],
                    position=jug + hand_offset,
                    rotation=rotation_offset,
                    what="the jug hover above home",
                    held=held,
                    neutral=neutral,
                ),
            }

        self.base_target = np.asarray(base_target, dtype=float).copy()
        try:
            self.plans = plan_from(self.base_target)
            return
        except PlanningRefused as exc:
            if not KitchenMotionPlanner.base_pose_blocked(
                sim=sim, state=x, target=self.base_target, arm=tuck[-1], held=held
            ):
                raise
            refusal = exc
        # The nominal pose itself is occupied (a dumpling flung onto the floor where the
        # base must stand), so no path can end there. Place from the nearest pose close
        # by from which the arm reaches the same hover above home, at the same margins.
        planned = 0
        for target in self.alternative_base_targets(nominal=self.base_target):
            if planned == self.MAX_PLANNED_ALTERNATIVES:
                break
            if KitchenMotionPlanner.base_pose_blocked(
                sim=sim, state=x, target=target, arm=tuck[-1], held=held
            ):
                continue
            planned += 1
            try:
                self.plans = plan_from(target)
            except PlanningRefused:
                continue
            self.base_target = target
            return
        raise refusal

    @classmethod
    def alternative_base_targets(cls, *, nominal: np.ndarray) -> list[np.ndarray]:
        """Base poses near `nominal`, nearest first: offsets along the counter (the
        nominal pose's lateral axis), small yaw changes and both, never `nominal`."""
        nominal = np.asarray(nominal, dtype=float)
        lateral = Rotation.from_euler("z", float(nominal[2])).apply([0.0, 1.0, 0.0])[:2]
        offsets = (0.0, *(sign * d for d in cls.ALONG_COUNTER_M for sign in (-1, 1)))
        turns = (0.0, *(sign * d for d in cls.YAW_RAD for sign in (-1, 1)))
        targets = [
            np.array([*(nominal[:2] + offset * lateral), nominal[2] + turn])
            for offset in offsets
            for turn in turns
            if offset or turn
        ]
        return sorted(targets, key=lambda t: cls.alternative_distance(nominal=nominal, target=t))

    @staticmethod
    def alternative_distance(*, nominal: np.ndarray, target: np.ndarray) -> float:
        """The upstream base planner's own pose distance: translation plus |yaw|."""
        turn = float(np.arctan2(np.sin(target[2] - nominal[2]), np.cos(target[2] - nominal[2])))
        return float(np.linalg.norm(np.asarray(target[:2]) - nominal[:2])) + abs(turn)


class DropShrimpController:
    """Carry a held dumpling from wherever it was picked to a hover above the jug.

    Planned before anything moves, as the jug's carries are: the arm tucks the dumpling
    into the carry pose; the base drives to the drop pose; the arm brings the dumpling
    above the jug's mouth. The dumpling is welded to the hand, so unlike the jug it may
    take any orientation on the way: the grasp's own orientation turned about vertical
    (`CARRY_TURNS_DEG`), then the hand straight down (`HOVER_DOWN_YAWS_DEG`), at each
    carry pose in `CARRY_HANDS_IN_BASE` in turn, until one plans. The centring and
    release over the mouth stay the scripted contact motion.
    """

    TUCK = "planned tuck held shrimp for carry"
    BASE_MOTION = "planned carry shrimp to drop pose"
    MOVE_ARM_TO_HOVER = "planned arm to shrimp hover above jug"
    # The jug's carry pose first. Holding the jug by its handle the hand is level there;
    # holding a dumpling it points down, and a downward hand reaches only about 0.97 m at
    # that distance. The lower pose is also nearer the base, so at the drop pose the
    # hanging dumpling stops short of the jug's rim. The base path checks the arm and the
    # dumpling in 3-D against the scene (worktops at 0.74 m).
    CARRY_HANDS_IN_BASE = (CarryToPourController.CARRY_HAND_IN_BASE, (0.35, 0.0, 0.95))
    CARRY_TURNS_DEG = (0.0, 90.0, -90.0, 180.0)
    HOVER_DOWN_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
    # From the drop pose the hover is about 0.5 m ahead and 1 m up, out of a vertical
    # hand's reach; a hand tilted toward the jug reaches it.
    HOVER_TILTS_TOWARD_JUG_DEG = (20.0, 40.0)

    def __init__(self, *, sim: KitchenPyBulletSim) -> None:
        self.sim = sim
        self.plans: dict[str, list[np.ndarray]] = {}
        self.hand_rotation = Rotation.identity()

    def reset(
        self,
        *,
        x: Any,
        arm: np.ndarray,
        base: np.ndarray,
        base_target: np.ndarray,
        name: str,
        hand_rotation: Rotation,
        payload_in_hand: np.ndarray,
        hover: np.ndarray,
    ) -> None:
        """`payload_in_hand` is the dumpling's offset from the hand in the hand's frame;
        `hover` is where the dumpling itself should end, above the mouth."""
        sim = self.sim
        sim.set_state(x)
        sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        held = HeldObject.grasped(sim=sim, arm=arm, name=name, riders=(), keep_upright=False)
        down = Rotation.from_euler("x", 180, degrees=True)
        orientations = [
            *(
                (f"turned {t:+.0f} deg", Rotation.from_euler("z", t, degrees=True) * hand_rotation)
                for t in self.CARRY_TURNS_DEG
            ),
            *(
                (f"down at yaw {t:.0f} deg", Rotation.from_euler("z", t, degrees=True) * down)
                for t in self.HOVER_DOWN_YAWS_DEG
            ),
        ]
        refusals: list[str] = []
        carried: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        for hand_in_base in self.CARRY_HANDS_IN_BASE:
            carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler(
                "z", float(base[2])
            ).apply(hand_in_base)
            for label, rotation in orientations:
                sim.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
                try:
                    tuck = KitchenMotionPlanner.plan_arm(
                        sim=sim,
                        start=arm,
                        position=carry,
                        rotation=rotation,
                        what="the shrimp carry pose",
                        held=held,
                    )
                    base_path = KitchenMotionPlanner.plan_base(
                        sim=sim,
                        state=x,
                        target=base_target,
                        arm=tuck[-1],
                        what="the drop pose",
                        held=held,
                        ignore=(name,),
                    )
                except PlanningRefused as exc:
                    refusals.append(f"carry at {hand_in_base[2]:.2f} m, {label}: {exc}")
                    continue
                carried = (tuck, base_path)
                break
            if carried is not None:
                break
        if carried is None:
            raise PlanningRefused("; ".join(refusals))
        tuck, base_path = carried
        # The hand may turn again between the carry and the hover: the weld holds. The
        # grasp's own orientation (turned about vertical) first, then the hand pointing
        # straight down, for a grasp too tilted to reach over the mouth.
        sim.set_robot_base(SE2(float(base_target[0]), float(base_target[1]), float(base_target[2])))
        ahead = np.array([hover[0] - base_target[0], hover[1] - base_target[1], 0.0])
        ahead /= np.linalg.norm(ahead)
        toward_jug = np.cross(ahead, [0.0, 0.0, 1.0])
        hovers = [
            *(rotation for _, rotation in orientations),
            *(
                Rotation.from_rotvec(np.deg2rad(tilt) * toward_jug)
                * Rotation.from_euler("z", t, degrees=True)
                * down
                for tilt in self.HOVER_TILTS_TOWARD_JUG_DEG
                for t in self.HOVER_DOWN_YAWS_DEG
            ),
        ]
        for turn, rotation in enumerate(hovers):
            try:
                hover_arm = KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=tuck[-1],
                    position=hover - rotation.apply(payload_in_hand),
                    rotation=rotation,
                    what="the shrimp hover above the jug",
                    held=held,
                )
            except PlanningRefused as exc:
                refusals.append(f"hover orientation {turn}: {exc}")
                continue
            self.hand_rotation = rotation
            self.plans = {
                self.TUCK: tuck,
                self.BASE_MOTION: base_path,
                self.MOVE_ARM_TO_HOVER: hover_arm,
            }
            return
        raise PlanningRefused("; ".join(refusals))


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
