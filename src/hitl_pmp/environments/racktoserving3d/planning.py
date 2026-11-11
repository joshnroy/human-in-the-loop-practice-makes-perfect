"""Planned free-space motions for the scraper skills, as the kitchen plans its jug's.

`RackPyBulletSim` is the kitchen's planning scene (`KitchenPyBulletSim`) over this
scene's bodies: the counter, the rack, the plate, the scraper and the dumplings, each
mirrored from MuJoCo's own contact geometry. Upstream's scene also builds a hard-coded
four-layer shelf for any object named "cupboard_1", a block for every "cube" and a box
for every compiled static collider; the mirrored bodies replace all three, so the
counter and the dumplings are each in the scene once, with their real shape.

`ScraperMotionPlanner` does all of a skill's planning before the robot moves: it tries
candidate stances in front of the scraper's home, nearest the configured base first,
and keeps the first from which a collision-free hand pose is reachable and a base path
and an arm path exist. A refusal (`PlanningRefused`) executes nothing. The contact
motions that follow (the end-on approach, the close, the seat and the release) stay
scripted, as the kitchen's do.
"""

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from spatialmath import SE2

from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
    HeldObject,
    KitchenMotionPlanner,
    PlanningRefused,
)
from hitl_pmp.environments.pouring3d.kitchen.planning.utils import (
    PLANNING_MARGIN_M,
    KitchenPyBulletSim,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

# The TidyBot chassis body, and the first arm link checked against it.
CHASSIS = "robot_base_link"
CHASSIS_CHECKED_FROM = "forearm_link"
# The carried scraper's boxes are grown by this in the planner's MuJoCo check, so
# tracking error under its payload cannot carry it into the execution guard's limit.
TOOL_INFLATE_M = 0.01
# Upstream's names for the counter's compiled collider boxes.
COUNTER_COLLIDERS = "collider:cupboard_1:"
# How far behind the grasp the base centre may stand, and how far to either side of it.
STANCE_BACK_M = (0.30, 0.80)
STANCE_SIDE_M = 0.15
STANCE_STEP_M = 0.05
STANCE_YAWS = (0.0, 0.25, -0.25, 0.5, -0.5)
# A planned seat keeps every joint at least this far from its MuJoCo limit, so the
# scripted seat can still lower the last few millimetres (the elbow at its -2.57 rad
# limit left a 0.9 m return 6 mm above the seat).
JOINT_LIMIT_MARGIN_RAD = 0.15
# The pick's approach, lift and tilt only need to be reachable.
PICK_JOINT_LIMIT_MARGIN_RAD = 0.03
# The deepest the scripted seat lowers below its nominal clearance.
SEAT_DEPTH_M = 0.008
RETURN_HOVER_M = 0.06
# How finely a scripted contact line is checked for one IK branch.
CHAIN_STEP_M = 0.04
CHAIN_STEP_DEG = 15.0
# Consecutive checked solutions further apart than this are on different IK branches.
# IKFast samples the Kinova's redundant joint, so even its closest solution to a
# neighbouring pose can sit about 0.4 rad away; a branch flip is over 2 rad.
BRANCH_JUMP_RAD = 1.2
# A planned base path is checked against the MuJoCo guard at most this far apart on
# every axis (m, rad), finer than the executor's own check (`RackScraper.step`: five
# configurations across one 15 mm / 0.015 rad base step).
BASE_PATH_CHECK = 0.001
# The body-check spacing each base plan is searched at, in turn, until its path passes
# that check: upstream's (None) first, as nearly every path does; a search at 1 mm
# costs about ten times one at upstream's 1 cm.
BASE_REPLAN_STEPS = (None, 0.003, BASE_PATH_CHECK)
# Base candidates tried per skill before refusing; each costs IK and possibly a BiRRT.
MAX_STANCES = 400


class RackPyBulletSim(KitchenPyBulletSim):
    """The kitchen's planning scene over RackToServing3D's bodies."""

    fixtures: ClassVar[tuple[str, ...]] = ("cupboard_1",)
    movables: ClassVar[tuple[str, ...]] = ("rack_0", "box_0", "wiper_0")
    movable_prefix: ClassVar[str] = "cube_"

    def __init__(self, initial_state: Any, *, model: Any, data: Any, base_q: Any) -> None:  # noqa: PLR0917 -- upstream signature
        super().__init__(initial_state, model=model, data=data, base_q=base_q)
        client = self.physics_client_id
        if self._cupboard1_shelf_id is not None:
            p.removeBody(self._cupboard1_shelf_id, physicsClientId=client)
            # Upstream's set_state poses this id from the state's cupboard_1.
            self._cupboard1_shelf_id = self._mirrored["cupboard_1"]
        for name, body in list(self._cubes.items()):
            p.removeBody(body, physicsClientId=client)
            self._cubes[name] = self._mirrored[name]
        for body in self._static_colliders.values():
            p.removeBody(body, physicsClientId=client)
        self._static_colliders = {}
        # Upstream plans the mobile base only as a 2-D footprint, but a low reach can
        # fold the hand back into the TidyBot's own chassis, which MuJoCo collides
        # with. It rides with the base and is checked against the distal arm only: the
        # arm's mount sits inside its hull.
        self._mirror_body(name=CHASSIS, model=model, data=data, subtree=False)
        self._chassis = self._mirrored.pop(CHASSIS)
        self._mirrored_counts.pop(CHASSIS)
        self._base_to_chassis = self.arm_mount(model=model, data=data, base_q=base_q, body=CHASSIS)
        robot = self.robot.robot_id
        names = [
            p.getJointInfo(robot, j, physicsClientId=client)[12].decode()
            for j in range(p.getNumJoints(robot, physicsClientId=client))
        ]
        self._chassis_links = tuple(range(names.index(CHASSIS_CHECKED_FROM), len(names)))
        self.place_chassis()
        # Every planning query is getClosestPoints, which ignores collision filters; the
        # contact pairs performCollisionDetection caches are never read. Unfiltered, each
        # check recomputed the scene's own resting contacts (the dumplings on the rack),
        # 2 ms of a 5.7 ms check. The call still refreshes the AABBs `near_bodies` reads.
        for index in range(p.getNumBodies(physicsClientId=client)):
            body = p.getBodyUniqueId(index, physicsClientId=client)
            for link in range(-1, p.getNumJoints(body, physicsClientId=client)):
                p.setCollisionFilterGroupMask(body, link, 0, 0, physicsClientId=client)

    def near_bodies(
        self, *, bodies: Iterable[int], held_object: int | None, margin: float
    ) -> list[int]:
        """Those of `bodies` whose AABB comes within `margin` of the AABB of the robot
        (and `held_object`) as posed now: the only ones getClosestPoints can find
        within `margin`, since every AABB contains its shape."""
        client = self.physics_client_id
        p.performCollisionDetection(physicsClientId=client)
        robot = self.robot.robot_id
        parts = [(robot, link) for link in range(-1, p.getNumJoints(robot, physicsClientId=client))]
        if held_object is not None:
            parts += [
                (held_object, link)
                for link in range(-1, p.getNumJoints(held_object, physicsClientId=client))
            ]
        boxes = np.array([p.getAABB(body, link, physicsClientId=client) for body, link in parts])
        lower, upper = boxes[:, 0].min(axis=0) - margin, boxes[:, 1].max(axis=0) + margin
        overlapping = p.getOverlappingObjects(
            lower.tolist(), upper.tolist(), physicsClientId=client
        )
        found = {body for body, _ in overlapping or ()}
        return [body for body in bodies if body in found]

    def robot_collides_at_base(  # noqa: PLR0917 -- upstream signature
        self,
        base_pose: SE2,
        joints: Any,
        held_object: int | None = None,
        collision_bodies: set[int] | None = None,
        distance_threshold: float = 1e-6,
    ) -> bool:
        """Upstream's check, over only the obstacles near the robot (`near_bodies`)."""
        from pybullet_helpers.inverse_kinematics import (
            check_collisions_with_held_object,
            set_robot_joints_with_held_object,
        )

        if collision_bodies is None:
            collision_bodies = self.get_collision_bodies(held_object=held_object)
        offset = self.base_link_to_held_obj if held_object is not None else None
        self.set_robot_base(base_pose)
        set_robot_joints_with_held_object(
            self.robot, self.physics_client_id, held_object, offset, list(joints)
        )
        near = self.near_bodies(
            bodies=collision_bodies, held_object=held_object, margin=distance_threshold
        )
        return check_collisions_with_held_object(
            self.robot,
            near,
            self.physics_client_id,
            held_object,
            offset,
            list(joints),
            distance_threshold=distance_threshold,
        )

    def place_chassis(self) -> None:
        from pybullet_helpers.geometry import multiply_poses, set_pose

        position, xyzw = p.getBasePositionAndOrientation(
            self.robot.robot_id, physicsClientId=self.physics_client_id
        )
        base = multiply_poses(Pose(tuple(position), tuple(xyzw)), self._base_to_arm_pose.invert())
        set_pose(self._chassis, multiply_poses(base, self._base_to_chassis), self.physics_client_id)

    def set_robot_base(self, base_pose: SE2) -> None:  # noqa: PLR0917 -- upstream signature
        super().set_robot_base(base_pose)
        self.place_chassis()

    def set_state(self, x: Any, held_object: Any = None) -> None:  # noqa: PLR0917 -- upstream signature
        super().set_state(x, held_object=held_object)
        self.place_chassis()

    def chassis_clear(self, *, margin: float, held: int | None = None) -> bool:
        """Whether the distal arm, the gripper and `held` keep `margin` from the
        chassis, as they are posed now."""
        client = self.physics_client_id
        for link in self._chassis_links:
            if p.getClosestPoints(
                self.robot.robot_id,
                self._chassis,
                distance=margin,
                linkIndexA=link,
                physicsClientId=client,
            ):
                return False
        return held is None or not p.getClosestPoints(
            held, self._chassis, distance=margin, physicsClientId=client
        )

    # Welded food handling plans against the fixtures alone (`WeldedFood`).
    ignores_food: bool = False

    def get_collision_bodies(self, held_object: int | None = None) -> set[int]:  # noqa: PLR0917 -- upstream signature
        bodies = super().get_collision_bodies(held_object=held_object)
        if self.ignores_food:
            bodies -= {
                body
                for name, body in self._mirrored.items()
                if name.startswith(self.movable_prefix)
            }
        return bodies

    @classmethod
    def from_shelf(cls, *, shelf: Any, ignores_food: bool = False) -> "RackPyBulletSim":
        state = shelf.core._get_object_centric_state()  # noqa: SLF001
        sim = cls(state, model=shelf.model, data=shelf.data, base_q=shelf.ba)
        sim.ignores_food = ignores_food
        sim.set_state(state)
        return sim


class ScraperPlan(BaseModel):
    """The planned motions of one skill, in execution order, and where they end."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stance: np.ndarray
    base_path: list[np.ndarray]
    arm_before_base: list[np.ndarray] = []
    arm_after_base: list[np.ndarray]


class ReachCandidate(BaseModel):
    """A stance and the hover the arm is planned to from it; `contact` says why the
    scripted contact motion from that hover (stance, hover joints) would fail, or None."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stance: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    contact: Callable[[np.ndarray, np.ndarray], str | None]
    payload: Any = None


class ScraperMotionPlanner:
    """A static-method container, never instantiated."""

    @staticmethod
    def stances(*, grasp: np.ndarray, preferred: np.ndarray) -> list[np.ndarray]:
        """Candidate bases facing the counter (+x) behind `grasp`, nearest `preferred`."""
        backs = np.arange(STANCE_BACK_M[0], STANCE_BACK_M[1] + 1e-9, STANCE_STEP_M)
        sides = np.arange(-STANCE_SIDE_M, STANCE_SIDE_M + 1e-9, STANCE_STEP_M)
        out = [
            np.array([grasp[0] - back, grasp[1] + side, yaw])
            for back in backs
            for side in sides
            for yaw in STANCE_YAWS
        ]
        weights = np.array([1.0, 1.0, 0.3])
        out.sort(key=lambda s: float(np.linalg.norm((s - preferred) * weights)))
        return out

    @staticmethod
    def within_limits(*, q: Any, lower: np.ndarray, upper: np.ndarray, margin: float) -> bool:
        arm = np.asarray(q[:7], dtype=float)
        return bool(np.all(arm >= lower + margin) and np.all(arm <= upper - margin))

    @staticmethod
    def contact_goal(
        *,
        sim: RackPyBulletSim,
        position: np.ndarray,
        rotation: np.ndarray,
        near: Any,
        ignore: tuple[str, ...],
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
    ) -> list[float] | None:
        """An IK solution for a contact pose (the hand on the handle, the scraper on its
        seat): the robot itself clear of everything but `ignore`, joints inside MuJoCo's
        limits by the margin, and on the branch of `near` (no wrist flip on the scripted
        straight line between them)."""
        named = sim.named_collision_bodies()
        bodies = sim.get_collision_bodies() - {named[n] for n in ignore if n in named}
        joints = KitchenMotionPlanner.with_fingers(sim=sim, arm=near)
        sim.robot.set_joints(joints)

        def invalid(q: Any) -> bool:  # noqa: PLR0917 -- KitchenMotionPlanner.goal callback
            if not ScraperMotionPlanner.within_limits(q=q, lower=lower, upper=upper, margin=margin):
                return True
            delta = (np.asarray(q[:7]) - np.asarray(near[:7]) + np.pi) % (2 * np.pi) - np.pi
            if float(np.max(np.abs(delta))) > BRANCH_JUMP_RAD:
                return True
            sim.robot.set_joints(list(q))
            for body in bodies:
                if p.getClosestPoints(
                    sim.robot.robot_id, body, distance=0.0, physicsClientId=sim.physics_client_id
                ):
                    return True
            return not sim.chassis_clear(margin=0.0)

        return KitchenMotionPlanner.goal(
            sim=sim,
            position=position,
            rotation=Rotation.from_matrix(rotation),
            valid=invalid,
        )

    @staticmethod
    def chain(
        *,
        sim: RackPyBulletSim,
        start: Any,
        start_pose: tuple[np.ndarray, np.ndarray],
        poses: list[tuple[np.ndarray, np.ndarray]],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> str | None:
        """Why the scripted straight lines through `poses` (from `start` at
        `start_pose`) have no contact IK solution on one branch, or None if they do.
        Each line is solved every `CHAIN_STEP_M` / `CHAIN_STEP_DEG`, as the scripted
        line follows it."""
        from scipy.spatial.transform import Slerp

        near = start
        here_position, here_rotation = start_pose
        for index, (position, rotation) in enumerate(poses):
            distance = float(np.linalg.norm(position - here_position))
            turn = float(np.rad2deg(Rotation.from_matrix(here_rotation.T @ rotation).magnitude()))
            count = max(1, int(np.ceil(max(distance / CHAIN_STEP_M, turn / CHAIN_STEP_DEG))))
            slerp = Slerp([0, 1], Rotation.from_matrix([here_rotation, rotation]))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                near = ScraperMotionPlanner.contact_goal(
                    sim=sim,
                    position=(1 - fraction) * here_position + fraction * position,
                    rotation=slerp(fraction).as_matrix(),
                    near=near,
                    ignore=("wiper_0",),
                    lower=lower,
                    upper=upper,
                    margin=PICK_JOINT_LIMIT_MARGIN_RAD,
                )
                if near is None:
                    return f"contact pose {index} at {fraction:.2f}"
            here_position, here_rotation = position, rotation
        return None

    @staticmethod
    def free_goal(
        *,
        sim: RackPyBulletSim,
        position: np.ndarray,
        rotation: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        guard: Callable[[Any], bool],
        held: HeldObject | None = None,
    ) -> list[float] | None:
        """A collision-free IK solution with the planning margin, inside MuJoCo's
        limits and passing `guard`, for a free-space hand pose."""
        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object

        def invalid(q: Any) -> bool:  # noqa: PLR0917 -- KitchenMotionPlanner.goal callback
            if not ScraperMotionPlanner.within_limits(q=q, lower=lower, upper=upper, margin=0.02):
                return True
            if check_collisions_with_held_object(
                sim.robot,
                bodies,
                sim.physics_client_id,
                None if held is None else held.body,
                None if held is None else held.base_link_to_held_obj,
                list(q),
                distance_threshold=PLANNING_MARGIN_M,
            ):
                return True
            if not sim.chassis_clear(
                margin=PLANNING_MARGIN_M, held=None if held is None else held.body
            ):
                return True
            return not guard(q)

        return KitchenMotionPlanner.goal(
            sim=sim, position=position, rotation=Rotation.from_matrix(rotation), valid=invalid
        )

    @staticmethod
    def plan_base(
        *,
        sim: RackPyBulletSim,
        state: Any,
        base: np.ndarray,
        target: np.ndarray,
        arm: Any,
        what: str,
        guard: Callable[[np.ndarray], bool],
        held: HeldObject | None = None,
        ignore: tuple[str, ...] | None = None,
    ) -> list[np.ndarray]:
        """A base path from `base` to `target` on which `guard` holds at every
        configuration at most `BASE_PATH_CHECK` apart, or PlanningRefused.

        Upstream checks the 3-D body 1 cm / 0.01 rad apart, and a chassis corner
        sweeping past a thin counter leg can collide for less than that between two
        clear checks; the executor, which checks every few millimetres, then stops
        mid-path, and a replan from there plans the same grazing segment again. So the
        path is checked at `BASE_PATH_CHECK`, and if any configuration fails, planned
        again with the body checked that densely throughout.
        """
        if np.linalg.norm(target[:2] - base[:2]) < 0.005 and abs(target[2] - base[2]) < 0.005:
            return [base.copy(), target.copy()]
        ignored = (
            *((() if held is None else ("wiper_0",)) if ignore is None else ignore),
            *ScraperMotionPlanner.counter_colliders(state=state),
        )
        out: list[np.ndarray] = []
        for step in BASE_REPLAN_STEPS:
            path = KitchenMotionPlanner.plan_base(
                sim=sim,
                state=state,
                target=target,
                arm=arm,
                what=what,
                held=held,
                ignore=ignored,
                also_collides=lambda pose: not guard(np.array([pose.x, pose.y, pose.theta()])),
                **(
                    {}
                    if step is None
                    else {"body_check_xy_step": step, "body_check_rot_step": step}
                ),
            )
            # Unwrap the heading so the base turns the short way between waypoints.
            out = [base.copy()]
            for pose in path[1:]:
                pose = pose.copy()
                pose[2] = out[-1][2] + (pose[2] - out[-1][2] + np.pi) % (2 * np.pi) - np.pi
                out.append(pose)
            if ScraperMotionPlanner.base_path_clear(path=out, guard=guard):
                return out
        raise PlanningRefused(f"No base path to {what} clear at {BASE_PATH_CHECK} m spacing")

    @staticmethod
    def base_path_clear(*, path: list[np.ndarray], guard: Callable[[np.ndarray], bool]) -> bool:
        """Whether `guard` holds along each straight segment of `path`, at most
        `BASE_PATH_CHECK` apart on every axis, as `follow_base_path` drives it."""
        for begin, end in zip(path[:-1], path[1:], strict=True):
            count = max(1, int(np.ceil(np.max(np.abs(end - begin)) / BASE_PATH_CHECK)))
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
                if not guard(begin + fraction * (end - begin)):
                    return False
        return True

    @staticmethod
    def counter_colliders(*, state: Any) -> tuple[str, ...]:
        """The counter's compiled collider boxes, which get no 2-D floor footprint.

        Upstream projects each box onto the floor, the counter top included, so a
        chassis that stands under the top's overhang (as one can: the top is 0.9 m up)
        "collides" in 2-D and no base path leaves it. The chassis is checked against
        the counter in 3-D at every base-path check instead (the MuJoCo guard: the
        `robot_base_link` geoms against the top and the legs).
        """
        return tuple(
            name for name in state.get_object_names() if name.startswith(COUNTER_COLLIDERS)
        )

    @staticmethod
    def plan_arm_to(
        *,
        sim: RackPyBulletSim,
        start: Any,
        goal: list[float],
        what: str,
        guard: Callable[[Any], bool],
        held: HeldObject | None = None,
    ) -> list[np.ndarray]:
        from kinder_models.dynamic3d.tossing.parameterized_skills import plan_checked_arm_motion

        bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
        planned = plan_checked_arm_motion(
            sim.robot,
            KitchenMotionPlanner.with_fingers(sim=sim, arm=start),
            goal,
            collision_bodies=bodies,
            physics_client_id=sim.physics_client_id,
            held_object=None if held is None else held.body,
            base_link_to_held_obj=None if held is None else held.base_link_to_held_obj,
            distance_threshold=PLANNING_MARGIN_M,
            # Called right after the collision check has posed the robot (and `held`).
            additional_state_constraint_fn=lambda q: (
                sim.chassis_clear(
                    margin=PLANNING_MARGIN_M, held=None if held is None else held.body
                )
                and guard(q)
            ),
        )
        if planned is None:
            raise PlanningRefused(f"No collision-free arm path to {what}")
        return [np.asarray(q[:7], dtype=float) for q in planned[1]]

    @staticmethod
    def plan_pick(
        *,
        tool: Any,
        grasp: np.ndarray,
        pre: np.ndarray,
        rotation: np.ndarray,
        preferred: np.ndarray,
        after: list[tuple[np.ndarray, np.ndarray]],
    ) -> ScraperPlan:
        """Base and arm to the pre-grasp hover, from the first stance that reaches the
        hover, the grasp and each scripted hand pose `after` it (lift, tilt) on one IK
        branch; PlanningRefused if none does."""
        shelf = tool.shelf
        lower, upper = ScraperMotionPlanner.limits(shelf=shelf)
        arm = shelf.data.qpos[shelf.qa].copy()
        base = shelf.data.qpos[shelf.ba].copy()
        sim = RackPyBulletSim.from_shelf(shelf=shelf)
        state = shelf.core._get_object_centric_state()  # noqa: SLF001
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=False)
        failures: list[str] = []
        try:
            for stance in ScraperMotionPlanner.stances(grasp=grasp, preferred=preferred)[
                :MAX_STANCES
            ]:
                sim.set_robot_base(SE2(*(float(v) for v in stance)))
                hover = ScraperMotionPlanner.free_goal(
                    sim=sim,
                    position=pre,
                    rotation=rotation,
                    lower=lower,
                    upper=upper,
                    guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                )
                if hover is None:
                    failures.append("hover")
                    continue
                why = ScraperMotionPlanner.chain(
                    sim=sim,
                    start=hover,
                    start_pose=(pre, rotation),
                    poses=[(grasp, rotation), *after],
                    lower=lower,
                    upper=upper,
                )
                if why is not None:
                    failures.append(why)
                    continue
                try:
                    base_path = ScraperMotionPlanner.plan_base(
                        sim=sim,
                        state=state,
                        base=base,
                        target=stance,
                        arm=arm,
                        what="the scraper pick stance",
                        guard=ScraperMotionPlanner.base_guard(ok=ok, arm=arm),
                    )
                    sim.set_robot_base(SE2(*(float(v) for v in stance)))
                    arm_path = ScraperMotionPlanner.plan_arm_to(
                        sim=sim,
                        start=arm,
                        goal=hover,
                        what="the scraper pre-grasp",
                        guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                    )
                except PlanningRefused as exc:
                    failures.append(str(exc))
                    continue
                return ScraperPlan(stance=stance, base_path=base_path, arm_after_base=arm_path)
        finally:
            sim.close()
        raise PlanningRefused(
            "PickScraper: no stance reaches the scraper "
            f"({ScraperMotionPlanner.tally(failures=failures)})"
        )

    @staticmethod
    def plan_return(
        *,
        tool: Any,
        seat: np.ndarray,
        rotation: np.ndarray,
        preferred: np.ndarray,
        carry_height: float,
    ) -> tuple[ScraperPlan, np.ndarray]:
        """Arm tucked if needed, base to a stance, arm to a hover above the seat, the
        scraper carried rigidly and checked throughout; the hover's hand position is
        returned with the plan."""
        shelf = tool.shelf
        lower, upper = ScraperMotionPlanner.limits(shelf=shelf)
        arm = shelf.data.qpos[shelf.qa].copy()
        base = shelf.data.qpos[shelf.ba].copy()
        sim = RackPyBulletSim.from_shelf(shelf=shelf)
        state = shelf.core._get_object_centric_state()  # noqa: SLF001
        hover_position = seat + [0.0, 0.0, RETURN_HOVER_M]
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=True)
        failures: list[str] = []
        try:
            sim.set_robot_base(SE2(*(float(v) for v in base)))
            held = HeldObject.grasped(sim=sim, arm=arm, name="wiper_0", riders=())
            hand = shelf.site_pose()
            carries: list[list[np.ndarray]] = [[arm.copy()]]
            tuck_position = np.array([hand[0, 3], hand[1, 3], max(hand[2, 3], carry_height)])
            tuck = ScraperMotionPlanner.free_goal(
                sim=sim,
                position=tuck_position,
                rotation=hand[:3, :3],
                lower=lower,
                upper=upper,
                guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
                held=held,
            )
            if tuck is not None:
                try:
                    carries.append(
                        ScraperMotionPlanner.plan_arm_to(
                            sim=sim,
                            start=arm,
                            goal=tuck,
                            what="the carry pose",
                            guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
                            held=held,
                        )
                    )
                except PlanningRefused as exc:
                    failures.append(str(exc))
            for stance in ScraperMotionPlanner.stances(grasp=seat, preferred=preferred)[
                :MAX_STANCES
            ]:
                sim.set_robot_base(SE2(*(float(v) for v in stance)))
                hover = ScraperMotionPlanner.free_goal(
                    sim=sim,
                    position=hover_position,
                    rotation=rotation,
                    lower=lower,
                    upper=upper,
                    guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                    held=held,
                )
                if hover is None:
                    failures.append("hover")
                    continue
                if any(
                    ScraperMotionPlanner.contact_goal(
                        sim=sim,
                        position=seat - [0.0, 0.0, depth],
                        rotation=rotation,
                        near=hover,
                        ignore=("wiper_0",),
                        lower=lower,
                        upper=upper,
                        margin=JOINT_LIMIT_MARGIN_RAD,
                    )
                    is None
                    for depth in (0.0, SEAT_DEPTH_M)
                ):
                    failures.append("seat")
                    continue
                for carry in carries:
                    try:
                        sim.set_robot_base(SE2(*(float(v) for v in base)))
                        base_path = ScraperMotionPlanner.plan_base(
                            sim=sim,
                            state=state,
                            base=base,
                            target=stance,
                            arm=carry[-1],
                            what="the scraper return stance",
                            guard=ScraperMotionPlanner.base_guard(ok=ok, arm=carry[-1]),
                            held=held,
                        )
                        sim.set_robot_base(SE2(*(float(v) for v in stance)))
                        arm_path = ScraperMotionPlanner.plan_arm_to(
                            sim=sim,
                            start=carry[-1],
                            goal=hover,
                            what="the hover above the seat",
                            guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                            held=held,
                        )
                    except PlanningRefused as exc:
                        failures.append(str(exc))
                        continue
                    plan = ScraperPlan(
                        stance=stance,
                        base_path=base_path,
                        arm_before_base=carry,
                        arm_after_base=arm_path,
                    )
                    return plan, hover_position
        finally:
            sim.close()
        raise PlanningRefused(
            "ReturnScraper: no stance reaches the seat "
            f"({ScraperMotionPlanner.tally(failures=failures)})"
        )

    @staticmethod
    def plan_reach(
        *,
        tool: Any,
        candidates: Iterable[ReachCandidate],
        held_name: str | None,
        carry_height: float,
        deadline_s: float,
        what: str,
        tuck_first: bool = False,
    ) -> tuple[ScraperPlan, ReachCandidate]:
        """The first candidate stance whose hover is collision-free, whose scripted
        contact motion its own `contact` check accepts from that hover, and to which a
        base path and an arm path exist; PlanningRefused if none does in time.

        Before the base moves the arm keeps its configuration, or (if the base cannot
        move with it) tucks: holding, up to `carry_height`; empty, to the home arm.
        `tuck_first` tries the tuck first: a hand left low over the counter can graze
        it from tracking error while the base drives under a path checked clear.
        `held_name`, when given, is carried rigidly (HeldObject) and has no footprint.
        """
        from time import monotonic

        shelf = tool.shelf
        lower, upper = ScraperMotionPlanner.limits(shelf=shelf)
        arm = shelf.data.qpos[shelf.qa].copy()
        base = shelf.data.qpos[shelf.ba].copy()
        ignores_food = bool(getattr(tool, "ignores_food", False))
        sim = RackPyBulletSim.from_shelf(shelf=shelf, ignores_food=ignores_food)
        state = shelf.core._get_object_centric_state()  # noqa: SLF001
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=False)
        deadline = monotonic() + deadline_s
        failures: list[str] = []
        try:
            sim.set_robot_base(SE2(*(float(v) for v in base)))
            held = (
                None
                if held_name is None
                else HeldObject.grasped(sim=sim, arm=arm, name=held_name, riders=())
            )
            ignore: tuple[str, ...] = () if held_name is None else (held_name,)
            if ignores_food:
                ignore = tuple(
                    name
                    for name in state.get_object_names()
                    if name.startswith(RackPyBulletSim.movable_prefix)
                )
            carries: list[list[np.ndarray]] = [[arm.copy()]]
            if held is None:
                tuck: list[float] | None = list(np.asarray(shelf.home_arm, dtype=float))
            else:
                hand = shelf.site_pose()
                tuck = ScraperMotionPlanner.free_goal(
                    sim=sim,
                    position=np.array([hand[0, 3], hand[1, 3], max(hand[2, 3], carry_height)]),
                    rotation=hand[:3, :3],
                    lower=lower,
                    upper=upper,
                    guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
                    held=held,
                )
            if tuck is not None:
                try:
                    carries.append(
                        ScraperMotionPlanner.plan_arm_to(
                            sim=sim,
                            start=arm,
                            goal=KitchenMotionPlanner.with_fingers(sim=sim, arm=tuck),
                            what="the carry pose",
                            guard=ScraperMotionPlanner.arm_guard(ok=ok, base=base),
                            held=held,
                        )
                    )
                except PlanningRefused as exc:
                    failures.append(str(exc))
            if tuck_first:
                carries.reverse()
            for candidate in candidates:
                if monotonic() >= deadline:
                    failures.append("deadline")
                    break
                stance = candidate.stance
                if not ok(base=stance, arm=shelf.home_arm):
                    failures.append("stance blocked")
                    continue
                sim.set_robot_base(SE2(*(float(v) for v in stance)))
                hover = ScraperMotionPlanner.free_goal(
                    sim=sim,
                    position=candidate.position,
                    rotation=candidate.rotation,
                    lower=lower,
                    upper=upper,
                    guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                    held=held,
                )
                if hover is None:
                    failures.append("hover")
                    continue
                why = candidate.contact(stance, np.asarray(hover[:7], dtype=float))
                if why is not None:
                    failures.append(why)
                    continue
                for carry in carries:
                    try:
                        sim.set_robot_base(SE2(*(float(v) for v in base)))
                        base_path = ScraperMotionPlanner.plan_base(
                            sim=sim,
                            state=state,
                            base=base,
                            target=stance,
                            arm=carry[-1],
                            what=what,
                            guard=ScraperMotionPlanner.base_guard(ok=ok, arm=carry[-1]),
                            held=held,
                            ignore=ignore,
                        )
                        sim.set_robot_base(SE2(*(float(v) for v in stance)))
                        arm_path = ScraperMotionPlanner.plan_arm_to(
                            sim=sim,
                            start=carry[-1],
                            goal=hover,
                            what=what,
                            guard=ScraperMotionPlanner.arm_guard(ok=ok, base=stance),
                            held=held,
                        )
                    except PlanningRefused as exc:
                        failures.append(str(exc))
                        continue
                    plan = ScraperPlan(
                        stance=stance,
                        base_path=base_path,
                        arm_before_base=carry,
                        arm_after_base=arm_path,
                    )
                    return plan, candidate
        finally:
            sim.close()
        raise PlanningRefused(
            f"No planned reach to {what} ({ScraperMotionPlanner.tally(failures=failures)})"
        )

    @staticmethod
    def start_blocked(*, tool: Any, held_name: str | None) -> bool:
        """Whether the live configuration itself fails the planner's own checks (the
        margin in PyBullet, the MuJoCo guard), so that no plan can start from it."""
        from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object

        shelf = tool.shelf
        arm = shelf.data.qpos[shelf.qa].copy()
        base = shelf.data.qpos[shelf.ba].copy()
        ok = ScraperMotionPlanner.mujoco_guard(tool=tool, holding=held_name is not None)
        if not ok(base=base, arm=arm):
            return True
        sim = RackPyBulletSim.from_shelf(
            shelf=shelf, ignores_food=bool(getattr(tool, "ignores_food", False))
        )
        try:
            sim.set_robot_base(SE2(*(float(v) for v in base)))
            held = (
                None
                if held_name is None
                else HeldObject.grasped(sim=sim, arm=arm, name=held_name, riders=())
            )
            bodies = sim.get_collision_bodies() if held is None else held.obstacles(sim=sim)
            if check_collisions_with_held_object(
                sim.robot,
                bodies,
                sim.physics_client_id,
                None if held is None else held.body,
                None if held is None else held.base_link_to_held_obj,
                KitchenMotionPlanner.with_fingers(sim=sim, arm=arm),
                distance_threshold=PLANNING_MARGIN_M,
            ):
                return True
            return not sim.chassis_clear(
                margin=PLANNING_MARGIN_M, held=None if held is None else held.body
            )
        finally:
            sim.close()

    @staticmethod
    def mujoco_guard(*, tool: Any, holding: bool) -> Callable[..., bool]:
        """The execution guard itself (`RackScraper.check_configuration`, in MuJoCo
        scratch data) as a planning constraint. It checks the carried scraper's boxes
        more conservatively than PyBullet's mirror of the plate, so a path PyBullet
        clears with the margin can still be refused mid-motion without it."""

        def ok(*, base: np.ndarray, arm: Any) -> bool:
            try:
                tool.check_configuration(
                    base=np.asarray(base, dtype=float),
                    arm=np.asarray(arm[:7], dtype=float),
                    holding=holding,
                    inflate=TOOL_INFLATE_M,
                )
            except SkillFailed:
                return False
            return True

        return ok

    @staticmethod
    def arm_guard(*, ok: Callable[..., bool], base: np.ndarray) -> Callable[[Any], bool]:
        return lambda q: ok(base=base, arm=q)

    @staticmethod
    def base_guard(*, ok: Callable[..., bool], arm: Any) -> Callable[[np.ndarray], bool]:
        return lambda pose: ok(base=pose, arm=arm)

    @staticmethod
    def limits(*, shelf: Any) -> tuple[np.ndarray, np.ndarray]:
        """MuJoCo's arm joint limits; an unlimited joint gets +-2 pi."""
        model = shelf.model
        lower = np.full(7, -2 * np.pi)
        upper = np.full(7, 2 * np.pi)
        for index, joint in enumerate(shelf.joints):
            if model.jnt_limited[joint]:
                lower[index], upper[index] = model.jnt_range[joint]
        return lower, upper

    @staticmethod
    def tally(*, failures: list[str]) -> str:
        counts: dict[str, int] = {}
        for failure in failures:
            counts[failure] = counts.get(failure, 0) + 1
        return ", ".join(f"{reason}: {count}" for reason, count in counts.items()) or "none tried"
