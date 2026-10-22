"""SweepOffShelf3D's robot skills on KINDER's own controllers: seven, behind eight operators.

Each skill runs to completion or raises `SkillFailed`, and every one, failed or not,
ends with the arm retracted: to KINDER's tucked home configuration (`HOME_ARM`), or,
holding the wiper, straight up to the carry height. So no skill starts where the last
one gave up -- the absorbing failure behind CCCC's PickWiper tally (`KinderSkills.run`).

Where each controller comes from:

- `pick_cube_from_floor`, and the pick in `return_cube_to_shelf`: `floor_pick.py`'s
  grasp x base candidate search over `shelf.PickShelfController`'s planners.
- `pick_wiper`, `pick_cube_from_shelf`, `place_cube_in_box`, `return_cube_to_shelf`:
  `sweep3D.PickWiperOriController`, `shelf.PickShelfController` and
  `shelf.PlaceShelfController`. Their planners -- base BiRRT, PyBullet inverse
  kinematics, arm BiRRT to the goal and back to home -- run as upstream wrote them;
  what is adapted, each for a measured reason given where it is defined:
  1. execution follows the joint plans the controller computes (`FollowPlans`), where
     upstream drives a straight joint-space line that ignores them;
  2. grasps in the cupboard are this package's leaned hand poses
     (`ShelfSkills.wiper_grasp`, `ShelfSkills.shelf_cube_pose`), since the slots are
     too short for upstream's vertical grasp;
  3. placements target this scene's box and cube shelf, through the controllers' own
     module constant `ARM_MOVEMENT_CUPBOARD` and base parameters;
  4. picks in the cupboard plan to a pre-grasp and descend the last few centimetres on
     MuJoCo's kinematics (`PreGraspApproach`, `KinderSkills.settler`);
  5. every plan is re-checked in the MuJoCo scene before it runs
     (`KinderSkills.plan_blocked`): PyBullet's cupboard and arm are not MuJoCo's;
  6. the shelf pick's held-cube retract allows upstream's 1 mm resting contact
     (`HELD_ON_SUPPORT_THRESHOLD`), and the box and the resting wiper are in every
     planning scene (`planning_sim`) -- upstream builds every cube but neither;
  7. the shelf pick is closed loop over its own candidates (`pick_cube_from_shelf`):
     grasps on the cube as it lies now, the pads on opposite faces (`shelf_hands`),
     each solved and checked in MuJoCo before KINDER plans to
     it (`shelf_candidates`), none whose pads would sweep another cube on the way in
     or as they close (`PadSweep`) -- refused, not shoved -- carried out of the
     cupboard's front before the retract plan (`lifter`), and on a miss, looked at
     again and replanned;
  8. the base planner leaves out of its obstacles every object the robot carries --
     held, or resting on its chassis (`robot_borne`) -- where upstream leaves out only
     the one object a controller places; and a pick that closes on two cubes is a miss.
- `sweep_shelf`, `return_wiper`: KINDER has no controller for either (its
  `SweepOriController` sweeps a floor-standing target, 0/15 here), so they are this
  package's, built on `ShelfSkills`' routines, starting from the wiper as it is actually
  held, whichever skill picked it up.

Nothing here imports KINDER at module scope: `kinder_models` is imported on first use.
"""

import contextlib
import itertools
from collections.abc import Callable, Generator
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from .backend import CUBE_REGION
from .objects import SweepOffShelf3DObjects as O
from .scripted_skills import (
    CARRY_Z,
    CUBE_LEANS_DEG,
    CUBE_SHELF,
    CUBE_STANDOFF,
    WIPER_SHELF,
    Frames,
    ShelfSkills,
    SkillFailed,
    StepBudgetExhausted,
)
from .types import SkillNote

CUPBOARD = "cupboard_1"
BOX = "box_0"
WIPER = "wiper_0"
# KINDER's tucked configuration, the one every upstream controller retracts to.
HOME_ARM = np.deg2rad([0.0, -20.0, 180.0, -146.0, 0.0, -50.0, 90.0])
# Control steps the gripper is given to close or open before the arm moves on.
GRIP_STEPS = 20
# Waypoint tracking for plan following (rad): pass-through band and final tolerance.
# The approach must end tighter than the rest -- upstream's ARM_FINAL_CONF_TOLERANCE
# (tossing), since 0.02 rad at full reach leaves the pads a centimetre off a 2 cm cube.
PASS_TOLERANCE = 0.04
FINAL_TOLERANCE = 0.02
APPROACH_FINAL_TOLERANCE = 5e-3
FINAL_STEP_LIMIT = 600
BASE_STEP_LIMIT = 600
FOLLOW_STEP_LIMIT = 60
RETRACT_TOLERANCE = 0.05
# The box's collision model in the planner: KINDER's own Bin primitive dimensions.
BOX_DIMENSIONS = {"length": 0.25, "width": 0.5, "height": 0.2, "wall_thickness": 0.01}
# Where the held cube is let go over the shelf, above the board: a short drop, not a
# set-down, since the leaned wrist cannot reach lower without meeting the board above.
SHELF_DROP_RISES = (0.06, 0.09, 0.12)
# Where on the cube shelf a returned cube is let go: a point on a grid this fine (m),
# inside the drop region, as far as the grid allows from every cube already on the shelf
# (the region's centre first, on an empty shelf). Let go at one fixed point, a returned
# cube landed on any cube already there. The region is KINDER's own `cubes_init_region`,
# where every shelf pick starts (x 1.43-1.57, y +-0.24 in this scene), inset by these
# margins (m) from its near (-x) and far edges and its sides to the points measured: on
# an empty shelf, a cube returned from the floor to each of 28 targets over x 1.45-1.54,
# y +-0.21 landed on the board within 8 mm of it and was picked off it again, 28/28.
SHELF_DROP_GRID = 0.005
SHELF_DROP_X_MARGINS = (0.02, 0.03)
SHELF_DROP_Y_MARGIN = 0.03
# No point nearer than this (m) to a cube on the shelf: two cubes' half-diagonals
# (28 mm), upstream's 1 cm grasp clearance and the 8 mm a dropped cube lands from its
# target. With no such point, the return refuses rather than stack.
SHELF_DROP_CLEARANCE = 0.046
# Drop points tried, in turn, if the best does not plan, each this far (m) from the others.
SHELF_DROP_TARGETS = 3
SHELF_DROP_TARGET_SPACING = 0.03
BOX_RELEASE_Z = 0.32
# Where in the box a cube is let go: a point on a grid this fine (m), inside the drop
# region, as far as the grid allows from every cube already in the box (the region's
# centre first, in an empty box). Dropped all at the box's centre, five cubes ended in
# one touching row with none pickable. The region is where a lone cube in the box can be
# picked again (>= 1 `FloorPicks` candidate, which keeps upstream's 1 cm clearance):
# 31/66 points of a 6 x 11 grid over the interior, all at least 10 cm from the near (-x)
# wall, 3 cm from the far one and 11 cm from the side walls; each margin here adds 1 cm
# for where a dropped cube lands (within 3 cm of its target, measured).
BOX_DROP_GRID = 0.005
BOX_DROP_X_MARGINS = (0.11, 0.04)
BOX_DROP_Y_MARGIN = 0.12
# Drop points tried, in turn, if the best does not plan, each this far (m) from the others.
BOX_DROP_TARGETS = 3
BOX_DROP_TARGET_SPACING = 0.03
PLACE_REACHES = (0.8, 0.7, 0.6)
PLACE_Y_OFFSETS = (0.0, 0.1, -0.1)
PICK_WIPER_DISTANCES = (0.7, 0.6, 0.8, 0.55)
# World headings from the base to the wiper. Square on first, then a little either side,
# which swings the base along the cupboard front: from an arm left over the box (seed 3)
# no square-on candidate planned an approach that clears board 3.
PICK_WIPER_HEADINGS = (0.0, 0.15, -0.15, 0.3, -0.3, np.pi / 2, -np.pi / 2, np.pi)
# Planned picks of the wiper in its slot, each re-observed and replanned, before giving up.
SLOT_PICK_ATTEMPTS = 3
# Taking the wiper out of its slot: lift off the board, and how far beyond the cupboard's
# open front its far end ends up.
SLOT_LIFT = 0.02
SLOT_EXIT_CLEARANCE = 0.05
# Base x positions for a shelf pick: as far back as the box lets it, and nearer.
SHELF_PICK_BASE_XS = (0.8, 0.72, 0.64, 0.56)
# Base y offsets from the cube, nearest first. The widest are for when something on the
# floor blocks the nearer ones: a wiper dropped across the front of the box left only
# bases 0.4-0.5 m to the side for upstream's base planner, which drives around it.
SHELF_PICK_Y_OFFSETS = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5)
# A second pre-grasp: this far (m) back along the approach, from where the hand
# goes in along its own axis, where the way to a nearer one is too narrow to plan.
SHELF_PICK_FAR_BACK = 0.08
# Where a picked cube is carried before the retract plan takes over: this far (m)
# outside the cupboard's open front, and this much higher than the pre-grasp.
SHELF_PICK_EXIT = 0.06
SHELF_PICK_EXIT_RISE = 0.02
# Penetration (m) KINDER's PyBullet planner allows the shelf pick's approach plan.
SHELF_PICK_PLANNING_SLACK = 0.008
# Joint-error gain for carrying a picked cube out along the retract plan, half the
# followers' 2.0: at full speed, cubes held across a corner slipped out of the pads
# on the way to the tucked arm, and fell into the box or onto the floor.
SHELF_PICK_CARRY_GAIN = 1.0
# BiRRT seeds tried for each shelf-pick plan, upstream's own (0) first.
SHELF_PICK_PLAN_SEEDS = (0, 1, 2)
# KINDER plans tried for one grasp, over its bases and pre-grasps, before the next grasp.
SHELF_PICK_PLANS_PER_GRASP = 6
# Grasp executions per shelf pick: a miss is seen (the cube is not between the pads),
# and the pick looks at the cube again and replans.
SHELF_PICK_EXECUTIONS = 3
# KINDER plans tried per execution, over candidates already clear in MuJoCo.
SHELF_PICK_PLANS = 24
# A shelf grasp's wrist must lean out of the cupboard: the approach's horizontal
# direction at most ~78 deg from +x, toward the open front.
SHELF_PICK_MIN_OUTWARD = 0.2
# Last of all, grasps whose wrist leans sideways along the shelf rather than out of
# it: the only ones whose pads close parallel to a side panel, for a cube against it.
SHELF_PICK_MIN_SIDEWAYS = -0.35
# Approaches whose directions agree this closely (cosine) count as the same grasp.
SHELF_PICK_SAME_APPROACH = float(np.cos(np.deg2rad(8.0)))
# Bases in a row from which a grasp is not clear before the grasp is given up.
SHELF_PICK_UNREACHABLE_BASES = 3
# A shelf-pick candidate: (grasp, entry poses, base, grasp arm, arm at the entry's first).
ShelfCandidate = tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]
# The space the pads sweep as they close, in the pinch site's frame (m): across the
# closing axis, and along the approach (the pads ride 11-24 mm ahead of the site as they
# close, 11 mm long), with a few millimetres' margin.
JAW_HALF_WIDTH = 0.016
JAW_DEPTH = (0.0, 0.05)
# The space each pad passes through on a shelf pick (`PadSweep`) is kept this far (m)
# from every other cube. Struck from 3-3.5 cm away, centre to centre, neighbours were
# shoved 2-7 cm or carried off in the jaws beside the target (2026-10-22). Over the
# skill benchmark's 100 shelf picks, a cube has no reachable grasp this clear in 2 of
# the 74 that `main` picked leaving every neighbour within 5 mm of where it was, and
# in 12 of the 23 it picked by moving one further; at 5 mm, 6/74 and 13/23, and at
# 35 mm, 34/74 and 19/23.
SHELF_PICK_NEIGHBOUR_CLEARANCE = 0.002
# The 2F-85's linkage joints fully closed (rad), read off the live sim; each side's
# `robot_<side>_<joint>`, coupler at 0 throughout. The pads' travel is posed at
# fractions `PAD_TRAVEL_STEP` apart of these.
GRIPPER_CLOSED = {"driver_joint": 0.787, "spring_link_joint": 0.785, "follower_joint": -0.766}
PAD_TRAVEL_STEP = 0.05
# The descent to a grasp is checked at hand poses at most this far (m) apart.
SWEEP_STEP = 0.005
# An oriented box: (centre, rotation, half-extents).
OrientedBox = tuple[np.ndarray, np.ndarray, np.ndarray]
# A cupboard geom whose top is no more than this (m) above the cube's bottom face is the
# board the cube rests on, not something the closing pads could clamp.
SUPPORT_SLACK = 0.002
# Largest joint change (rad) from the grasp to the pre-grasp: more is another IK branch.
SHELF_PICK_MAX_JUMP = 0.5
# A cube resting on a board starts the held-object retract plan touching it, which the
# planner's default threshold (1e-6 m) counts as a collision, so no retract ever plans.
# Upstream's own allowance for a cube resting on its support: tossing.PickCubeController
# .BIN_HELD_OBJECT_COLLISION_THRESHOLD, 1 mm of penetration in planning only.
HELD_ON_SUPPORT_THRESHOLD = -0.001
# KINDER plans in PyBullet, whose cupboard boards sit half a board low and whose arm
# links are not MuJoCo's, so a plan clear there can drive the wrist into a board here.
# Each plan is re-checked in the MuJoCo scene, every CHECK_STEP rad along it, and
# refused if any arm link would penetrate the cupboard or box.
CHECK_STEP = 0.05
CARRIED_ALLOWANCE = 0.004
HAND_ALLOWANCE = 0.001
PRE_GRASP_RISE = 0.03
# How far the hand backs out toward the base when the retract cannot plan from where it is.
BACK_OUT = 0.15
# An object this close (m) to any robot geom rests on or against the robot.
ROBOT_TOUCH = 0.001
# The upstream modules whose controllers plan the base with `run_base_motion_planning`.
BASE_PLANNER_MODULES = (
    "kinder_models.dynamic3d.shelf.parameterized_skills",
    "kinder_models.dynamic3d.sweep3D.parameterized_skills",
)
# The wiper this close (m) to any robot geom rests on or against the robot.
WIPER_ROBOT_TOUCH = 0.001
# The resting wiper's boxes grow by this much on every side in the planning scene.
WIPER_OBSTACLE_MARGIN = 0.01


class FollowPlans:
    """Mixin over an upstream controller with an approach and a retract plan: drive
    the arm through those plans' own waypoints instead of a straight joint-space line.

    The upstream controllers this wraps keep `_current_arm_joint_plan` (to the grasp or
    place configuration) and `_current_retract_plan` (from there to home), both from
    BiRRT, and then never execute them: `step` follows a trapezoidal profile on the
    straight line from start to goal, which goes through the cupboard. Navigation is
    left to upstream's `step`."""

    _followers: dict[str, "PlanFollower"]
    _base_steps: int
    _grip_wait: int
    # Set by the caller: run once, between the end of the approach and the gripper.
    settle: Callable[[], None] | None = None
    # Set by the caller: end a pick once the gripper has closed, leaving the hand where
    # it is, so the caller can check the grasp and take the object out itself.
    stop_after_grip: bool = False
    # Set by the caller: run once, between the gripper and the retract plan.
    lift: Callable[[], None] | None = None
    # Set by the caller: the joint-error gain the retract plan is followed with.
    retract_gain: float = 2.0

    def _follow(
        self, *, key: str, plan: list[Any], final: float = FINAL_TOLERANCE
    ) -> np.ndarray | None:
        follower = self._followers.get(key)
        if follower is None:
            follower = PlanFollower(
                plan=[np.array(q[:7], dtype=float) for q in plan],
                final=final,
                gain=self.retract_gain if key == "retract" else 2.0,
            )
            self._followers[key] = follower
        current = np.array(self._get_current_robot_arm_conf()[:7])  # type: ignore[attr-defined]
        grip = self._get_current_robot_gripper_pose()  # type: ignore[attr-defined]
        return follower.action(current=current, grip=grip)

    @staticmethod
    def _hold(*, grip: float) -> np.ndarray:
        action = np.zeros(11, dtype=np.float32)
        action[-1] = grip
        return action

    def step(self) -> Any:
        if not hasattr(self, "_followers"):
            self._followers, self._base_steps, self._grip_wait = {}, 0, 0
        if not self._navigated:  # type: ignore[attr-defined]
            # Upstream drives the base until it arrives, with no limit: a base the box
            # blocks would spend the skill's whole budget pushing against it.
            self._base_steps += 1
            if self._base_steps > BASE_STEP_LIMIT:
                raise RuntimeError(f"the base did not reach its goal in {BASE_STEP_LIMIT} steps")
            return super().step()  # type: ignore[misc]
        pick = hasattr(self, "_pre_grasp")
        ready = "_pre_grasp" if pick else "_pre_place"
        if not getattr(self, ready):
            action = self._follow(
                key="approach",
                plan=self._current_arm_joint_plan,  # type: ignore[attr-defined]
                final=APPROACH_FINAL_TOLERANCE,
            )
            if action is not None:
                return action
            setattr(self, ready, True)
            settle = getattr(self, "settle", None)
            if settle is not None:
                self.settle = None
                settle()
        gripped = "_closed_gripper" if pick else "_open_gripper"
        if not getattr(self, gripped):
            self._grip_wait += 1
            if self._grip_wait >= GRIP_STEPS:
                setattr(self, gripped, True)
            return self._hold(grip=1.0 if pick else 0.0)
        if pick and self.stop_after_grip:
            self._lifted = True
            return self._hold(grip=1.0)
        lift = getattr(self, "lift", None)
        if lift is not None:
            self.lift = None
            lift()
        action = self._follow(key="retract", plan=self._current_retract_plan)  # type: ignore[attr-defined]
        if action is not None:
            return action
        setattr(self, "_lifted" if pick else "_returned", True)
        return self._hold(grip=self._get_current_robot_gripper_pose())  # type: ignore[attr-defined]


class PreGraspApproach:
    """Mixin over an upstream pick (`PickShelfController`, `PickWiperOriController`):
    plan the approach to `PRE_GRASP_RISE` above the grasp instead of to the grasp.

    Upstream's arm BiRRT goes straight for the grasp configuration and may arrive from
    the side, so a pad meets the 2 cm cube or the 15 mm wiper head before the fingers
    are around it. Planned the same way, but to the pre-grasp; the last few centimetres
    are a straight descent the caller makes (`KinderSkills.settler`), as the scripted
    picks in the cupboard do."""

    # The upstream module whose planner and grasp constant the controller reads.
    grasp_module: ClassVar[str] = "kinder_models.dynamic3d.shelf.parameterized_skills"
    grasp_constant: ClassVar[str] = "GRASP_TRANSFORM_TO_OBJECT"
    # Set by the caller: the pre-grasp arm configuration, already solved and checked in
    # MuJoCo (`KinderSkills.shelf_confs`); None solves it with PyBullet's IK.
    pre_conf: list[float] | None = None

    def reset(self, x: Any, params: Any, **kwargs: Any) -> None:  # noqa: PLR0917
        import importlib

        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.inverse_kinematics import inverse_kinematics

        shelf_skills = importlib.import_module(self.grasp_module)

        super().reset(x, params, **kwargs)  # type: ignore[misc]
        sim = self._pybullet_sim  # type: ignore[attr-defined]
        target = self.objects[1]  # type: ignore[attr-defined]
        plan_x = x.copy()
        end = self._current_base_motion_plan[-1]  # type: ignore[attr-defined]
        robot = self.objects[0]  # type: ignore[attr-defined]
        plan_x.set(robot, "pos_base_x", end.x)
        plan_x.set(robot, "pos_base_y", end.y)
        plan_x.set(robot, "pos_base_rot", end.theta())
        sim.set_state(plan_x)
        grasp = multiply_poses(
            Pose(
                tuple(plan_x.get(target, k) for k in ("x", "y", "z")),
                tuple(plan_x.get(target, k) for k in ("qx", "qy", "qz", "qw")),
            ),
            getattr(shelf_skills, self.grasp_constant),
        )
        above = Pose(
            (grasp.position[0], grasp.position[1], grasp.position[2] + PRE_GRASP_RISE),
            grasp.orientation,
        )
        sim.set_state(plan_x)
        if self.pre_conf is None:
            pre_conf = inverse_kinematics(sim.robot, above, set_joints=False)
        else:
            # The arm's seven joints; the fingers stay as they are.
            pre_conf = [*self.pre_conf[:7], *list(sim.get_robot_joints())[7:]]
        sim.set_state(plan_x)
        to_pre = shelf_skills.run_motion_planning(
            sim.robot,
            sim.get_robot_joints(),
            pre_conf,
            collision_bodies=sim.get_collision_bodies(),
            seed=0,
            physics_client_id=sim.physics_client_id,
        )
        if to_pre is None:
            raise AssertionError("Motion planning to the pre-grasp failed")
        self._current_arm_joint_plan = list(to_pre)  # type: ignore[attr-defined]


class KinderControllers:
    """The adapted upstream controller classes, built once on first use."""

    _classes: ClassVar[dict[str, type]] = {}

    @staticmethod
    def get(*, name: str) -> type:
        if not KinderControllers._classes:
            from kinder_models.dynamic3d.shelf.parameterized_skills import (
                PickShelfController,
                PlaceShelfController,
            )
            from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
                PickWiperOriController,
            )
            from kinder_models.dynamic3d.tossing.parameterized_skills import (
                PickCubeController,
            )

            KinderControllers._classes.update({
                "PickCube": PickCubeController,
                "PickWiper": type("PlannedPickWiper", (FollowPlans, PickWiperOriController), {}),
                "PickWiperInSlot": type(
                    "PlannedPickWiperInSlot",
                    (FollowPlans, PreGraspApproach, PickWiperOriController),
                    {
                        "grasp_module": "kinder_models.dynamic3d.sweep3D.parameterized_skills",
                        "grasp_constant": "WIPER_TRANSFORM_TO_OBJECT",
                    },
                ),
                "PickShelf": type(
                    "PlannedPickShelf", (FollowPlans, PreGraspApproach, PickShelfController), {}
                ),
                "PlaceShelf": type("PlannedPlaceShelf", (FollowPlans, PlaceShelfController), {}),
            })
        return KinderControllers._classes[name]


class KinderSkills:
    """Runs the robot skills on one live scene, sharing `ShelfSkills`' step
    counter, budget, gripper command and recorder, so a KINDER step and a scripted
    step are counted, capped and filmed the same way."""

    def __init__(self, *, shelf: ShelfSkills, backend: Any) -> None:
        self.shelf = shelf
        self.backend = backend
        self.env = shelf.env
        self.core = shelf.core
        self.notes: list[SkillNote] = []
        # Every cube between the pads; `shelf.held_cube` is one of them, the one a
        # retract plan attaches.
        self.held_cubes: frozenset[str] = frozenset()

    def note(self, *, stage: str, kind: str, reason: str, params: Any = None) -> None:
        """Record one candidate's fate for the skill trace (`skill_trace.py`)."""
        values = None if params is None else np.asarray(params, dtype=float).reshape(-1).tolist()
        self.notes.append(SkillNote(stage=stage, kind=kind, reason=reason[:240], params=values))

    # --- state ----------------------------------------------------------------------

    def state(self) -> Any:
        return self.core._get_object_centric_state()  # noqa: SLF001

    def held(self, *, name: str) -> bool:
        """Both gripper pads touch the named body (the backend's own test)."""
        return bool(self.backend.held(name=name))

    def sync(self) -> None:
        """Re-derive `ShelfSkills`' hand bookkeeping from the scene itself: what the
        pads hold, the measured hand-to-wiper pose, the current base and arm. Run
        before and after every skill, so a skill that failed half way -- dropping the
        wiper, missing a cube -- never leaves the next one believing the hand holds
        something it does not (or the reverse)."""
        shelf = self.shelf
        shelf.base_target = shelf.data.qpos[shelf.ba].copy()
        shelf.arm_reference = shelf.data.qpos[shelf.qa].copy()
        if self.held(name=WIPER):
            if shelf.in_hand is None:
                shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ shelf.wiper_pose()
            shelf.grip = 1.0
        else:
            shelf.in_hand = None
        held_cubes = [c for c in (f"cube_{i}" for i in range(5)) if self.held(name=c)]
        self.held_cubes = frozenset(held_cubes)
        if shelf.held_cube not in self.held_cubes:
            shelf.held_cube = held_cubes[0] if held_cubes else None
        if shelf.held_cube is not None:
            shelf.grip = 1.0
        if shelf.in_hand is None and shelf.held_cube is None:
            shelf.grip = 0.0

    def other_held_cubes(self, *, cube: str) -> list[str]:
        """The cubes other than `cube` that both pads touch now."""
        return [c for c in (f"cube_{i}" for i in range(5)) if c != cube and self.held(name=c)]

    def robot_borne(self) -> list[str]:
        """The movable objects the robot carries: held by the pads, or resting on or
        against any robot geom. KINDER's base planner puts the robot's overhead
        footprint at its start pose among every object's footprint, so one of these
        starts every base path in collision -- a cube dropped on the chassis refused
        every pick, and a second held cube every placement, at 0 control steps."""
        shelf = self.shelf
        m, d = shelf.model, shelf.data
        names = [*(f"cube_{i}" for i in range(5)), WIPER]
        bodies = {int(m.body(n).id): n for n in names}
        out = {n for n in names if self.held(name=n)}
        for contact in d.contact[: d.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            for mine, other in (pair, pair[::-1]):
                name = bodies.get(int(m.geom_bodyid[other]))
                if name is not None and mine in shelf.robot_geoms and contact.dist < ROBOT_TOUCH:
                    out.add(name)
        return sorted(out)

    @contextlib.contextmanager
    def base_planner_ignores_robot_borne(self) -> Generator[None, None, None]:
        """Within this, every upstream controller's base planner leaves the objects
        the robot carries (`robot_borne`) out of its obstacles, as upstream already
        does for the one object a controller is placing."""
        import importlib

        borne = self.robot_borne()
        if not borne:
            yield
            return
        with contextlib.ExitStack() as stack:
            for name in BASE_PLANNER_MODULES:
                module = importlib.import_module(name)
                planner = module.run_base_motion_planning

                def ignoring(*args: Any, _planner: Any = planner, **kwargs: Any) -> Any:
                    disabled = list(kwargs.get("disable_collision_objects") or [])
                    kwargs["disable_collision_objects"] = disabled + [
                        n for n in borne if n not in disabled
                    ]
                    return _planner(*args, **kwargs)

                stack.enter_context(
                    self.constant(module=module, name="run_base_motion_planning", value=ignoring)
                )
            yield

    # --- driving an upstream controller -----------------------------------------------

    def env_step(self, *, action: np.ndarray, label: str) -> None:
        shelf = self.shelf
        if shelf.ticks >= shelf.tick_limit:
            raise StepBudgetExhausted(f"{shelf.tick_limit} control steps reached during {label}")
        clipped = np.clip(
            np.asarray(action, dtype=np.float64),
            self.env.action_space.low,
            self.env.action_space.high,
        )
        self.env.step(clipped)
        shelf.ticks += 1
        shelf.grip = float(clipped[-1])
        shelf.tally_contacts()
        shelf.track_slip()
        if shelf.recorder is not None:
            shelf.recorder.on_step(skills=shelf, label=label)

    def drive(self, *, controller: Any, params: Any, label: str) -> str:
        """Plan with `controller.reset`, then execute it to termination. Returns
        "planned-out" if planning found nothing for these parameters (nothing moved);
        raises `SkillFailed` if execution failed after moving."""
        from bilevel_planning.trajectory_samplers.trajectory_sampler import (
            TrajectorySamplingFailure,
        )

        try:
            with self.base_planner_ignores_robot_borne():
                controller.reset(self.state(), params)
        except (TrajectorySamplingFailure, AssertionError, ValueError, RuntimeError) as exc:
            self.note(stage=label, kind="no_plan", params=params, reason=repr(exc))
            return "planned-out"
        except Exception as exc:  # noqa: BLE001  (IK and planner errors carry many types)
            self.note(
                stage=label, kind="no_plan", params=params, reason=f"{type(exc).__name__}: {exc}"
            )
            return "planned-out"
        blocked = self.plan_blocked(controller=controller)
        if blocked:
            self.note(stage=label, kind="plan_blocked", params=params, reason=blocked)
            return "planned-out"
        start = self.shelf.ticks
        try:
            while not controller.terminated():
                self.env_step(action=controller.step(), label=label)
                controller.observe(self.state())
        except SkillFailed as exc:
            self.note(stage=label, kind="execution_failed", params=params, reason=str(exc))
            raise
        except (TrajectorySamplingFailure, Exception) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.note(stage=label, kind="execution_failed", params=params, reason=reason)
            raise SkillFailed(f"{label}: execution failed: {reason}") from exc
        finally:
            self.sync()
        self.note(
            stage=label,
            kind="executed",
            params=params,
            reason=f"{self.shelf.ticks - start} control steps",
        )
        return "done"

    def plan_blocked(self, *, controller: Any) -> str | None:
        """Why the controller's arm plans would collide in the MuJoCo scene, if they
        would; None when clear (or the controller keeps no such plans)."""
        shelf = self.shelf
        bases = getattr(controller, "_current_base_motion_plan", None)
        if not bases:
            return None
        end = bases[-1]
        base = np.array([end.x, end.y, end.theta()])
        d = shelf.ik_data
        objects = getattr(controller, "objects", None) or []
        targets = {getattr(o, "name", None) for o in objects[1:]}
        wiper_resting = WIPER not in targets and not self.wiper_on_robot()
        for name in ("_current_arm_joint_plan", "_current_retract_plan"):
            if name == "_current_retract_plan" and getattr(controller, "stop_after_grip", False):
                continue
            plan = getattr(controller, name, None)
            if not plan:
                continue
            confs = [np.array(q[:7], dtype=float) for q in plan]
            for a, b in zip(confs, confs[1:], strict=False):
                gap = ((b - a) + np.pi) % (2 * np.pi) - np.pi
                for t in np.linspace(0.0, 1.0, int(np.ceil(np.max(np.abs(gap)) / CHECK_STEP)) + 1):
                    d.qpos[:] = shelf.data.qpos
                    d.qpos[shelf.ba] = base
                    d.qpos[shelf.qa] = a + t * gap
                    shelf.mujoco.mj_kinematics(shelf.model, d)
                    if shelf.arm_touches(d=d):
                        return f"{name.split('_')[2]} plan hits {shelf.touching(d=d)}"
                    if wiper_resting and self.robot_touches_wiper(d=d):
                        return f"{name.split('_')[2]} plan hits the resting wiper"
                    if name == "_current_arm_joint_plan" and (hit := self.hand_hits(d=d)):
                        return f"approach plan puts the hand into {hit}"
        return None

    def robot_touches_wiper(self, *, d: Any) -> bool:
        """Whether any robot geom penetrates the wiper in `d` (contacts already computed
        by `arm_touches`)."""
        shelf = self.shelf
        wiper = set(shelf.wiper_geoms)
        for contact in d.contact[: d.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & shelf.robot_geoms and pair & wiper and contact.dist < 0.0:
                return True
        return False

    def hand_hits(self, *, d: Any) -> str | None:
        """What the gripper itself touches in `d` on the way in (contacts already
        computed by `arm_touches`): the wiper, which the planning scene has only at rest,
        or the cupboard or box beyond `HAND_ALLOWANCE` -- PyBullet's side panels sit a
        centimetre outside MuJoCo's. Only the approach is checked so: the retract
        starts at the grasp, where the pads are meant to be against things."""
        shelf = self.shelf
        wiper = set(shelf.wiper_geoms)
        fixed = shelf.cupboard_geoms | shelf.box_geoms
        for contact in d.contact[: d.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & shelf.robot_geoms:
                continue
            if pair & wiper:
                return "the wiper"
            if pair & fixed and -contact.dist > HAND_ALLOWANCE:
                other = (pair & fixed).pop()
                return str(shelf.model.geom(other).name or "the box")
        return None

    def first_that_plans(
        self, *, make: Callable[[], Any], candidates: list[Any], label: str
    ) -> None:
        """Bilevel-refiner style: the first parameter vector whose `reset` plans is
        executed; one execution only."""
        for params in candidates:
            if self.drive(controller=make(), params=params, label=label) == "done":
                return
        raise SkillFailed(f"{label}: no candidate planned ({len(candidates)} tried)")

    # --- planning-scene helpers -------------------------------------------------------

    def planning_sim(self, *, target: str | None = None) -> Any:
        """KINDER's PyBullet planning scene for the current state. Upstream builds the
        cupboard and every cube; this adds the box and the resting wiper -- unless the
        wiper is `target`, the object the skill picks up, or `wiper_on_robot`."""
        from kinder_models.dynamic3d.utils import PyBulletSim
        from pybullet_helpers.geometry import Pose

        x = self.state()
        sim = PyBulletSim(x)
        box = x.get_object_from_name(BOX)
        pose = Pose(
            tuple(x.get(box, k) for k in ("x", "y", "z")),
            tuple(x.get(box, k) for k in ("qx", "qy", "qz", "qw")),
        )
        sim.add_bin(name=BOX, pose=pose, **BOX_DIMENSIONS)
        if target != WIPER and not self.wiper_on_robot():
            self.add_wiper_obstacle(sim=sim)
        return sim

    def wiper_on_robot(self) -> bool:
        """Whether the robot carries the wiper: held by the pads, or resting on or
        against any robot geom (within `WIPER_ROBOT_TOUCH`). Either way it moves with the
        robot or starts every plan in collision, so it is no obstacle to plan around."""
        shelf = self.shelf
        if self.held(name=WIPER):
            return True
        m, d = shelf.model, shelf.data
        body = int(m.body(WIPER).id)
        for contact in d.contact[: d.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            for mine, other in (pair, pair[::-1]):
                if (
                    int(m.geom_bodyid[other]) == body
                    and mine in shelf.robot_geoms
                    and contact.dist < WIPER_ROBOT_TOUCH
                ):
                    return True
        return False

    def add_wiper_obstacle(self, *, sim: Any) -> None:
        """The wiper's two boxes, as they lie now and grown by `WIPER_OBSTACLE_MARGIN`,
        as static colliders. At its edge home the handle overhangs the cupboard's front,
        outside the cupboard that once shielded it from every plan, and the approaches
        planned into it were refused by the MuJoCo re-check."""
        from pybullet_helpers.geometry import Pose, set_pose
        from pybullet_helpers.utils import create_pybullet_block

        shelf = self.shelf
        for g in shelf.wiper_geoms:
            hx, hy, hz = (float(v) + WIPER_OBSTACLE_MARGIN for v in shelf.model.geom_size[g])
            body = create_pybullet_block(
                color=(0.2, 0.2, 0.2, 1.0),
                half_extents=(hx, hy, hz),
                physics_client_id=sim.physics_client_id,
            )
            x, y, z = (float(v) for v in shelf.data.geom_xpos[g])
            rotation = Rotation.from_matrix(shelf.data.geom_xmat[g].reshape(3, 3))
            qx, qy, qz, qw = (float(v) for v in rotation.as_quat())
            set_pose(body, Pose((x, y, z), (qx, qy, qz, qw)), sim.physics_client_id)
            sim._static_colliders[f"{WIPER}/geom{g}"] = body  # noqa: SLF001

    def objects(self, *names: str) -> list[Any]:
        x = self.state()
        return [x.get_object_from_name(n) for n in names]

    def yaw(self, *, name: str) -> float:
        x = self.state()
        o = x.get_object_from_name(name)
        q = [x.get(o, k) for k in ("qx", "qy", "qz", "qw")]
        return float(Rotation.from_quat(q).as_euler("xyz")[2])

    @staticmethod
    def pose_in(*, obj_pose: np.ndarray, hand: np.ndarray) -> Any:
        """A pinch-site hand pose, expressed as the end-effector pose in the object's
        frame that an upstream `*_TRANSFORM_TO_OBJECT` constant holds."""
        from pybullet_helpers.geometry import Pose

        site_to_ee = np.eye(4)
        site_to_ee[:3, 3] = [0.0, 0.0005, -0.0052]
        m = np.linalg.inv(obj_pose) @ hand @ site_to_ee
        return Pose(tuple(m[:3, 3]), tuple(Rotation.from_matrix(m[:3, :3]).as_quat()))

    @staticmethod
    @contextlib.contextmanager
    def constant(*, module: Any, name: str, value: Any) -> Generator[None, None, None]:
        """Set one of an upstream controller module's constants for one call."""
        original = getattr(module, name)
        setattr(module, name, value)
        try:
            yield
        finally:
            setattr(module, name, original)

    def cube_pose(self, *, cube: str) -> np.ndarray:
        position, quaternion = self.core.get_object(cube).get_pose()
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = position
        return pose

    def held_transform(self, *, sim: Any, cube: str) -> Any:
        from pybullet_helpers.geometry import Pose, multiply_poses

        x = self.state()
        sim.set_state(x)
        c = x.get_object_from_name(cube)
        cube_pose = Pose(
            tuple(x.get(c, k) for k in ("x", "y", "z")),
            tuple(x.get(c, k) for k in ("qx", "qy", "qz", "qw")),
        )
        return multiply_poses(sim.get_ee_pose().invert(), cube_pose)

    def base_params(self, *, base_xy: tuple[float, float]) -> tuple[float, float, float]:
        """`PlaceShelfController`'s (distance, y offset, rotation) for a base at
        `base_xy` facing +x: its base pose is set relative to the cupboard."""
        x = self.state()
        cupboard = x.get_object_from_name(CUPBOARD)
        return (
            float(x.get(cupboard, "x") - base_xy[0]),
            float(base_xy[1] - x.get(cupboard, "y")),
            -np.pi / 2,
        )

    # --- the arm retract --------------------------------------------------------------

    def gripper_closed(self) -> bool:
        x = self.state()
        return float(x.get(x.get_object_from_name("robot"), "pos_gripper")) > 0.02

    def arm_from_home(self) -> float:
        arm = self.shelf.data.qpos[self.shelf.qa]
        return float(np.max(np.abs((arm - HOME_ARM + np.pi) % (2 * np.pi) - np.pi)))

    def retract(self, *, label: str) -> None:
        """Bring the arm to `HOME_ARM` along a collision-free plan (KINDER's arm BiRRT,
        the held cube attached if there is one) -- or, holding the wiper, straight up
        to the carry height `CARRY_Z`."""
        if self.arm_from_home() < RETRACT_TOLERANCE:
            return
        shelf = self.shelf
        if shelf.in_hand is not None:
            # Holding the wiper the arm goes no further than the scripted carry height:
            # at KINDER's home, the sweep grasp lets the wiper slide out of the pads.
            with contextlib.suppress(SkillFailed):
                shelf.raise_hand(z=CARRY_Z, label=f"{label}_raise", holding=True)
            return
        plan = self.plan_home()
        if plan is None:
            # KINDER's PyBullet cupboard is not MuJoCo's: a hand MuJoCo leaves just inside
            # the cupboard's open front, touching nothing, can start the plan in collision
            # there, and BiRRT never plans from a colliding start -- so the arm stayed out
            # and every later skill planned from it and failed too. Back the hand out
            # toward the base on MuJoCo's own kinematics, then plan again.
            self.back_out(label=f"{label}_back_out")
            plan = self.plan_home()
        if plan is None:
            self.note(stage=label, kind="retract", reason="no collision-free retract; arm left")
            return
        confs = [np.array(q[:7]) for q in plan]
        if shelf.in_hand is not None and self.carried_wiper_blocked(confs=confs):
            self.note(
                stage=label, kind="retract", reason="carried wiper would hit; arm left raised"
            )
            return
        follower = PlanFollower(plan=confs)
        for _ in range(FINAL_STEP_LIMIT + 20 * len(plan)):
            action = follower.action(current=shelf.data.qpos[shelf.qa], grip=shelf.grip)
            if action is None:
                break
            self.env_step(action=action, label=label)
        self.sync()

    def plan_home(self) -> list[Any] | None:
        """KINDER's arm BiRRT from the current arm to `HOME_ARM`, the held cube attached
        if there is one; None if it finds no plan."""
        from pybullet_helpers.motion_planning import run_motion_planning

        shelf = self.shelf
        sim = self.planning_sim()
        x = self.state()
        sim.set_state(x)
        held = None
        if shelf.held_cube is not None:
            sim.base_link_to_held_obj = self.held_transform(sim=sim, cube=shelf.held_cube)
            held = sim._cubes[shelf.held_cube]  # noqa: SLF001
            sim.set_state(x, x.get_object_from_name(shelf.held_cube))
        return run_motion_planning(
            sim.robot,
            sim.get_robot_joints(),
            [*HOME_ARM.tolist(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            collision_bodies=sim.get_collision_bodies(held_object=held),
            held_object=held,
            base_link_to_held_obj=sim.base_link_to_held_obj if held is not None else None,
            seed=0,
            physics_client_id=sim.physics_client_id,
        )

    def back_out(self, *, label: str) -> None:
        """Move the hand `BACK_OUT` straight toward the base, orientation kept."""
        shelf = self.shelf
        hand = shelf.site_pose()
        base = shelf.data.qpos[shelf.ba]
        toward = np.array([base[0] - hand[0, 3], base[1] - hand[1, 3], 0.0])
        if np.linalg.norm(toward) < 1e-6:
            return
        target = hand[:3, 3] + BACK_OUT * toward / np.linalg.norm(toward)
        with contextlib.suppress(SkillFailed):
            shelf.path(
                position=target,
                rotation=hand[:3, :3],
                label=label,
                count=3,
                holding=shelf.held_cube is not None,
            )
        self.sync()

    def carried_wiper_blocked(self, *, confs: list[np.ndarray]) -> bool:
        """Whether the held wiper, carried rigidly along these arm configurations with
        the grasp as measured, would cut into the cupboard, the box or the robot's own
        arm or base (the planning scene leaves out a carried wiper)."""
        shelf = self.shelf
        assert shelf.in_hand is not None
        m, d = shelf.model, shelf.ik_data
        base = shelf.data.qpos[shelf.ba].copy()
        address = m.jnt_qposadr[m.joint(f"{WIPER}_joint").id]
        wiper = set(shelf.wiper_geoms)
        others = shelf.cupboard_geoms | shelf.box_geoms | (shelf.robot_geoms - shelf.gripper_geoms)
        for a, b in zip(confs, confs[1:], strict=False):
            gap = ((b - a) + np.pi) % (2 * np.pi) - np.pi
            for t in np.linspace(0.0, 1.0, int(np.ceil(np.max(np.abs(gap)) / CHECK_STEP)) + 1):
                pose = shelf.site_at(base=base, arm=a + t * gap) @ shelf.in_hand
                d.qpos[address : address + 3] = pose[:3, 3]
                d.qpos[address + 3 : address + 7] = Rotation.from_matrix(pose[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                shelf.mujoco.mj_kinematics(m, d)
                shelf.mujoco.mj_collision(m, d)
                for contact in d.contact[: d.ncon]:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & wiper and pair & others and -contact.dist > CARRIED_ALLOWANCE:
                        return True
        return False

    # --- the skills -------------------------------------------------------------------

    def run(self, *, skill: Callable[..., None], label: str, **kwargs: Any) -> None:
        """Every skill: bookkeeping from the scene, an empty hand opened, the skill, and
        the arm retract -- after a failure too, so no skill starts where the last one
        gave up."""
        self.sync()
        shelf = self.shelf
        if shelf.in_hand is None and shelf.held_cube is None:
            for _ in range(GRIP_STEPS if self.gripper_closed() else 0):
                self.env_step(action=FollowPlans._hold(grip=0.0), label=f"{label}_open")  # noqa: SLF001
        try:
            skill(**kwargs)
        finally:
            self.sync()
            with contextlib.suppress(SkillFailed):
                self.retract(label=f"{label}_retract")

    def settler(
        self,
        *,
        hand: np.ndarray,
        label: str,
        entry: tuple[np.ndarray, ...] = (),
        cube: str | None = None,
    ) -> Callable[[], None]:
        """The approach from the pre-grasp (`PreGraspApproach`) to the pinch-site pose
        `hand`, in straight lines on MuJoCo's own kinematics: KINDER solves IK in
        PyBullet, whose arm sits up to ~5 mm from MuJoCo's at the same joints -- enough
        to put a pad on top of a 15 mm wiper head or a 2 cm cube. The pre-grasp is
        `entry`'s first pose, the hand passing through the rest on the way in; with no
        `entry`, `PRE_GRASP_RISE` straight above the grasp.

        A move that stops short is tolerated, since the hand may still have arrived
        close enough; but given a `cube`, the approach fails unless that cube is between
        the pads (`cube_in_jaws`). Tolerated without that check, an approach stalled
        17 cm short closed the gripper on air (layout-56)."""

        def settle() -> None:
            shelf = self.shelf
            shelf.base_target = shelf.data.qpos[shelf.ba].copy()
            shelf.arm_reference = shelf.data.qpos[shelf.qa].copy()
            above = hand.copy()
            above[2, 3] += PRE_GRASP_RISE
            first, *rest = entry or (above,)
            stopped = ""
            try:
                shelf.settle_at(pose=first, label=label)
                for pose in rest:
                    shelf.path(position=pose[:3, 3], rotation=pose[:3, :3], label=label, count=6)
                shelf.path(position=hand[:3, 3], rotation=hand[:3, :3], label=label, count=3)
            except SkillFailed as exc:
                stopped = f" ({exc})"
            with contextlib.suppress(SkillFailed):
                shelf.settle_at(pose=hand, label=label)
            if cube is not None and not self.cube_in_jaws(cube=cube):
                raise SkillFailed(
                    f"{label}: the approach ended with {cube} not between the pads{stopped}"
                )

        return settle

    def lifter(
        self,
        *,
        pre: np.ndarray,
        label: str,
        cube: str,
        outside: np.ndarray | None = None,
    ) -> Callable[[], None]:
        """The way back out from the grasp to the pre-grasp `pre`, then on to `outside`
        just outside the cupboard's open front (`shelf_exit`) when there is one,
        holding the cube, in straight lines on MuJoCo's kinematics, so the retract plan
        starts clear of the boards. Planned from the grasp itself, a cube at the back of
        the shelf left the wrist inside KINDER's lower board above, and no retract
        planned from any base; from the pre-grasp, a cube in the back corner still had
        none, with either end clear.

        Pads closed on `cube` and another cube too are opened where they are, before
        anything is lifted, and the execution fails: the caller's pick counts it as a
        miss and tries again. Carried out together, two cubes came home in the hand and
        the pick reported success."""

        def lift() -> None:
            shelf = self.shelf
            others = self.other_held_cubes(cube=cube)
            if others:
                self.release(label=f"{label}_open")
                raise RuntimeError(
                    f"the pads closed on two cubes, {cube} and {', '.join(others)}; opened"
                )
            shelf.base_target = shelf.data.qpos[shelf.ba].copy()
            shelf.arm_reference = shelf.data.qpos[shelf.qa].copy()
            for pose in (pre,) if outside is None else (pre, outside):
                try:
                    shelf.path(
                        position=pose[:3, 3],
                        rotation=pose[:3, :3],
                        label=label,
                        count=3 if pose is pre else 6,
                        holding=True,
                    )
                except SkillFailed as exc:
                    # The retract plan starts where the lift was to end: followed from
                    # anywhere else it cuts through the cupboard's front edge, and the
                    # board knocked a held cube out of the pads. Stop here; the skill's
                    # own retract (`retract`) plans from where the hand really is.
                    raise RuntimeError(f"the lift out of the cupboard stopped: {exc}") from exc

        return lift

    def shelf_exit(
        self, *, pre: np.ndarray, pre_conf: np.ndarray, base: np.ndarray, cube: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The hand pose `SHELF_PICK_EXIT` outside the cupboard's open front, level with
        the pre-grasp and `SHELF_PICK_EXIT_RISE` above it, and its arm configuration
        from `base` on the pre-grasp's IK branch, clear as `shelf_grasp_conf` checks;
        None for the configuration if there is none."""
        shelf = self.shelf
        out = pre.copy()
        out[0, 3] = min(float(pre[0, 3]), float(shelf.front_x) - SHELF_PICK_EXIT)
        out[2, 3] += SHELF_PICK_EXIT_RISE
        try:
            conf, collision = shelf.ik(
                position=out[:3, 3], rotation=out[:3, :3], base=base, seed=pre_conf, attempts=2
            )
        except SkillFailed:
            return out, None
        if collision > 0 or self.touches_other_cube(base=base, arm=conf, cube=cube):
            return out, None
        return out, self.wrapped(arm=conf)

    def pick_wiper(self) -> None:
        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_wiper: the hand is not empty")
        if ("WiperHome", (WIPER,)) not in self.backend.abstract_atoms():
            self.pick_wiper_once()
            return
        # Closed loop in the slot: re-observe the wiper and replan for each attempt, and
        # take it out only once both pads hold it.
        for attempt in range(1, SLOT_PICK_ATTEMPTS + 1):
            self.pick_wiper_once()
            if self.held(name=WIPER):
                self.sync()
                # Out of the slot the arm is left as the scripted pick leaves it, for
                # `run`'s retract to raise to the carry height: SweepShelf's and
                # ReturnWiper's routes start from there.
                self.take_wiper_out_of_slot(label="pick_wiper_out_of_slot")
                if self.held(name=WIPER):
                    return
                raise SkillFailed("pick_wiper: the wiper slipped out of the hand on the way out")
            self.note(
                stage="pick_wiper",
                kind="execution_failed",
                reason=f"attempt {attempt} closed on nothing; opening",
            )
            for _ in range(GRIP_STEPS):
                self.env_step(action=FollowPlans._hold(grip=0.0), label="pick_wiper_open")  # noqa: SLF001
            self.sync()
            self.retract(label="pick_wiper_regrasp_retract")
            if ("WiperHome", (WIPER,)) not in self.backend.abstract_atoms():
                break
        raise SkillFailed("pick_wiper: the wiper did not come with the hand")

    def take_wiper_out_of_slot(self, *, label: str) -> None:
        """Lift the held wiper off its board, then back the base straight away from the
        cupboard, the arm held still, until the wiper's far end is `SLOT_EXIT_CLEARANCE`
        outside the cupboard's open front: the wiper leaves the slot along the slot,
        unturned. (`ShelfSkills.leave_wiper_slot` pulls it out with the arm instead, and
        from KINDER's pick base its exit pose has no IK solution.)"""
        shelf = self.shelf
        lift = shelf.site_pose()
        lift[2, 3] += SLOT_LIFT
        with contextlib.suppress(SkillFailed):
            shelf.path(
                position=lift[:3, 3], rotation=lift[:3, :3], label=label, count=4, holding=True
            )
        self.sync()
        out = self.wiper_far_x() - shelf.front_x + SLOT_EXIT_CLEARANCE
        if out > 0.0:
            base = shelf.data.qpos[shelf.ba].copy()
            away = np.array([np.cos(base[2]), np.sin(base[2])])
            base[:2] -= out * away
            shelf.move_base(target=base, label=label)
        self.sync()

    def wiper_far_x(self) -> float:
        """The largest world x any part of the wiper reaches (its geoms' bounding boxes)."""
        shelf = self.shelf
        m, d = shelf.model, shelf.data
        far = -np.inf
        for g in shelf.wiper_geoms:
            rotation = d.geom_xmat[g].reshape(3, 3)
            centre = d.geom_xpos[g] + rotation @ m.geom_aabb[g][:3]
            half = np.abs(rotation) @ m.geom_aabb[g][3:]
            far = max(far, float(centre[0] + half[0]))
        return far

    def pick_wiper_once(self) -> None:
        """One planned pick: off the shelf it is upstream's, lifted on its retract plan;
        in the slot the hand stops as the gripper closes (`FollowPlans.stop_after_grip`).
        That retract plan is BiRRT in a planning scene that leaves out the wiper it picks:
        from a wiper 0.15 deg off square it took the hand 9 cm deeper into the cupboard,
        and the back panel tore the wiper out of the pads."""
        import kinder_models.dynamic3d.sweep3D.parameterized_skills as sweep3d

        shelf = self.shelf
        robot, wiper = self.objects("robot", WIPER)
        yaw = self.yaw(name=WIPER)
        candidates = [(0.7, -np.pi)]  # upstream's own fixed parameter first
        for heading in PICK_WIPER_HEADINGS:
            candidates += [(d, heading - yaw) for d in PICK_WIPER_DISTANCES]
        # At home, the one leaned grasp the wiper is swept with; anywhere else,
        # upstream's own grasp.
        grasp = sweep3d.WIPER_TRANSFORM_TO_OBJECT
        controller = KinderControllers.get(name="PickWiper")
        settle = None
        in_slot = ("WiperHome", (WIPER,)) in self.backend.abstract_atoms()
        if in_slot:
            hand = shelf.wiper_grasp()
            grasp = self.pose_in(obj_pose=shelf.wiper_pose(), hand=hand)
            controller = KinderControllers.get(name="PickWiperInSlot")
            settle = self.settler(hand=hand, label="pick_wiper_settle")

        def make() -> Any:
            pick = controller(objects=[robot, wiper], pybullet_sim=self.planning_sim(target=WIPER))
            pick.settle = settle
            pick.stop_after_grip = in_slot
            return pick

        with self.constant(module=sweep3d, name="WIPER_TRANSFORM_TO_OBJECT", value=grasp):
            self.first_that_plans(
                make=make,
                candidates=candidates,
                label="pick_wiper",
            )
        if not in_slot and not self.held(name=WIPER):
            raise SkillFailed("pick_wiper: the wiper did not come with the hand")

    def pick_cube_from_floor(self, *, cube: str) -> None:
        """The grasp x base candidate search of `floor_pick.FloorPicks`."""
        from .floor_pick import FloorPicks  # it imports this module

        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_cube_from_floor: the hand is not empty")
        FloorPicks(kinder=self).pick(cube=cube, label="pick_cube_from_floor")
        if not self.held(name=cube):
            raise SkillFailed(f"pick_cube_from_floor: {cube} did not come with the hand")
        if self.other_held_cubes(cube=cube):
            raise SkillFailed(f"pick_cube_from_floor: {cube} came with another cube")

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        """Closed loop over grasp attempts: plan from the cube as it lies now, execute
        the first candidate that plans, and if the cube did not come with the hand,
        open, look again and replan -- up to `SHELF_PICK_EXECUTIONS` executions."""
        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_cube_from_shelf: the hand is not empty")
        missed: list[str] = []
        tried: list[np.ndarray] = []
        refused: list[str] = []
        for _ in range(SHELF_PICK_EXECUTIONS):
            if not self.on_cube_shelf(cube=cube):
                break
            try:
                planned = self.shelf_pick_attempt(cube=cube, tried=tried, refused=refused)
            except StepBudgetExhausted:
                raise
            except SkillFailed as exc:
                missed.append(str(exc))
                planned = True
            if not planned:
                if missed:
                    break
                swept = f"; refused grasps sweeping {', '.join(sorted(set(refused)))}"
                raise SkillFailed(
                    f"pick_cube_from_shelf: no grasp on {cube} planned{swept if refused else ''}"
                )
            if self.held(name=cube):
                if not self.double_grasp(cube=cube, missed=missed, label="pick_cube_from_shelf"):
                    return
                continue
            missed.append(f"{cube} did not come with the hand")
            self.note(
                stage="pick_cube_from_shelf",
                kind="execution_failed",
                reason=f"attempt {len(missed)} missed; replanning",
            )
            self.sync()
            self.retract(label="pick_cube_from_shelf_regrasp")
            for _ in range(GRIP_STEPS):
                self.env_step(action=FollowPlans._hold(grip=0.0), label="pick_cube_from_shelf_open")  # noqa: SLF001
            self.sync()
        raise SkillFailed(f"pick_cube_from_shelf: {cube} not picked ({'; '.join(missed)})"[:300])

    def double_grasp(self, *, cube: str, missed: list[str], label: str) -> bool:
        """Whether a pick holding `cube` holds another cube too; if so it is a miss:
        recorded, the gripper opened and the arm retracted, ready for the next try."""
        others = self.other_held_cubes(cube=cube)
        if not others:
            return False
        missed.append(f"closed on two cubes, {cube} and {', '.join(others)}")
        self.note(
            stage=label,
            kind="execution_failed",
            reason=f"attempt {len(missed)} closed on two cubes, {cube} and "
            f"{', '.join(others)}; opening",
        )
        self.release(label=f"{label}_open")
        self.retract(label=f"{label}_regrasp")
        return True

    def release(self, *, label: str) -> None:
        """Open the gripper where the hand is."""
        for _ in range(GRIP_STEPS):
            self.env_step(action=FollowPlans._hold(grip=0.0), label=label)  # noqa: SLF001
        self.sync()

    def on_cube_shelf(self, *, cube: str) -> bool:
        """Whether the cube still rests on the cube shelf, inside the cupboard."""
        shelf = self.shelf
        x, _, z = shelf.cube_position(cube=cube)
        return bool(x > shelf.front_x and abs(z - shelf.board[CUBE_SHELF] - 0.01) < 0.01)

    def shelf_pick_attempt(
        self, *, cube: str, tried: list[np.ndarray], refused: list[str] | None = None
    ) -> bool:
        """One grasp execution, from the first (grasp, base) candidate whose arm
        configurations are clear in MuJoCo and whose plans KINDER finds; False when
        none plans (nothing moved). Approach directions in `tried` already missed this
        cube and are not tried again; the one executed is added. Cubes that ruled
        grasps out (`shelf_candidates`) are added to `refused`."""
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills

        shelf = self.shelf
        robot, target = self.objects("robot", cube)
        centre = shelf.cube_position(cube=cube)
        obj_pose = self.cube_pose(cube=cube)
        controller = KinderControllers.get(name="PickShelf")
        planner = shelf_skills.run_motion_planning
        solved_grasp: list[np.ndarray] = []
        solved_pre: list[np.ndarray] = []
        failures: dict[int, int] = {}

        def held_on_support(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("held_object") is not None:
                kwargs.setdefault("distance_threshold", HELD_ON_SUPPORT_THRESHOLD)
                if solved_pre:
                    # The retract starts where `lifter` leaves the hand, outside the
                    # front (`shelf_exit`) or at the pre-grasp, not at the grasp in
                    # among the boards.
                    start = [*solved_pre[-1].tolist(), *list(args[1])[7:]]
                    args = (args[0], start, *args[2:])
            elif solved_grasp and np.allclose(np.asarray(args[2])[:7], solved_grasp[-1]):
                # Upstream's own approach, straight to the grasp: `PreGraspApproach`
                # replaces it with its plan to the pre-grasp, so it is not planned.
                return [list(args[1]), list(args[2])]
            else:
                # The approach to the pre-grasp, whose end is solved clear in MuJoCo.
                # KINDER's planning scene adds its own cupboard model to the compiled
                # one, boards 6.35 mm low, and an arm a few millimetres off MuJoCo's:
                # into the back of the shelf it put the wrist 1-8 mm into the board
                # above at every pre-grasp MuJoCo found clear. Planned with that much
                # slack, every plan is still re-checked in MuJoCo (`plan_blocked`).
                kwargs.setdefault("distance_threshold", -SHELF_PICK_PLANNING_SLACK)
            # Out of the back of the shelf with a cube in hand the way out is narrow:
            # BiRRT with upstream's seed found none where another seed does.
            for seed in SHELF_PICK_PLAN_SEEDS:
                plan = planner(*args, **{**kwargs, "seed": seed})
                if plan is not None:
                    return plan
            return None

        for drives, (hand, entry, base, grasp_conf, pre_conf) in enumerate(
            self.shelf_candidates(cube=cube, centre=centre, tried=tried, refused=refused),
            start=1,
        ):
            if failures.get(id(hand), 0) >= SHELF_PICK_PLANS_PER_GRASP:
                continue
            pre = entry[-1]
            exit_pose, exit_conf = self.shelf_exit(pre=pre, pre_conf=pre_conf, base=base, cube=cube)
            solved_grasp.append(grasp_conf)
            solved_pre.append(pre_conf if exit_conf is None else exit_conf)
            grasp = self.pose_in(obj_pose=obj_pose, hand=hand)
            with (
                self.constant(module=shelf_skills, name="GRASP_TRANSFORM_TO_OBJECT", value=grasp),
                self.constant(
                    module=shelf_skills,
                    name="inverse_kinematics",
                    value=self.fixed_ik(conf=grasp_conf),
                ),
                self.constant(
                    module=shelf_skills, name="run_motion_planning", value=held_on_support
                ),
            ):
                pick = controller(objects=[robot, target], pybullet_sim=self.planning_sim())
                pick.settle = self.settler(
                    hand=hand, entry=entry, label="pick_cube_from_shelf_settle", cube=cube
                )
                pick.lift = self.lifter(
                    pre=pre,
                    outside=None if exit_conf is None else exit_pose,
                    label="pick_cube_from_shelf_lift",
                    cube=cube,
                )
                pick.pre_conf = pre_conf.tolist()
                pick.retract_gain = SHELF_PICK_CARRY_GAIN
                params = self.facing_params(base=base, centre=centre, cube=cube)
                try:
                    outcome = self.drive(
                        controller=pick, params=params, label="pick_cube_from_shelf"
                    )
                except SkillFailed:
                    # Executed and failed: the next attempt must not run it again.
                    tried.append(hand[:3, 2].copy())
                    raise
            if outcome == "done":
                tried.append(hand[:3, 2].copy())
                return True
            failures[id(hand)] = failures.get(id(hand), 0) + 1
            if drives >= SHELF_PICK_PLANS:
                return False
        return False

    def shelf_candidates(
        self,
        *,
        cube: str,
        centre: np.ndarray,
        tried: list[np.ndarray],
        refused: list[str] | None = None,
    ) -> Generator[ShelfCandidate, None, None]:
        """(grasp, entry, base, grasp arm, arm at the entry's first pose) candidates
        clear in MuJoCo, lazily, in `shelf_hands` order. A grasp whose pads would close
        on the cupboard itself (`cupboard_between_pads`) is not offered, nor is one
        whose pads pass within `SHELF_PICK_NEIGHBOUR_CLEARANCE` of another cube on the
        way in or as they close (`neighbour_in_sweep`; the cube is added to `refused`),
        nor one whose hand touches another cube at the grasp or the pre-grasp. In a
        cluster those were offered last, after every other grasp, and were then the
        only grasps left: the jaws struck neighbours 3-3.5 cm away, shoving them 2-7 cm
        or carrying them off beside the target. With no other grasp, the pick refuses.

        Each pre-grasp is an entry of its own; after them, one that starts outside the
        cupboard's front (`shelf_exit`) and comes in through the first pre-grasp: into
        the back of the shelf, the plan to a pre-grasp can be refused by the MuJoCo
        re-check from every base while the way in from the front is clear.

        Whether a grasp is clear barely depends on the base, so the first pass gives a
        grasp up after `SHELF_PICK_UNREACHABLE_BASES` unreachable bases in a row. Once
        everything that pass found has been offered, a grasp it gave up that was clear
        from some base it did try gets the bases after the run, with no early exit: a
        cube by a side panel had one clear grasp, clear from 16/39 bases but with
        pre-grasps from only three of them, all after a run of three unreachable ones.
        The first pass is unchanged, so a pick it finds pays nothing for the second."""
        refused = [] if refused is None else refused
        bases = [b for b in self.shelf_pick_bases(centre=centre) if self.base_clear(base=b)]
        drivable: dict[int, bool] = {}
        given_up: list[tuple[np.ndarray, tuple[np.ndarray, ...], int]] = []
        for hand, all_pres in self.shelf_hands(cube=cube, centre=centre):
            if any(float(hand[:3, 2] @ t) > SHELF_PICK_SAME_APPROACH for t in tried):
                continue
            if self.cupboard_between_pads(hand=hand, cube=cube):
                continue
            if self.swept(path=[hand], cube=cube, refused=refused):
                continue
            pres = tuple(
                pre
                for pre in all_pres
                if not self.swept(path=[pre, hand], cube=cube, refused=refused)
            )
            if not pres:
                continue
            unreachable = 0
            clear_somewhere = False
            for i, base in enumerate(bases):
                if i not in drivable:
                    drivable[i] = self.base_drivable(base=base)
                if not drivable[i]:
                    continue
                grasp_conf = self.shelf_grasp_conf(hand=hand, base=base, cube=cube)
                if grasp_conf is None:
                    unreachable += 1
                    if unreachable >= SHELF_PICK_UNREACHABLE_BASES:
                        if clear_somewhere:
                            given_up.append((hand, pres, i + 1))
                        break
                    continue
                unreachable = 0
                clear_somewhere = True
                yield from self.shelf_base_candidates(
                    hand=hand,
                    pres=pres,
                    base=base,
                    grasp_conf=grasp_conf,
                    cube=cube,
                    refused=refused,
                )
        for hand, pres, start in given_up:
            for i in range(start, len(bases)):
                if i not in drivable:
                    drivable[i] = self.base_drivable(base=bases[i])
                if not drivable[i]:
                    continue
                grasp_conf = self.shelf_grasp_conf(hand=hand, base=bases[i], cube=cube)
                if grasp_conf is not None:
                    yield from self.shelf_base_candidates(
                        hand=hand,
                        pres=pres,
                        base=bases[i],
                        grasp_conf=grasp_conf,
                        cube=cube,
                        refused=refused,
                    )

    def shelf_base_candidates(
        self,
        *,
        hand: np.ndarray,
        pres: tuple[np.ndarray, ...],
        base: np.ndarray,
        grasp_conf: np.ndarray,
        cube: str,
        refused: list[str],
    ) -> Generator[ShelfCandidate, None, None]:
        """`shelf_candidates`' candidates for one grasp, clear from `base`: one per
        pre-grasp clear on the grasp's IK branch, then the way in from outside the front
        if the pads' way in from there is clear too. None whose hand touches another
        cube at the grasp or at the entry's first pose."""
        if self.touches_other_cube(base=base, arm=grasp_conf, cube=cube):
            return
        entries: list[tuple[tuple[np.ndarray, ...], np.ndarray]] = []
        for pre in pres:
            pre_conf = self.shelf_pre_conf(pre=pre, base=base, grasp_conf=grasp_conf, cube=cube)
            if pre_conf is not None:
                entries.append(((pre,), pre_conf))
        if entries:
            (first,), first_conf = entries[0]
            out, out_conf = self.shelf_exit(pre=first, pre_conf=first_conf, base=base, cube=cube)
            if out_conf is not None and not self.swept(
                path=[out, first, hand], cube=cube, refused=refused
            ):
                entries.append(((out, first), out_conf))
        for entry, conf in entries:
            if not self.touches_other_cube(base=base, arm=conf, cube=cube):
                yield (hand, entry, base, grasp_conf, conf)

    def swept(self, *, path: list[np.ndarray], cube: str, refused: list[str]) -> bool:
        """Whether `neighbour_in_sweep` finds a cube along `path`; if so, it is added
        to `refused`."""
        other = self.neighbour_in_sweep(path=path, cube=cube)
        if other is not None:
            refused.append(other)
        return other is not None

    def neighbour_in_sweep(self, *, path: list[np.ndarray], cube: str) -> str | None:
        """Another cube within `SHELF_PICK_NEIGHBOUR_CLEARANCE` of the space the pads
        pass through (`PadSweep`) on the way in along `path`, pinch-site poses from
        the first pre-grasp to the grasp, and as they close at its last pose."""
        travel = self.pad_travel()
        sweep = PadSweep.descent(path=path, pads=travel[0]) + PadSweep.closing(
            hand=path[-1], travel=travel
        )
        others = {}
        for i in range(len(O.cubes)):
            other = f"cube_{i}"
            if other != cube:
                pose = self.cube_pose(cube=other)
                others[other] = (pose[:3, 3], pose[:3, :3], np.full(3, O.cube_half_size))
        return PadSweep.first_hit(
            sweep=sweep, others=others, clearance=SHELF_PICK_NEIGHBOUR_CLEARANCE
        )

    def pad_travel(self) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """Each side's pads as (low, high) corners in the pinch site's frame, at stages
        `PAD_TRAVEL_STEP` apart from fully open until they would pass the face of the
        2 cm cube: posed on the gripper's own kinematics, its linkage joints at that
        fraction of `GRIPPER_CLOSED`."""
        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        out: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for t in np.arange(0.0, 1.0 + 1e-9, PAD_TRAVEL_STEP):
            d.qpos[:] = shelf.data.qpos
            for side in ("left", "right"):
                for joint, closed in GRIPPER_CLOSED.items():
                    d.qpos[m.jnt_qposadr[m.joint(f"robot_{side}_{joint}").id]] = t * closed
            shelf.mujoco.mj_kinematics(m, d)
            site = Frames.pose(
                rotation=d.site_xmat[shelf.site].reshape(3, 3), position=d.site_xpos[shelf.site]
            )
            here = np.linalg.inv(site)
            pads = []
            for side_geoms in shelf.pad_sides:
                corners = []
                for g in side_geoms:
                    rotation = here[:3, :3] @ d.geom_xmat[g].reshape(3, 3)
                    centre = (here @ np.r_[d.geom_xpos[g], 1.0])[:3]
                    corners.append(centre + (rotation @ (signs * m.geom_size[g]).T).T)
                stacked = np.concatenate(corners)
                pads.append((stacked.min(axis=0), stacked.max(axis=0)))
            if out and min(min(abs(lo[0]), abs(hi[0])) for lo, hi in pads) < O.cube_half_size:
                break
            out.append(pads)
        return out

    @staticmethod
    def fixed_ik(*, conf: np.ndarray) -> Callable[..., list[float]]:
        """Stands in for upstream's `inverse_kinematics(robot, pose, ...)`: the arm
        configuration already solved and checked in MuJoCo, fingers as they are."""
        return lambda robot, *_, **__: [*conf.tolist(), *list(robot.get_joint_positions())[7:]]

    def shelf_hands(
        self, *, cube: str, centre: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[np.ndarray, ...]]]:
        """(grasp, pre-grasps) pinch-site poses for a cube on the shelf, in the order
        they are tried.

        Each grasp has the pads on two side faces and the approach leaning
        `CUBE_LEANS_DEG` from vertical, the wrist out of the cupboard's open front
        (`ShelfSkills.cube_grasp`): the approach leans along a face normal, so the pads
        close on the opposite pair of faces -- never across the cube at an angle to them
        (Josh, 2026-10-15). Last, grasps whose wrist leans sideways along the shelf
        (`SHELF_PICK_MIN_SIDEWAYS`).

        Each grasp has three pre-grasps: backed `PRE_GRASP_RISE` out along the
        approach, then `SHELF_PICK_FAR_BACK` out along it, where the way to the nearer
        one is too narrow to plan (cubes at the back of the shelf), then straight
        above. Raised straight up, a steeply leaned wrist meets the board above in
        KINDER's planning scene, whose boards sit 6.35 mm lower than MuJoCo's."""
        quaternion = self.shelf.core.get_object(cube).get_pose()[1]
        axes = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        normal = np.arctan2(axes[1, 0], axes[0, 0])
        faces = [normal + k * np.pi / 2 for k in range(4)]
        out: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
        seen: list[float] = []
        for outward in (SHELF_PICK_MIN_OUTWARD, SHELF_PICK_MIN_SIDEWAYS):
            for lean in CUBE_LEANS_DEG:
                for face in faces:
                    toward = np.array([np.cos(face), np.sin(face), 0.0])
                    if toward[0] <= outward:
                        continue
                    key = float(np.rad2deg(face) % 360.0) + 1000.0 * lean
                    if any(abs(key - k) < 1.0 for k in seen):
                        continue
                    seen.append(key)
                    closing = np.array([toward[1], -toward[0], 0.0])
                    rotation = Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean)
                    hand = Frames.pose(
                        rotation=rotation,
                        position=np.asarray(centre) - CUBE_STANDOFF * rotation[:, 2],
                    )
                    back = hand.copy()
                    back[:3, 3] -= PRE_GRASP_RISE * rotation[:, 2]
                    above = hand.copy()
                    above[2, 3] += PRE_GRASP_RISE
                    far = hand.copy()
                    far[:3, 3] -= SHELF_PICK_FAR_BACK * rotation[:, 2]
                    out.append((hand, (back, far, above)))
        return out

    @staticmethod
    def shelf_pick_bases(*, centre: np.ndarray) -> list[np.ndarray]:
        """Base poses (x, y, heading) at `SHELF_PICK_BASE_XS` and cube y +
        `SHELF_PICK_Y_OFFSETS`, each facing the cube, as upstream's parameters put it."""
        out = []
        for base_x in SHELF_PICK_BASE_XS:
            for dy in SHELF_PICK_Y_OFFSETS:
                y = float(centre[1]) + dy
                heading = float(np.arctan2(float(centre[1]) - y, float(centre[0]) - base_x))
                out.append(np.array([base_x, y, heading]))
        return out

    def facing_params(
        self, *, base: np.ndarray, centre: np.ndarray, cube: str
    ) -> tuple[float, float]:
        """`PickShelfController`'s (distance, rotation) for `base`: its rotation is
        relative to the cube's own yaw."""
        back = float(centre[0] - base[0])
        side = float(centre[1] - base[1])
        return float(np.hypot(back, side)), float(np.arctan2(side, back) - self.yaw(name=cube))

    def base_drivable(self, *, base: np.ndarray) -> bool:
        """Whether upstream's own base planner (`run_base_motion_planning`, over the
        scene's overhead footprints, as `PickShelfController.reset` calls it) reaches
        `base`: it fails the same way for every grasp tried from that base. What the
        robot carries is not an obstacle (`robot_borne`), as in `drive`."""
        from kinder_models.dynamic3d.shelf.parameterized_skills import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            run_base_motion_planning,
        )
        from spatialmath import SE2

        plan = run_base_motion_planning(
            state=self.state(),
            target_base_pose=SE2(float(base[0]), float(base[1]), float(base[2])),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=self.robot_borne(),
        )
        return plan is not None

    def base_clear(self, *, base: np.ndarray) -> bool:
        """Whether the robot, at `base` with the arm at `HOME_ARM`, touches nothing but
        the floor: a wiper dropped on the floor where the base would stand rules that
        base out for every grasp, not just the first few tried from it."""
        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        d.qpos[:] = shelf.data.qpos
        d.qpos[shelf.ba] = base
        d.qpos[shelf.qa] = HOME_ARM
        shelf.mujoco.mj_kinematics(m, d)
        shelf.mujoco.mj_collision(m, d)
        for contact in d.contact[: d.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            for mine, other in (pair, pair[::-1]):
                if (
                    mine in shelf.robot_geoms
                    and other not in shelf.robot_geoms
                    and m.geom_bodyid[other] != 0
                    and contact.dist < 0
                ):
                    return False
        return True

    def shelf_grasp_conf(
        self, *, hand: np.ndarray, base: np.ndarray, cube: str
    ) -> np.ndarray | None:
        """The arm configuration at the grasp `hand` from `base`, solved on MuJoCo's own
        kinematics and clear of the cupboard and the box; None if
        there is none.

        KINDER solves one IK in PyBullet and plans to whatever it returns: on cubes by
        a side panel or at the back of the shelf, that one solution put the wrist into
        the board above or a finger into the panel from every base, while other
        solutions to the same hand pose are clear."""
        shelf = self.shelf
        try:
            conf, collision = shelf.ik(
                position=hand[:3, 3], rotation=hand[:3, :3], base=base, seed=HOME_ARM
            )
        except SkillFailed:
            return None
        if collision > 0:
            return None
        return self.wrapped(arm=conf)

    def cube_between_pads(self, *, base: np.ndarray, arm: np.ndarray, cube: str) -> str | None:
        """Another cube that lies where the pads close, at this base and arm: the jaws
        would squeeze it with the target or shove it into it. Seen on human reset
        layouts with cubes 3 cm apart, where the close knocked the target 9 cm along
        the shelf and stacked it on its neighbour."""
        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        half_gap = self.jaw_half_gap()
        d.qpos[:] = shelf.data.qpos
        d.qpos[shelf.ba] = base
        d.qpos[shelf.qa] = arm
        shelf.mujoco.mj_kinematics(m, d)
        site = Frames.pose(
            rotation=d.site_xmat[shelf.site].reshape(3, 3), position=d.site_xpos[shelf.site]
        )
        to_site = np.linalg.inv(site)
        half = O.cube_half_size
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for i in range(5):
            other = f"cube_{i}"
            if other == cube:
                continue
            pose = self.cube_pose(cube=other)
            points = (pose[:3, :3] @ (half * corners).T).T + pose[:3, 3]
            local = (to_site[:3, :3] @ points.T).T + to_site[:3, 3]
            inside = (
                (np.abs(local[:, 0]) < half_gap)
                & (np.abs(local[:, 1]) < JAW_HALF_WIDTH)
                & (local[:, 2] > JAW_DEPTH[0])
                & (local[:, 2] < JAW_DEPTH[1])
            )
            if inside.any():
                return other
        return None

    def jaw_half_gap(self) -> float:
        """Half the distance between the pads as they are now, across the closing axis."""
        shelf = self.shelf
        here = np.linalg.inv(shelf.site_pose())
        pads = [
            np.mean([(here @ np.r_[shelf.data.geom_xpos[g], 1.0])[:3] for g in side], axis=0)
            for side in shelf.pad_sides
        ]
        return 0.5 * abs(float(pads[0][0] - pads[1][0]))

    def cupboard_between_pads(self, *, hand: np.ndarray, cube: str) -> str | None:
        """The cupboard geom (side panel, back, a board other than the one the cube
        rests on) inside the space between the open pads at grasp `hand`, from the site
        to the cube's centre (`CUBE_STANDOFF` ahead): the pads would close on it, not
        the cube. A cube 6 mm from a side panel, grasped across the panel's normal, put
        the open outer pad outside the 1 cm panel; the jaws clamped the panel and the
        lift stalled on it. The board under the cube is left out: the pads close above
        it, and `shelf_grasp_conf` already refuses a pad that touches it."""
        shelf = self.shelf
        m, d = shelf.model, shelf.data
        jaw_half = np.array([self.jaw_half_gap(), JAW_HALF_WIDTH, 0.5 * CUBE_STANDOFF])
        jaw_centre = hand[:3, 3] + 0.5 * CUBE_STANDOFF * hand[:3, 2]
        bottom = float(shelf.cube_position(cube=cube)[2]) - O.cube_half_size
        for g in sorted(shelf.cupboard_geoms):
            if m.geom_type[g] != shelf.mujoco.mjtGeom.mjGEOM_BOX:
                continue
            rotation = d.geom_xmat[g].reshape(3, 3)
            half = m.geom_size[g]
            if float(d.geom_xpos[g][2] + np.abs(rotation[2]) @ half) <= bottom + SUPPORT_SLACK:
                continue
            if self.boxes_overlap(
                first=(jaw_centre, hand[:3, :3], jaw_half),
                second=(d.geom_xpos[g], rotation, half),
            ):
                return str(m.geom(g).name)
        return None

    @staticmethod
    def boxes_overlap(
        *,
        first: tuple[np.ndarray, np.ndarray, np.ndarray],
        second: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> bool:
        """Whether two oriented boxes, each (centre, rotation, half-extents), intersect:
        the separating-axis test over both boxes' face normals and their edge crossings."""
        (ca, ra, ha), (cb, rb, hb) = first, second
        axes = [*ra.T, *rb.T]
        axes += [np.cross(a, b) for a in ra.T for b in rb.T]
        gap = np.asarray(cb) - np.asarray(ca)
        for axis in axes:
            if np.linalg.norm(axis) < 1e-9:
                continue
            reach = np.abs(ra.T @ axis) @ ha + np.abs(rb.T @ axis) @ hb
            if abs(float(gap @ axis)) > float(reach):
                return False
        return True

    def cube_in_jaws(self, *, cube: str) -> bool:
        """Whether the cube's centre lies between the pads, as the hand is now."""
        shelf = self.shelf
        local = np.linalg.inv(shelf.site_pose()) @ np.r_[shelf.cube_position(cube=cube), 1.0]
        return bool(
            abs(local[0]) < self.jaw_half_gap()
            and abs(local[1]) < JAW_HALF_WIDTH
            and JAW_DEPTH[0] < local[2] < JAW_DEPTH[1]
        )

    def shelf_pre_conf(
        self, *, pre: np.ndarray, base: np.ndarray, grasp_conf: np.ndarray, cube: str
    ) -> np.ndarray | None:
        """The arm configuration at the pre-grasp `pre`, on the grasp's IK branch and
        clear as `shelf_grasp_conf` checks; None if there is none."""
        shelf = self.shelf
        try:
            conf, collision = shelf.ik(
                position=pre[:3, 3], rotation=pre[:3, :3], base=base, seed=grasp_conf, attempts=2
            )
        except SkillFailed:
            return None
        jump = np.max(np.abs((conf - grasp_conf + np.pi) % (2 * np.pi) - np.pi))
        if collision > 0 or jump > SHELF_PICK_MAX_JUMP:
            return None
        return self.wrapped(arm=conf)

    def wrapped(self, *, arm: np.ndarray) -> np.ndarray:
        """Unlimited joints brought into (-pi, pi], as PyBullet reports them."""
        shelf = self.shelf
        out = np.array(arm, dtype=float)
        for i, joint in enumerate(shelf.joints):
            if not shelf.model.jnt_limited[joint]:
                out[i] = (out[i] + np.pi) % (2 * np.pi) - np.pi
        return out

    def touches_other_cube(self, *, base: np.ndarray, arm: np.ndarray, cube: str) -> bool:
        """Whether the robot, at this base and arm, touches a cube other than `cube`."""
        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        d.qpos[:] = shelf.data.qpos
        d.qpos[shelf.ba] = base
        d.qpos[shelf.qa] = arm
        shelf.mujoco.mj_kinematics(m, d)
        shelf.mujoco.mj_collision(m, d)
        for contact in d.contact[: d.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            for mine, other in (pair, pair[::-1]):
                if mine in shelf.robot_geoms and contact.dist < 0:
                    name = str(m.body(m.geom_bodyid[other]).name)
                    if name.startswith("cube_") and name != cube:
                        return True
        return False

    def place(self, *, cube: str, hand_targets: list[np.ndarray], label: str) -> None:
        """`PlaceShelfController` releasing the held cube with the hand at each target
        in turn (the first that plans), from base poses `PLACE_REACHES` behind it."""
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills
        from pybullet_helpers.geometry import Pose

        orientation = shelf_skills.ARM_MOVEMENT_CUPBOARD.orientation
        robot, target, cupboard = self.objects("robot", cube, CUPBOARD)
        controller = KinderControllers.get(name="PlaceShelf")
        for hand in hand_targets:
            for reach in PLACE_REACHES:
                base_x = float(hand[0]) - 0.12 - reach
                for dy in PLACE_Y_OFFSETS:
                    params = self.base_params(base_xy=(base_x, float(hand[1]) + dy))
                    arm_target = Pose((reach, -dy, float(hand[2]) - 0.4), orientation)
                    sim = self.planning_sim()
                    sim.base_link_to_held_obj = self.held_transform(sim=sim, cube=cube)
                    with self.constant(
                        module=shelf_skills, name="ARM_MOVEMENT_CUPBOARD", value=arm_target
                    ):
                        outcome = self.drive(
                            controller=controller(
                                objects=[robot, target, cupboard], pybullet_sim=sim
                            ),
                            params=params,
                            label=label,
                        )
                    if outcome == "done":
                        return
        raise SkillFailed(f"{label}: no placement planned")

    def place_cube_in_box(self, *, cube: str) -> None:
        if cube not in self.held_cubes:
            raise SkillFailed(f"place_cube_in_box: not holding {cube}")
        dx, dy = self.held_cube_offset(cube=cube)
        self.place(
            cube=cube,
            hand_targets=[
                np.array([x - dx, y - dy, BOX_RELEASE_Z]) for x, y in self.box_drop_points()
            ],
            label="place_in_box",
        )
        for _ in range(15):
            self.env_step(action=FollowPlans._hold(grip=0.0), label="settle")  # noqa: SLF001
        if not self.core.cube_in_box(cube):
            raise SkillFailed(f"place_cube_in_box: {cube} not in the box")

    def box_drop_points(self) -> list[tuple[float, float]]:
        """Where a held cube should land in the box, best first: the grid points of the
        drop region (`BOX_DROP_X_MARGINS`, `BOX_DROP_Y_MARGIN` inside the inner walls),
        ranked by their distance to the nearest cube already in the box, farthest first
        (ties: nearest the region's centre, then smallest x, y), taking up to
        `BOX_DROP_TARGETS` that are `BOX_DROP_TARGET_SPACING` apart. Deterministic in the
        cubes' positions."""
        (x0, y0, x1, y1), _ = self.core.box_interior()
        near, far = BOX_DROP_X_MARGINS
        xs = self.grid(low=x0 + near, high=x1 - far, step=BOX_DROP_GRID)
        ys = self.grid(low=y0 + BOX_DROP_Y_MARGIN, high=y1 - BOX_DROP_Y_MARGIN, step=BOX_DROP_GRID)
        centre = np.array([(x0 + near + x1 - far) / 2, (y0 + y1) / 2])
        if not xs or not ys:
            return [(float(centre[0]), float(centre[1]))]
        inside = [
            np.asarray(self.core.get_object(c).get_pose()[0][:2], dtype=float)
            for c in (f"cube_{i}" for i in range(5))
            if c not in self.held_cubes and self.core.cube_in_box(c)
        ]

        def rank(*, point: tuple[float, float]) -> tuple[float, float, float, float]:
            xy = np.asarray(point)
            free = min((float(np.linalg.norm(xy - c)) for c in inside), default=np.inf)
            return (-round(free, 6), round(float(np.linalg.norm(xy - centre)), 6), *point)

        out: list[tuple[float, float]] = []
        points = [(x, y) for x in xs for y in ys]
        for point in sorted(points, key=lambda p: rank(point=p)):
            if all(np.hypot(point[0] - a, point[1] - b) >= BOX_DROP_TARGET_SPACING for a, b in out):
                out.append(point)
            if len(out) == BOX_DROP_TARGETS:
                break
        return out

    def return_cube_to_shelf(self, *, cube: str) -> None:
        """Pick the cube from the box or the floor and drop it onto the cube shelf from
        `SHELF_DROP_RISES` above the board: both `ReturnCubeToShelf*` planner skills."""
        from .floor_pick import FloorPicks  # it imports this module

        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("return_cube_to_shelf: the hand is not empty")
        FloorPicks(kinder=self).pick(cube=cube, label="return_pick")
        if not self.held(name=cube):
            raise SkillFailed(f"return_cube_to_shelf: {cube} did not come with the hand")
        points = self.shelf_drop_points()
        if not points:
            raise SkillFailed("return_cube_to_shelf: no free spot on the cube shelf")
        board = float(self.shelf.board[CUBE_SHELF])
        dx, dy = self.held_cube_offset(cube=cube)
        self.place(
            cube=cube,
            hand_targets=[
                np.array([x - dx, y - dy, board + rise])
                for x, y in points
                for rise in SHELF_DROP_RISES
            ],
            label="return_place",
        )
        if ("OnShelf", (cube,)) not in self.backend.abstract_atoms():
            raise SkillFailed(f"return_cube_to_shelf: {cube} is not resting on the cube shelf")

    def held_cube_offset(self, *, cube: str) -> tuple[float, float]:
        """Where the held cube hangs (m, world xy) from the hand target `place` drives
        the end effector to: the measured end-effector-to-cube transform, turned by the
        placement's fixed hand orientation. A cube picked off the floor hangs ~46 mm
        ahead of it (the fixed hand target x = 1.45 landed cubes at x = 1.496), and one
        picked off the shelf ~47 mm: every cube dropped at the box's centre landed ~5 cm
        toward its far wall. Used by both the shelf return and the box drop."""
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills

        held = self.held_transform(sim=self.planning_sim(), cube=cube)
        orientation = shelf_skills.ARM_MOVEMENT_CUPBOARD.orientation
        offset = Rotation.from_quat(orientation).apply(np.asarray(held.position, dtype=float))
        return float(offset[0]), float(offset[1])

    def shelf_region(self) -> tuple[float, float, float, float]:
        """KINDER's `cubes_init_region` in world (x0, y0, x1, y1): its ranges, in the
        cupboard's frame, turned by the cupboard's yaw and moved to its position."""
        cupboard = self.core._fixtures_dict[CUPBOARD]  # noqa: SLF001
        x0, y0, _, x1, y1, _ = cupboard.regions[CUBE_REGION]["ranges"][0]
        yaw = float(cupboard.yaw)
        turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)]) @ turn.T
        corners += np.asarray(cupboard.position, dtype=float)[:2]
        low, high = corners.min(axis=0), corners.max(axis=0)
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    def shelf_cube_xys(self) -> list[np.ndarray]:
        """Where (world xy) each cube lies on the cube shelf, the held cube aside:
        anything inside the cupboard between the cube board and the board above."""
        shelf = self.shelf
        out = []
        for name in (f"cube_{i}" for i in range(5)):
            if self.held(name=name):
                continue
            x, y, z = shelf.cube_position(cube=name)
            if x > shelf.front_x and shelf.board[CUBE_SHELF] < z < shelf.board[WIPER_SHELF]:
                out.append(np.array([x, y]))
        return out

    def shelf_drop_points(self) -> list[tuple[float, float]]:
        """Where a held cube should land on the cube shelf, best first: the grid points
        of the drop region (`shelf_region`, inset by `SHELF_DROP_X_MARGINS` and
        `SHELF_DROP_Y_MARGIN`) at least `SHELF_DROP_CLEARANCE` from every cube on the
        shelf, ranked by their distance to the nearest of them, farthest first (ties:
        nearest the region's centre, then smallest x, y), taking up to
        `SHELF_DROP_TARGETS` that are `SHELF_DROP_TARGET_SPACING` apart. Deterministic
        in the cubes' positions; empty when no point is clear."""
        x0, y0, x1, y1 = self.shelf_region()
        near, far = SHELF_DROP_X_MARGINS
        x0, x1 = x0 + near, x1 - far
        y0, y1 = y0 + SHELF_DROP_Y_MARGIN, y1 - SHELF_DROP_Y_MARGIN
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        xs, ys = self.grid(low=x0, high=x1), self.grid(low=y0, high=y1)
        if not xs or not ys:
            return []
        cubes = self.shelf_cube_xys()

        def free(*, point: tuple[float, float]) -> float:
            xy = np.asarray(point)
            return min((float(np.linalg.norm(xy - c)) for c in cubes), default=np.inf)

        def rank(*, point: tuple[float, float]) -> tuple[float, float, float, float]:
            near_centre = round(float(np.linalg.norm(np.asarray(point) - centre)), 6)
            return (-round(free(point=point), 6), near_centre, *point)

        out: list[tuple[float, float]] = []
        points = [(x, y) for x in xs for y in ys if free(point=(x, y)) >= SHELF_DROP_CLEARANCE]
        for point in sorted(points, key=lambda p: rank(point=p)):
            if all(
                np.hypot(point[0] - a, point[1] - b) >= SHELF_DROP_TARGET_SPACING for a, b in out
            ):
                out.append(point)
            if len(out) == SHELF_DROP_TARGETS:
                break
        return out

    @staticmethod
    def grid(*, low: float, high: float, step: float = SHELF_DROP_GRID) -> list[float]:
        """`step`-spaced values from `low` to `high`, centred in that range: the shelf
        drop region's (`SHELF_DROP_GRID`, the default) and the box's (`BOX_DROP_GRID`)."""
        if high < low:
            return []
        count = int(np.floor((high - low) / step + 1e-9))
        start = (low + high) / 2 - count * step / 2
        return [float(start + i * step) for i in range(count + 1)]

    def sweep_shelf(self, *, height: float, speed: float) -> None:
        if self.shelf.in_hand is None:
            raise SkillFailed("sweep_shelf: the wiper is not in hand")
        self.shelf.sweep_shelf(height=height, speed=speed)

    def return_wiper(self) -> None:
        """`ShelfSkills.return_wiper`, except that once the wiper is back on its shelf
        and out of the hand, a withdrawal that cannot finish its straight-line exit is
        not a failure: the arm retract that follows every skill takes it out."""
        shelf = self.shelf
        if shelf.in_hand is None:
            raise SkillFailed("return_wiper: the wiper is not in hand")
        try:
            shelf.return_wiper()
        except SkillFailed as exc:
            home = float(np.linalg.norm(shelf.wiper_pose()[:3, 3] - shelf.wiper_start[:3, 3]))
            lifted = abs(shelf.wiper_lift()) > 0.01
            if "withdraw" not in str(exc) or self.held(name=WIPER) or home > 0.05 or lifted:
                raise
            self.note(
                stage="return_wiper", kind="scripted", reason=f"wiper home ({home:.3f} m); {exc}"
            )


class PadSweep:
    """The space the gripper's pads pass through on a pick, as oriented boxes: the open
    pads at each pose of the descent to the grasp, and the pads at each stage of their
    travel as they close there.

    Pads are given as one (low, high) corner pair per side, axis-aligned in the pinch
    site's frame (x the closing axis, z the approach). A travel is a list of those,
    open first (`KinderSkills.pad_travel`): the 2F-85's pads close on an arc, riding
    forward 13 mm along the approach as they come in, so no one box bounds them
    without taking in space they never reach."""

    @staticmethod
    def closing(
        *, hand: np.ndarray, travel: list[list[tuple[np.ndarray, np.ndarray]]]
    ) -> list[OrientedBox]:
        """The pads at every stage of `travel`, at grasp `hand`."""
        return [
            PadSweep.placed(hand=hand, low=np.asarray(low), high=np.asarray(high))
            for pads in travel
            for low, high in pads
        ]

    @staticmethod
    def descent(
        *, path: list[np.ndarray], pads: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[OrientedBox]:
        """The open pads at hand poses along `path` (pre-grasp first, grasp last),
        straight lines between them, at most `SWEEP_STEP` apart."""
        poses = [path[0]]
        for start, end in itertools.pairwise(path):
            poses += Frames.line(start=start, end=end, step=SWEEP_STEP)
        return [
            PadSweep.placed(hand=pose, low=low, high=high) for pose in poses for low, high in pads
        ]

    @staticmethod
    def placed(*, hand: np.ndarray, low: np.ndarray, high: np.ndarray) -> OrientedBox:
        """The site-frame box (low, high) at hand pose `hand`, in the world."""
        centre = hand[:3, :3] @ (0.5 * (low + high)) + hand[:3, 3]
        return centre, hand[:3, :3], 0.5 * (high - low)

    @staticmethod
    def first_hit(
        *, sweep: list[OrientedBox], others: dict[str, OrientedBox], clearance: float
    ) -> str | None:
        """The first of `others` that comes within `clearance` of any box in `sweep`."""
        for name, (centre, rotation, half) in others.items():
            grown = (centre, rotation, np.asarray(half) + clearance)
            if any(KinderSkills.boxes_overlap(first=box, second=grown) for box in sweep):
                return name
        return None


class PlanFollower:
    """Tracks a joint plan waypoint by waypoint: through each within PASS_TOLERANCE,
    onto the last within `final`. A blocked arm stops trying -- at the last waypoint
    after FINAL_STEP_LIMIT steps, anywhere after FOLLOW_STEP_LIMIT per waypoint --
    and whether the skill then did its job is the scene's to say, not the follower's."""

    def __init__(
        self, *, plan: list[np.ndarray], final: float = FINAL_TOLERANCE, gain: float = 2.0
    ) -> None:
        self.plan = plan
        self.final = final
        self.gain = gain
        self.index = 0
        self.count = 0

    def error(self, *, at: int, current: np.ndarray) -> np.ndarray:
        return ((self.plan[at] - current) + np.pi) % (2 * np.pi) - np.pi

    def action(self, *, current: np.ndarray, grip: float) -> np.ndarray | None:
        while (
            self.index < len(self.plan) - 1
            and np.max(np.abs(self.error(at=self.index, current=current))) < PASS_TOLERANCE
        ):
            self.index += 1
        self.count += 1
        e = self.error(at=self.index, current=current)
        last = self.index == len(self.plan) - 1
        if (last and (np.max(np.abs(e)) < self.final or self.count > FINAL_STEP_LIMIT)) or (
            self.count > FOLLOW_STEP_LIMIT * len(self.plan)
        ):
            return None
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = self.gain * e
        action[-1] = grip
        return action
