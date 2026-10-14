"""SweepOffShelf3D's six robot skills on KINDER's own controllers.

Each skill runs to completion or raises `SkillFailed`, and every one, failed or not,
ends with the arm retracted: to KINDER's tucked home configuration (`HOME_ARM`), or,
holding the wiper, straight up to the carry height. So no skill starts where the last
one gave up -- the absorbing failure behind CCCC's PickWiper tally (`KinderSkills.run`).

Where each controller comes from:

- `pick_cube_from_floor`: `tossing.PickCubeController` as it is upstream, with this
  scene's box added to its planning scene.
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
     (`KinderSkills.plan_blocked`): PyBullet's cupboard and arm are not MuJoCo's, and
     its scene has no wiper;
  6. the shelf pick's held-cube retract allows upstream's 1 mm resting contact
     (`HELD_ON_SUPPORT_THRESHOLD`), and the box is in every planning scene.
- `sweep_shelf`, `return_wiper`: KINDER has no controller for either (its
  `SweepOriController` sweeps a floor-standing target, 0/15 here), so they are this
  package's, built on `ShelfSkills`' routines, starting from the wiper as it is actually
  held, whichever skill picked it up.

Nothing here imports KINDER at module scope: `kinder_models` is imported on first use.
"""

import contextlib
from collections.abc import Callable, Generator
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

from .scripted_skills import CARRY_Z, ShelfSkills, SkillFailed, StepBudgetExhausted

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
SHELF_PLACE_XY = (1.45, 0.0)
BOX_RELEASE_Z = 0.32
PLACE_REACHES = (0.8, 0.7, 0.6)
PLACE_Y_OFFSETS = (0.0, 0.1, -0.1)
PICK_WIPER_DISTANCES = (0.7, 0.6, 0.8, 0.55)
# Base x positions for a shelf pick: as far back as the box lets it, and nearer.
SHELF_PICK_BASE_XS = (0.8, 0.72, 0.64)
SHELF_PICK_GRASPS = 6
SHELF_PICK_Y_OFFSETS = (0.0, 0.1, -0.1)
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

    def _follow(
        self, *, key: str, plan: list[Any], final: float = FINAL_TOLERANCE
    ) -> np.ndarray | None:
        follower = self._followers.get(key)
        if follower is None:
            follower = PlanFollower(plan=[np.array(q[:7], dtype=float) for q in plan], final=final)
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
        pre_conf = inverse_kinematics(sim.robot, above, set_joints=False)
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
    """Runs the six planner skills on one live scene, sharing `ShelfSkills`' step
    counter, budget, gripper command and recorder, so a KINDER step and a scripted
    step are counted, capped and filmed the same way."""

    def __init__(self, *, shelf: ShelfSkills, backend: Any) -> None:
        self.shelf = shelf
        self.backend = backend
        self.env = shelf.env
        self.core = shelf.core
        self.notes: list[str] = []

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
                shelf.grasp_kind = self.nearest_wiper_grasp()
            shelf.grip = 1.0
        else:
            shelf.in_hand, shelf.grasp_kind = None, None
        held_cubes = [c for c in (f"cube_{i}" for i in range(5)) if self.held(name=c)]
        shelf.held_cube = held_cubes[0] if held_cubes else None
        if shelf.held_cube is not None:
            shelf.grip = 1.0
        if shelf.in_hand is None and shelf.held_cube is None:
            shelf.grip = 0.0

    def nearest_wiper_grasp(self) -> str | None:
        """The fixed wiper grasp ("shelf" or "sweep") the hand is at, if any."""
        shelf = self.shelf
        hand_in_wiper = np.linalg.inv(shelf.wiper_pose()) @ shelf.site_pose()
        for kind, nominal in shelf.grasp_in_wiper.items():
            offset = float(np.linalg.norm(hand_in_wiper[:3, 3] - nominal[:3, 3]))
            turn = Rotation.from_matrix(hand_in_wiper[:3, :3] @ nominal[:3, :3].T).magnitude()
            if offset < 0.02 and np.rad2deg(turn) < 10.0:
                return kind
        return None

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
            controller.reset(self.state(), params)
        except (TrajectorySamplingFailure, AssertionError, ValueError, RuntimeError) as exc:
            self.notes.append(f"{label}: no plan for {np.round(params, 3).tolist()}: {exc!r}"[:200])
            return "planned-out"
        except Exception as exc:  # noqa: BLE001  (IK and planner errors carry many types)
            self.notes.append(f"{label}: no plan ({type(exc).__name__}: {exc})"[:200])
            return "planned-out"
        blocked = self.plan_blocked(controller=controller)
        if blocked:
            self.notes.append(f"{label}: plan for {np.round(params, 3).tolist()} {blocked}"[:200])
            return "planned-out"
        try:
            while not controller.terminated():
                self.env_step(action=controller.step(), label=label)
                controller.observe(self.state())
        except SkillFailed:
            raise
        except (TrajectorySamplingFailure, Exception) as exc:
            raise SkillFailed(f"{label}: execution failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self.sync()
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
        for name in ("_current_arm_joint_plan", "_current_retract_plan"):
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
                    if name == "_current_arm_joint_plan" and (hit := self.hand_hits(d=d)):
                        return f"approach plan puts the hand into {hit}"
        return None

    def hand_hits(self, *, d: Any) -> str | None:
        """What the gripper itself touches in `d` on the way in (contacts already
        computed by `arm_touches`): the wiper, which KINDER's planning scene does not
        have, or the cupboard or box beyond `HAND_ALLOWANCE` -- PyBullet's side panels
        sit a centimetre outside MuJoCo's. Only the approach is checked so: the retract
        starts at the grasp, where the pads are meant to be against things."""
        shelf = self.shelf
        wiper = set(shelf.wiper_geoms)
        fixed = shelf.cupboard_geoms | shelf.box_geoms
        for contact in d.contact[: d.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & shelf.robot_geoms:
                continue
            if pair & wiper:
                return "the wiper (not in KINDER's scene)"
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

    def planning_sim(self) -> Any:
        """KINDER's PyBullet planning scene for the current state, with the box in it."""
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
        return sim

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
        from pybullet_helpers.motion_planning import run_motion_planning

        if self.arm_from_home() < RETRACT_TOLERANCE:
            return
        shelf = self.shelf
        if shelf.in_hand is not None:
            # Holding the wiper the arm goes no further than the scripted carry height:
            # at KINDER's home, the sweep grasp lets the wiper slide out of the pads.
            with contextlib.suppress(SkillFailed):
                shelf.raise_hand(z=CARRY_Z, label=f"{label}_raise", holding=True)
            return
        sim = self.planning_sim()
        x = self.state()
        sim.set_state(x)
        held = None
        if shelf.held_cube is not None:
            sim.base_link_to_held_obj = self.held_transform(sim=sim, cube=shelf.held_cube)
            held = sim._cubes[shelf.held_cube]  # noqa: SLF001
            sim.set_state(x, x.get_object_from_name(shelf.held_cube))
        plan = run_motion_planning(
            sim.robot,
            sim.get_robot_joints(),
            [*HOME_ARM.tolist(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            collision_bodies=sim.get_collision_bodies(held_object=held),
            held_object=held,
            base_link_to_held_obj=sim.base_link_to_held_obj if held is not None else None,
            seed=0,
            physics_client_id=sim.physics_client_id,
        )
        if plan is None:
            self.notes.append(f"{label}: no collision-free retract; arm left where it is")
            return
        confs = [np.array(q[:7]) for q in plan]
        if shelf.in_hand is not None and self.carried_wiper_blocked(confs=confs):
            self.notes.append(f"{label}: the carried wiper would hit something; arm left raised")
            return
        follower = PlanFollower(plan=confs)
        for _ in range(FINAL_STEP_LIMIT + 20 * len(plan)):
            action = follower.action(current=shelf.data.qpos[shelf.qa], grip=shelf.grip)
            if action is None:
                break
            self.env_step(action=action, label=label)
        self.sync()

    def carried_wiper_blocked(self, *, confs: list[np.ndarray]) -> bool:
        """Whether the held wiper, carried rigidly along these arm configurations with
        the grasp as measured, would cut into the cupboard, the box or the robot's own
        arm or base (KINDER's planning scene has no wiper in it)."""
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

    def settler(self, *, hand: np.ndarray, label: str) -> Callable[[], None]:
        """The descent from the pre-grasp (`PreGraspApproach`) to the pinch-site pose
        `hand`, straight down, on MuJoCo's own kinematics: KINDER solves IK in PyBullet,
        whose arm sits up to ~5 mm from MuJoCo's at the same joints -- enough to put a
        pad on top of a 15 mm wiper head or a 2 cm cube."""

        def settle() -> None:
            shelf = self.shelf
            shelf.base_target = shelf.data.qpos[shelf.ba].copy()
            shelf.arm_reference = shelf.data.qpos[shelf.qa].copy()
            above = hand.copy()
            above[2, 3] += PRE_GRASP_RISE
            with contextlib.suppress(SkillFailed):
                shelf.settle_at(pose=above, label=label)
                shelf.path(position=hand[:3, 3], rotation=hand[:3, :3], label=label, count=3)
            with contextlib.suppress(SkillFailed):
                shelf.settle_at(pose=hand, label=label)

        return settle

    def pick_wiper(self) -> None:
        import kinder_models.dynamic3d.sweep3D.parameterized_skills as sweep3d

        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_wiper: the hand is not empty")
        robot, wiper = self.objects("robot", WIPER)
        yaw = self.yaw(name=WIPER)
        candidates = [(0.7, -np.pi)]  # upstream's own fixed parameter first
        for heading in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            candidates += [(d, heading - yaw) for d in PICK_WIPER_DISTANCES]
        # In its slot only the leaned grasp clears the board above; off the shelf,
        # upstream's own grasp.
        grasp = sweep3d.WIPER_TRANSFORM_TO_OBJECT
        controller = KinderControllers.get(name="PickWiper")
        settle = None
        if ("WiperHome", (WIPER,)) in self.backend.abstract_atoms():
            hand = shelf.wiper_grasp(kind="shelf")
            grasp = self.pose_in(obj_pose=shelf.wiper_pose(), hand=hand)
            controller = KinderControllers.get(name="PickWiperInSlot")
            settle = self.settler(hand=hand, label="pick_wiper_settle")

        def make() -> Any:
            pick = controller(objects=[robot, wiper], pybullet_sim=self.planning_sim())
            pick.settle = settle
            return pick

        with self.constant(module=sweep3d, name="WIPER_TRANSFORM_TO_OBJECT", value=grasp):
            self.first_that_plans(
                make=make,
                candidates=candidates,
                label="pick_wiper",
            )
        if not self.held(name=WIPER):
            raise SkillFailed("pick_wiper: the wiper did not come with the hand")

    def pick_cube_from_floor(self, *, cube: str) -> None:
        """`tossing.PickCubeController` as it is upstream, the box in its scene."""
        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_cube_from_floor: the hand is not empty")
        robot, target = self.objects("robot", cube)
        controller = KinderControllers.get(name="PickCube")
        self.first_that_plans(
            make=lambda: controller(objects=[robot, target], pybullet_sim=self.planning_sim()),
            candidates=[np.zeros(0)],
            label="pick_cube_from_floor",
        )
        if not self.held(name=cube):
            raise SkillFailed(f"pick_cube_from_floor: {cube} did not come with the hand")

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills

        shelf = self.shelf
        if shelf.in_hand is not None or shelf.held_cube is not None:
            raise SkillFailed("pick_cube_from_shelf: the hand is not empty")
        robot, target = self.objects("robot", cube)
        centre = shelf.cube_position(cube=cube)
        obj_pose = self.cube_pose(cube=cube)
        controller = KinderControllers.get(name="PickShelf")
        # (distance, rotation) putting the base at (x, cube y + dy), facing the cube:
        # upstream's rotation parameter is relative to the cube's own yaw.
        yaw = self.yaw(name=cube)
        candidates = []
        for base_x in SHELF_PICK_BASE_XS:
            for dy in SHELF_PICK_Y_OFFSETS:
                back = float(centre[0]) - base_x
                candidates.append((float(np.hypot(back, dy)), float(np.arctan2(-dy, back) - yaw)))
        tried = 0
        planner = shelf_skills.run_motion_planning

        def held_on_support(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("held_object") is not None:
                kwargs.setdefault("distance_threshold", HELD_ON_SUPPORT_THRESHOLD)
            return planner(*args, **kwargs)

        for hand, _ in shelf.shelf_cube_pose(cube=cube, centre=centre)[:SHELF_PICK_GRASPS]:
            grasp = self.pose_in(obj_pose=obj_pose, hand=hand)
            with (
                self.constant(module=shelf_skills, name="GRASP_TRANSFORM_TO_OBJECT", value=grasp),
                self.constant(
                    module=shelf_skills, name="run_motion_planning", value=held_on_support
                ),
            ):
                for params in candidates:
                    pick = controller(objects=[robot, target], pybullet_sim=self.planning_sim())
                    pick.settle = self.settler(hand=hand, label="pick_cube_from_shelf_settle")
                    outcome = self.drive(
                        controller=pick, params=params, label="pick_cube_from_shelf"
                    )
                    if outcome == "done":
                        tried += 1
                        break
            if tried:
                break
        if not tried:
            raise SkillFailed(f"pick_cube_from_shelf: no grasp on {cube} planned")
        if not self.held(name=cube):
            raise SkillFailed(f"pick_cube_from_shelf: {cube} did not come with the hand")

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
        shelf = self.shelf
        if shelf.held_cube != cube:
            raise SkillFailed(f"place_cube_in_box: not holding {cube}")
        box = np.asarray(self.core.get_object(BOX).get_pose()[0], dtype=float)
        self.place(
            cube=cube,
            hand_targets=[np.array([box[0], box[1], BOX_RELEASE_Z])],
            label="place_in_box",
        )
        for _ in range(15):
            self.env_step(action=FollowPlans._hold(grip=0.0), label="settle")  # noqa: SLF001
        if not self.core.cube_in_box(cube):
            raise SkillFailed(f"place_cube_in_box: {cube} not in the box")

    def return_cube_to_shelf(self, *, cube: str) -> None:
        """Pick the cube from the box or the floor and drop it onto the cube shelf from
        `SHELF_DROP_RISES` above the board. Not a planner skill (the human returns the
        cubes); kept beside the others for the validation it was asked for."""
        robot, target = self.objects("robot", cube)
        controller = KinderControllers.get(name="PickCube")

        def make() -> Any:
            pick = controller(objects=[robot, target], pybullet_sim=self.planning_sim())
            if self.core.cube_in_box(cube):
                pick.HELD_OBJECT_COLLISION_THRESHOLD = pick.BIN_HELD_OBJECT_COLLISION_THRESHOLD
            return pick

        self.first_that_plans(make=make, candidates=[np.zeros(0)], label="return_pick")
        if not self.held(name=cube):
            raise SkillFailed(f"return_cube_to_shelf: {cube} did not come with the hand")
        board = float(self.shelf.board[1])
        self.place(
            cube=cube,
            hand_targets=[np.array([*SHELF_PLACE_XY, board + rise]) for rise in SHELF_DROP_RISES],
            label="return_place",
        )
        if ("OnShelf", (cube,)) not in self.backend.abstract_atoms():
            raise SkillFailed(f"return_cube_to_shelf: {cube} is not resting on the cube shelf")

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
            self.notes.append(f"return_wiper: wiper home ({home:.3f} m); {exc}")


class PlanFollower:
    """Tracks a joint plan waypoint by waypoint: through each within PASS_TOLERANCE,
    onto the last within `final`. A blocked arm stops trying -- at the last waypoint
    after FINAL_STEP_LIMIT steps, anywhere after FOLLOW_STEP_LIMIT per waypoint --
    and whether the skill then did its job is the scene's to say, not the follower's."""

    def __init__(self, *, plan: list[np.ndarray], final: float = FINAL_TOLERANCE) -> None:
        self.plan = plan
        self.final = final
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
        action[3:10] = 2.0 * e
        action[-1] = grip
        return action
