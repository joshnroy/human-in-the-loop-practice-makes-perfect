"""The floor and box picks: `PickCubeFromFloor` and the pick that starts both
`ReturnCubeToShelf*` skills, as a grasp x base candidate search over KINDER's planners.

Upstream's `tossing.PickCubeController` tries four (base, grasp) pairs: a vertical grasp
per cube yaw, the base 0.55 m behind it. Beside the box or the cupboard's side panel,
and inside the box, all four put the wrist into a wall. And where one does plan, its
grasp transform -- calibrated for Tossing3D's 5 cm cube, 3 cm along the approach --
puts this scene's 2 cm cube so deep between the pads that, as they close and ride
forward, their tips reach the floor: a pad is pushed back, the cube is pinched off
centre and works loose on the lift.

So, as the shelf pick does (`KinderSkills.pick_cube_from_shelf`), the grasp and the base
are this package's and the planning is KINDER's:

1. grasps on the cube as it lies, the pads on a pair of opposite side faces, the
   approach vertical or leaning `FLOOR_PICK_LEANS_DEG` along the other face normal, the
   pinch site as far back as keeps the closed pads' tips above the cube's support
   (`FloorPicks.standoff`);
2. bases on rings around the cube, sampled separately from the grasp, nearest the
   robot first;
3. each (grasp, base) solved on MuJoCo's own kinematics at the grasp, the pre-grasp and
   the end of the lift, and kept only if every robot geom stays upstream's
   `PickCubeController.GRASP_OBSTACLE_CLEARANCE` from the box, the cupboard and the other
   cubes there, with no other cube where the pads close (`KinderSkills.cube_between_pads`);
4. KINDER's `shelf.PickShelfController` plans the base path, the arm to the pre-grasp
   and the held-cube retract home, re-checked in MuJoCo (`KinderSkills.drive`); the
   descent to the grasp and the straight lift run on MuJoCo's kinematics
   (`KinderSkills.settler`, `KinderSkills.lifter`), and the retract carries the cube at
   the shelf pick's reduced gain;
5. closed loop: a cube that did not come with the hand is looked at again and replanned.

Nothing here imports KINDER at module scope.
"""

from collections.abc import Generator
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .kinder_skills import (
    BOX,
    GRIP_STEPS,
    HOME_ARM,
    SHELF_PICK_CARRY_GAIN,
    SHELF_PICK_MAX_JUMP,
    SHELF_PICK_PLAN_SEEDS,
    SHELF_PICK_SAME_APPROACH,
    FollowPlans,
    KinderControllers,
    KinderSkills,
)
from .objects import SweepOffShelf3DObjects as O
from .scripted_skills import CUBE_STANDOFF, Frames, SkillFailed, StepBudgetExhausted
from .types import FloorGrasp

# Leans of the approach from vertical (deg), about the closing axis, least first: a cube
# with room around it keeps the vertical grasp, and a lean takes the wrist off a wall.
FLOOR_PICK_LEANS_DEG = (0.0, 15.0, 30.0, 45.0)
# Where the closed pads reach, in the pinch-site frame (m), measured on the live model
# with the pads closed on the 2 cm cube: their tips ride 42 mm ahead of the site, and
# each pad is 22 mm wide across the closing axis.
PAD_TIP_REACH = 0.042
PAD_HALF_WIDTH = 0.011
# How far (m) the closed pads' tips stay above the cube's support (floor or box bottom).
PAD_SUPPORT_MARGIN = 0.003
# Off-centre grasps, for a cube with a neighbour on one side along the pads' width (the
# lean direction): the hand shifted this far (m) away from the neighbour, so the 22 mm
# pads overhang the free side instead of clipping the neighbour. A neighbour counts when
# its centre is within ADJACENT_ALONG along that axis and ADJACENT_ACROSS along the
# closing axis (the open jaw's reach). In a touching row in the box, the end cube's only
# grasp was the hand 6 mm off centre (2026-10-21 refusal diagnosis, case 1).
FLOOR_PICK_OFFSETS = (0.006, 0.009, 0.012)
ADJACENT_ALONG = 0.04
ADJACENT_ACROSS = 0.055
# The pre-grasp: this far (m) back along the approach.
FLOOR_PICK_PRE_BACK = 0.05
# The lift: straight up this far (m), and in the box until the cube's bottom clears the
# rim by `BOX_RIM_CLEARANCE`.
FLOOR_PICK_LIFT = 0.10
BOX_RIM_CLEARANCE = 0.04
# Base distances (m) from the cube, and headings around it, each base facing the cube.
FLOOR_PICK_REACHES = (0.55, 0.45, 0.65)
FLOOR_PICK_AROUND_DEG = tuple(range(0, 360, 30))
# A leaned grasp's wrist must be on the base's side of the cube: the approach's
# horizontal direction at least this aligned (cosine) with base -> cube.
FLOOR_PICK_MIN_FACING = -0.3
# IK restarts at the grasp (MuJoCo, `ShelfSkills.ik`), and at the pre-grasp and lift
# end, which are seeded from the grasp's branch.
GRASP_IK_ATTEMPTS = 4
BRANCH_IK_ATTEMPTS = 2
# Grasp executions per pick, KINDER plans tried per execution, and per grasp.
FLOOR_PICK_EXECUTIONS = 3
FLOOR_PICK_PLANS = 24
FLOOR_PICK_PLANS_PER_GRASP = 6


class FloorPicks:
    """The floor and box picks on one live scene, through `KinderSkills`' controllers,
    MuJoCo checks and bookkeeping."""

    def __init__(self, *, kinder: KinderSkills) -> None:
        self.kinder = kinder
        self.shelf = kinder.shelf

    # --- the skill --------------------------------------------------------------------

    def pick(self, *, cube: str, label: str) -> None:
        """Pick `cube` off the floor or out of the box, closed loop over executions;
        raises `SkillFailed` unless the cube ends in the hand."""
        kinder = self.kinder
        where = self.support(cube=cube)
        missed: list[str] = []
        tried: list[np.ndarray] = []
        for _ in range(FLOOR_PICK_EXECUTIONS):
            if self.support(cube=cube) != where:
                break
            try:
                planned = self.attempt(cube=cube, tried=tried, label=label)
            except StepBudgetExhausted:
                raise
            except SkillFailed as exc:
                missed.append(str(exc))
                planned = True
            if not planned:
                if missed:
                    break
                raise SkillFailed(f"{label}: no grasp on {cube} planned")
            if kinder.held(name=cube):
                if not kinder.double_grasp(cube=cube, missed=missed, label=label):
                    return
                continue
            missed.append(f"{cube} did not come with the hand")
            kinder.note(
                stage=label, kind="execution_failed", reason=f"attempt {len(missed)} missed"
            )
            kinder.sync()
            kinder.retract(label=f"{label}_regrasp")
            for _ in range(GRIP_STEPS):
                kinder.env_step(action=FollowPlans._hold(grip=0.0), label=f"{label}_open")  # noqa: SLF001
            kinder.sync()
        raise SkillFailed(f"{label}: {cube} not picked ({'; '.join(missed)})"[:300])

    def support(self, *, cube: str) -> str | None:
        """What the cube rests on, as the planner's predicates say: "floor", "box" or
        None (held, on the shelf, or lodged somewhere else)."""
        atoms = self.kinder.backend.abstract_atoms()
        if ("InBox", (cube, BOX)) in atoms:
            return "box"
        if ("OnFloor", (cube,)) in atoms:
            return "floor"
        return None

    def attempt(self, *, cube: str, tried: list[np.ndarray], label: str) -> bool:
        """One grasp execution, from the first candidate whose plans KINDER finds and
        MuJoCo clears; False when none does (nothing moved). Approaches in `tried`
        already missed this cube; the one executed is added."""
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills

        kinder = self.kinder
        robot, target = kinder.objects("robot", cube)
        centre = self.shelf.cube_position(cube=cube)
        obj_pose = kinder.cube_pose(cube=cube)
        controller = KinderControllers.get(name="PickShelf")
        planner = shelf_skills.run_motion_planning
        current: list[FloorGrasp] = []
        failures: dict[int, int] = {}

        def plan(*args: Any, **kwargs: Any) -> Any:
            grasp = current[-1]
            if kwargs.get("held_object") is not None:
                # The retract home starts where the straight lift leaves the hand.
                args = (args[0], [*grasp.lift_conf.tolist(), *list(args[1])[7:]], *args[2:])
            elif np.allclose(np.asarray(args[2])[:7], grasp.grasp_conf):
                # Upstream's approach straight to the grasp: `PreGraspApproach` replaces
                # it with its plan to the pre-grasp, so it is not planned.
                return [list(args[1]), list(args[2])]
            for seed in SHELF_PICK_PLAN_SEEDS:
                result = planner(*args, **{**kwargs, "seed": seed})
                if result is not None:
                    return result
            return None

        for drives, grasp in enumerate(self.candidates(cube=cube, tried=tried), start=1):
            key = id(grasp.hand)
            if failures.get(key, 0) >= FLOOR_PICK_PLANS_PER_GRASP:
                continue
            current.append(grasp)
            with (
                kinder.constant(
                    module=shelf_skills,
                    name="GRASP_TRANSFORM_TO_OBJECT",
                    value=kinder.pose_in(obj_pose=obj_pose, hand=grasp.hand),
                ),
                kinder.constant(
                    module=shelf_skills,
                    name="inverse_kinematics",
                    value=kinder.fixed_ik(conf=grasp.grasp_conf),
                ),
                kinder.constant(module=shelf_skills, name="run_motion_planning", value=plan),
            ):
                pick = controller(objects=[robot, target], pybullet_sim=kinder.planning_sim())
                pick.settle = kinder.settler(
                    hand=grasp.hand, entry=(grasp.pre,), label=f"{label}_settle"
                )
                pick.lift = kinder.lifter(pre=grasp.lift, label=f"{label}_lift", cube=cube)
                pick.pre_conf = grasp.pre_conf.tolist()
                pick.retract_gain = SHELF_PICK_CARRY_GAIN
                params = kinder.facing_params(base=grasp.base, centre=centre, cube=cube)
                outcome = kinder.drive(controller=pick, params=params, label=label)
            if outcome == "done":
                tried.append(grasp.hand[:3, 2].copy())
                return True
            failures[key] = failures.get(key, 0) + 1
            if drives >= FLOOR_PICK_PLANS:
                return False
        return False

    # --- the candidates ---------------------------------------------------------------

    def candidates(
        self, *, cube: str, tried: list[np.ndarray] | None = None
    ) -> Generator[FloorGrasp, None, None]:
        """(grasp, base) candidates clear in MuJoCo, lazily: grasps in `hands` order, and
        for each, bases in `bases` order. A grasp whose approach is in `tried` is skipped."""
        kinder = self.kinder
        centre = self.shelf.cube_position(cube=cube)
        bases = [b for b in self.bases(centre=centre) if kinder.base_clear(base=b)]
        drivable: dict[int, bool] = {}
        lift_to = self.lift_height(cube=cube, centre=centre)
        for lean, hand in self.hands(cube=cube, centre=centre):
            if any(float(hand[:3, 2] @ t) > SHELF_PICK_SAME_APPROACH for t in tried or ()):
                continue
            pre = hand.copy()
            pre[:3, 3] -= FLOOR_PICK_PRE_BACK * hand[:3, 2]
            lift = hand.copy()
            lift[2, 3] = lift_to
            for i, base in enumerate(bases):
                if lean > 0.0 and not self.facing(hand=hand, base=base, centre=centre):
                    continue
                if i not in drivable:
                    drivable[i] = kinder.base_drivable(base=base)
                if not drivable[i]:
                    continue
                grasp_conf = self.clear_conf(pose=hand, base=base, cube=cube, seed=None)
                if grasp_conf is None:
                    continue
                if kinder.cube_between_pads(base=base, arm=grasp_conf, cube=cube):
                    # A neighbour where the pads close: they would take both cubes (two
                    # cubes face to face in the box came out together and blocked the
                    # base), so the grasp is refused rather than tried last.
                    continue
                pre_conf = self.clear_conf(pose=pre, base=base, cube=cube, seed=grasp_conf)
                if pre_conf is None:
                    continue
                lift_conf = self.clear_conf(pose=lift, base=base, cube=cube, seed=grasp_conf)
                if lift_conf is None:
                    continue
                yield FloorGrasp(
                    lean_deg=lean,
                    hand=hand,
                    pre=pre,
                    lift=lift,
                    base=base,
                    grasp_conf=grasp_conf,
                    pre_conf=pre_conf,
                    lift_conf=lift_conf,
                )

    def hands(self, *, cube: str, centre: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """(lean, pinch-site pose) grasps on the cube as it lies, least lean first: the
        pads close on a pair of opposite side faces, the approach leans along the other
        faces' normal (`Frames.leaned`), the site `standoff` back from the centre.

        After every centred grasp, the off-centre ones: for a grasp with a neighbour
        on one side along the pads' width (`neighbour_side`), the same grasp shifted
        `FLOOR_PICK_OFFSETS` away from it. A cube with neighbours on both sides gets none."""
        quaternion = self.shelf.core.get_object(cube).get_pose()[1]
        axes = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        normal = float(np.arctan2(axes[1, 0], axes[0, 0]))
        out = []
        shifted = []
        for lean in FLOOR_PICK_LEANS_DEG:
            standoff = FloorPicks.standoff(lean_deg=lean)
            for k in range(4):
                face = normal + k * np.pi / 2
                toward = np.array([np.cos(face), np.sin(face), 0.0])
                closing = np.array([toward[1], -toward[0], 0.0])
                rotation = Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean)
                hand = Frames.pose(
                    rotation=rotation, position=np.asarray(centre) - standoff * rotation[:, 2]
                )
                out.append((lean, hand))
                side = self.neighbour_side(cube=cube, centre=centre, along=toward, across=closing)
                for offset in FLOOR_PICK_OFFSETS if side else ():
                    moved = hand.copy()
                    moved[:3, 3] -= side * offset * toward
                    shifted.append((lean, moved))
        return out + shifted

    def neighbour_side(
        self, *, cube: str, centre: np.ndarray, along: np.ndarray, across: np.ndarray
    ) -> float:
        """+1 or -1: the side, along `along`, of the one neighbouring cube's side the
        pads would clip; 0 for no neighbour there, or neighbours on both sides."""
        sides = set()
        for i in range(5):
            other = f"cube_{i}"
            if other == cube:
                continue
            offset = self.shelf.cube_position(cube=other) - np.asarray(centre)
            if abs(float(offset[2])) > 2 * O.cube_half_size:
                continue
            a, b = float(offset @ along), float(offset @ across)
            if O.cube_half_size <= abs(a) < ADJACENT_ALONG and abs(b) < ADJACENT_ACROSS:
                sides.add(float(np.sign(a)))
        return sides.pop() if len(sides) == 1 else 0.0

    @staticmethod
    def standoff(*, lean_deg: float) -> float:
        """Pinch site to cube centre (m) along the approach: the shelf's `CUBE_STANDOFF`,
        or more when leaning, so the closed pads' lower corners -- `PAD_TIP_REACH` along
        the approach and `PAD_HALF_WIDTH` across it -- stay `PAD_SUPPORT_MARGIN` above the
        cube's bottom face."""
        lean = np.deg2rad(lean_deg)
        drop = O.cube_half_size - PAD_SUPPORT_MARGIN - PAD_HALF_WIDTH * np.sin(lean)
        return float(max(CUBE_STANDOFF, PAD_TIP_REACH - drop / np.cos(lean)))

    def lift_height(self, *, cube: str, centre: np.ndarray) -> float:
        """The pinch site's height at the end of the straight lift."""
        lift = float(centre[2]) + CUBE_STANDOFF + FLOOR_PICK_LIFT
        if self.support(cube=cube) == "box":
            _, rim = self.shelf.core.box_interior()
            # The cube hangs below the site by up to the standoff plus its half size.
            needed = (
                rim
                + BOX_RIM_CLEARANCE
                + O.cube_half_size
                + FloorPicks.standoff(lean_deg=max(FLOOR_PICK_LEANS_DEG))
            )
            lift = max(lift, needed)
        return lift

    def bases(self, *, centre: np.ndarray) -> list[np.ndarray]:
        """Base poses (x, y, heading) on `FLOOR_PICK_REACHES` rings around the cube, each
        facing it, the direction nearest the robot's current base first."""
        here = self.shelf.data.qpos[self.shelf.ba]
        toward_robot = float(np.arctan2(here[1] - centre[1], here[0] - centre[0]))
        around = sorted(
            np.deg2rad(FLOOR_PICK_AROUND_DEG),
            key=lambda a: abs((a - toward_robot + np.pi) % (2 * np.pi) - np.pi),
        )
        out = []
        for angle in around:
            for reach in FLOOR_PICK_REACHES:
                xy = np.asarray(centre[:2]) + reach * np.array([np.cos(angle), np.sin(angle)])
                heading = float(np.arctan2(centre[1] - xy[1], centre[0] - xy[0]))
                out.append(np.array([xy[0], xy[1], heading]))
        return out

    @staticmethod
    def facing(*, hand: np.ndarray, base: np.ndarray, centre: np.ndarray) -> bool:
        """Whether a leaned grasp's wrist is on the base's side of the cube."""
        horizontal = hand[:2, 2] / np.linalg.norm(hand[:2, 2])
        inward = np.asarray(centre[:2]) - base[:2]
        return float(horizontal @ inward / np.linalg.norm(inward)) >= FLOOR_PICK_MIN_FACING

    def clear_conf(
        self, *, pose: np.ndarray, base: np.ndarray, cube: str, seed: np.ndarray | None
    ) -> np.ndarray | None:
        """The arm at the pinch-site `pose` from `base`, on MuJoCo's kinematics, with no
        penetration (`ShelfSkills.ik`), searched from home (`seed` None) or kept within
        `SHELF_PICK_MAX_JUMP` of `seed`'s IK branch, and `keeps_clearance`; None if
        there is none."""
        shelf = self.shelf
        try:
            conf, collision = shelf.ik(
                position=pose[:3, 3],
                rotation=pose[:3, :3],
                base=base,
                seed=HOME_ARM if seed is None else seed,
                attempts=GRASP_IK_ATTEMPTS if seed is None else BRANCH_IK_ATTEMPTS,
            )
        except SkillFailed:
            return None
        if collision > 0:
            return None
        if seed is not None:
            jump = np.max(np.abs((conf - seed + np.pi) % (2 * np.pi) - np.pi))
            if jump > SHELF_PICK_MAX_JUMP:
                return None
        if not self.keeps_clearance(base=base, arm=conf, cube=cube):
            return None
        return self.kinder.wrapped(arm=conf)

    def keeps_clearance(self, *, base: np.ndarray, arm: np.ndarray, cube: str) -> bool:
        """Every collision geom of the arm and gripper at least upstream's
        `PickCubeController.GRASP_OBSTACLE_CLEARANCE` from the box, the cupboard and every
        cube but `cube`, at this base and arm. Upstream reserves it, in its own planning
        scene, from every non-target obstacle on the way down to the grasp."""
        from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController

        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        clearance = PickCubeController.GRASP_OBSTACLE_CLEARANCE
        d.qpos[:] = shelf.data.qpos
        d.qpos[shelf.ba] = base
        d.qpos[shelf.qa] = arm
        shelf.mujoco.mj_kinematics(m, d)
        obstacles = self.obstacle_geoms(cube=cube)
        for g in self.arm_geoms():
            for f in obstacles:
                if shelf.mujoco.mj_geomDistance(m, d, g, f, clearance, None) < clearance:
                    return False
        return True

    def arm_geoms(self) -> list[int]:
        """The robot's collision geoms but the mobile base's: the arm and the gripper,
        which upstream's planning scene models as the robot."""
        shelf = self.shelf
        m = shelf.model
        return sorted(
            g
            for g in shelf.robot_geoms
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and m.body(m.geom_bodyid[g]).name != "robot_base_link"
        )

    def obstacle_geoms(self, *, cube: str) -> list[int]:
        """The box's, the cupboard's and every other cube's collision geoms."""
        shelf = self.shelf
        m = shelf.model
        others = {
            g
            for g in range(m.ngeom)
            if str(m.body(m.geom_bodyid[g]).name).startswith("cube_")
            and m.body(m.geom_bodyid[g]).name != cube
        }
        return sorted(
            g
            for g in shelf.cupboard_geoms | shelf.box_geoms | others
            if m.geom_contype[g] or m.geom_conaffinity[g]
        )
