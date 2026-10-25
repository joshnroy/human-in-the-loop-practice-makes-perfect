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
   approach as upstream's grid 36 has them (yaw offsets about the approach, tilts about
   the closing axis, a full or narrow opening), then this package's additions
   (`FLOOR_PICK_EXTRA_OPENINGS`), the pinch site as far back as keeps the closed pads'
   tips above the cube's support (`FloorPicks.standoff`);
2. bases on rings around the cube, sampled separately from the grasp, nearest the
   robot first;
3. each (grasp, base) solved on MuJoCo's own kinematics at the grasp, the pre-grasp and
   the end of the lift, the jaw already opened as that grasp has it, and kept only
   if every robot geom stays `FLOOR_PICK_OBSTACLE_CLEARANCE` from the box, the cupboard
   and the other cubes there, with no other cube where the pads close
   (`KinderSkills.cube_between_pads`);
4. KINDER's `shelf.PickShelfController` plans the base path, the arm to the pre-grasp
   and the held-cube retract home with the fingers as the shaped jaw holds them (from
   the drive to the grasp), re-checked in MuJoCo (`KinderSkills.drive`); the
   descent to the grasp and the straight lift run on MuJoCo's kinematics
   (`KinderSkills.settler`, `KinderSkills.lifter`), and the retract carries the cube at
   the shelf pick's reduced gain;
5. closed loop: a cube that did not come with the hand is looked at again and replanned.

Nothing here imports KINDER at module scope.
"""

from collections.abc import Generator
from typing import Any

import numpy as np

from .kinder_skills import (
    BOX,
    GRIP_STEPS,
    GRIPPER_CLOSED,
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
from .types import FloorGrasp, FloorHand

# The grasps, least tilted first (`FloorPicks.hands`). Upstream's own near-wall list, exactly
# (`tossing.PickCubeController.grasp_candidates`, "grid 36": the gripper turned about its
# approach by PICK_CUBE_GRASP_YAW_OFFSETS_DEGREES from a face, tilted about its closing
# axis by PICK_CUBE_GRASP_TILTS_DEGREES, opened fully or to PICK_CUBE_NARROW_OPENING),
# read from upstream so that it moves with the pin. Then this package's additions, each
# after the whole grid, for the pocket between a box's outer wall and the cupboard's
# bottom board that a missed sweep drops cubes into (2026-10-24 floor-pick-strip
# diagnosis: the PoC's 16 such cubes, 0 picked):
# - the grid again with the pads opened only FLOOR_PICK_EXTRA_OPENINGS (m), 12 mm off a
#   2 cm cube's faces: the narrowest upstream's gap table commands. It reaches a cube
#   balanced on the box's rim under the cube shelf (4/4 of the PoC's, against 0/4);
# - rolls (deg): the approach tilted along the closing axis, the far pad raised over
#   the bottom board and the wrist out from under the cube shelf's front edge, which
#   no tilt about the closing axis took it (pocket cubes 8/16, against 6/16).
# Tilts to 60 deg about the closing axis reached the same rim cubes and nothing else.
# Before all of these: the grid's grasps with the wrist turned half round (the jaw is
# symmetric, the arm is not), and the tilts this package used before the grid (deg), on
# all four faces -- the off-centre grasp that returns a cube face to face with another
# in the box is one of them.
FLOOR_PICK_LEANS_DEG = (15.0, 30.0, 45.0)
FLOOR_PICK_EXTRA_OPENINGS = (0.044,)
FLOOR_PICK_ROLLS_DEG = (10.0, 20.0)
# How far (m) every arm and gripper geom stays from the box, the cupboard and the other
# cubes at the grasp, the pre-grasp and the end of the lift: none, beyond not touching.
# Ours, not upstream's -- a deviation pending Josh's approval. The tossing pick reserves
# `PickCubeController.GRASP_OBSTACLE_CLEARANCE` (1 cm) against tracking error; in the
# pocket between the box's outer wall and the cupboard's bottom board almost no grasp
# keeps it. On the PoC's 25 refused poses, picks held: 1 cm 2/25, 5 mm 4/25, 2 mm 10/25
# (2 executions missed), none 12/25.
FLOOR_PICK_OBSTACLE_CLEARANCE = 0.0
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

    def plannable(self, *, cube: str) -> bool:
        """`PickPlannable`'s dry run (`pick_plannable.py`): whether the first execution of
        `pick` would find a grasp whose plans KINDER finds and MuJoCo clears -- the same
        candidates and the same planning, stopped before the first control step. False
        exactly where `pick` would refuse with "no grasp planned" and nothing moved."""
        return self.attempt(cube=cube, tried=[], label="pick_plannable", dry_run=True)

    def support(self, *, cube: str) -> str | None:
        """What the cube rests on, as the planner's predicates say: "floor", "box" or
        None (held, on the shelf, or lodged somewhere else)."""
        atoms = self.kinder.backend.abstract_atoms()
        if ("InBox", (cube, BOX)) in atoms:
            return "box"
        if ("OnFloor", (cube,)) in atoms:
            return "floor"
        return None

    def attempt(
        self, *, cube: str, tried: list[np.ndarray], label: str, dry_run: bool = False
    ) -> bool:
        """One grasp execution, from the first candidate whose plans KINDER finds and
        MuJoCo clears; False when none does (nothing moved). Approaches in `tried`
        already missed this cube; the one executed is added. A `dry_run` returns True at
        that candidate without executing it (`plannable`)."""
        import kinder_models.dynamic3d.shelf.parameterized_skills as shelf_skills
        from kinder_models.dynamic3d.tossing.parameterized_skills import (
            gripper_command_for_opening,
        )

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
            # The fingers as the shaped jaw holds them from the drive to the grasp (and
            # no wider on the cube): planned open, the retract home from the end of a
            # lift beside the box's wall started 13 mm into it, and never planned.
            fingers = FloorPicks.shaped_fingers(robot=args[0], opening=grasp.opening)
            start, goal = list(args[1]), list(args[2])
            start[7:], goal[7:] = fingers, fingers
            args = (args[0], start, goal, *args[3:])
            if kwargs.get("held_object") is not None:
                # The retract home starts where the straight lift leaves the hand.
                args = (args[0], [*grasp.lift_conf.tolist(), *fingers], *args[2:])
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
                pick.approach_grip = gripper_command_for_opening(grasp.opening)
                params = kinder.facing_params(base=grasp.base, centre=centre, cube=cube)
                outcome = kinder.drive(controller=pick, params=params, label=label, dry_run=dry_run)
            if outcome == "planned":
                return True
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
        for option in self.hands(cube=cube, centre=centre):
            lean, hand, jaw = option.tilt_deg, option.hand, self.shaped_jaw(opening=option.opening)
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
                grasp_conf = self.clear_conf(pose=hand, base=base, cube=cube, seed=None, jaw=jaw)
                if grasp_conf is None:
                    continue
                if kinder.cube_between_pads(base=base, arm=grasp_conf, cube=cube, held=jaw):
                    # A neighbour where the pads close: they would take both cubes (two
                    # cubes face to face in the box came out together and blocked the
                    # base), so the grasp is refused rather than tried last.
                    continue
                pre_conf = self.clear_conf(pose=pre, base=base, cube=cube, seed=grasp_conf, jaw=jaw)
                if pre_conf is None:
                    continue
                lift_conf = self.clear_conf(
                    pose=lift, base=base, cube=cube, seed=grasp_conf, jaw=jaw
                )
                if lift_conf is None:
                    continue
                yield FloorGrasp(
                    lean_deg=lean,
                    opening=option.opening,
                    hand=hand,
                    pre=pre,
                    lift=lift,
                    base=base,
                    grasp_conf=grasp_conf,
                    pre_conf=pre_conf,
                    lift_conf=lift_conf,
                )

    def hands(self, *, cube: str, centre: np.ndarray) -> list[FloorHand]:
        """The grasps on the cube as it lies, in the order they are tried (see
        `FLOOR_PICK_EXTRA_OPENINGS`): upstream's grid 36, then this package's additions.
        A grasp is a closing axis on one of the cube's two face normals turned by a yaw
        offset, a tilt about it or a roll along it (`grasp_pose`), and an opening.

        After all of them, the off-centre ones: for a grasp with a neighbour on one side
        along the pads' width (`neighbour_side`), the same grasp shifted
        `FLOOR_PICK_OFFSETS` away from it. A cube with neighbours on both sides gets none."""
        from kinder_models.dynamic3d.tossing import parameterized_skills as tossing

        normal = KinderSkills.side_face_yaw(
            quaternion=self.shelf.core.get_object(cube).get_pose()[1]
        )
        full = float(tossing.PICK_CUBE_FULL_OPENING)
        faces = range(tossing.PICK_CUBE_GRASP_FACES)

        def openings(*, offset_deg: float) -> list[float]:
            # Upstream's rule: the narrow opening only where the cube's span across the
            # closing axis, plus its clearance, fits inside it.
            phi = np.deg2rad(offset_deg)
            span = 2 * O.cube_half_size * (abs(np.cos(phi)) + abs(np.sin(phi)))
            narrow = float(tossing.PICK_CUBE_NARROW_OPENING)
            fits = span + tossing.PICK_CUBE_NARROW_OPENING_CLEARANCE <= narrow
            return [full, narrow] if fits else [full]

        # (yaw offset, tilt, roll, face, opening)
        keys: list[tuple[float, float, float, int, float]] = [
            (0.0, 0.0, 0.0, k, full) for k in faces
        ]
        for grid_offset in tossing.PICK_CUBE_GRASP_YAW_OFFSETS_DEGREES:
            for grid_tilt in tossing.PICK_CUBE_GRASP_TILTS_DEGREES:
                for k in faces:
                    for opening in openings(offset_deg=grid_offset):
                        key = (float(grid_offset), float(grid_tilt), 0.0, k, opening)
                        if key not in keys:
                            keys.append(key)
        # The same grasps with the wrist turned half round: the jaw is symmetric, so
        # upstream lists two faces, but the arm reaching them is not.
        keys += [(o, t, r, k + 2, opening) for o, t, r, k, opening in list(keys)]
        grid = list(keys)
        for lean in FLOOR_PICK_LEANS_DEG:
            for k in range(4):
                keys.append((0.0, lean, 0.0, k, full))
        for opening in FLOOR_PICK_EXTRA_OPENINGS:
            keys += [(o, t, r, k, opening) for o, t, r, k, was in grid if was == full]
        for roll in FLOOR_PICK_ROLLS_DEG:
            for k in range(4):
                for opening in (full, *FLOOR_PICK_EXTRA_OPENINGS):
                    keys.append((0.0, 0.0, roll, k, opening))

        # Least tilt first, as this package always tried them; within a tilt, in the order
        # above. In upstream's own order its 20 deg tilt came before the 15 deg lean that
        # picks a cube 2 cm in front of the box's 20 cm wall, and its lift and retract
        # caught the wall (floor-out-near-centre: arm stalled 1.8 rad from home).
        keys.sort(key=lambda key: (key[4] in FLOOR_PICK_EXTRA_OPENINGS, key[2] != 0.0, abs(key[1])))
        out = []
        shifted = []
        for offset, tilt, roll, k, opening in keys:
            yaw = normal + k * np.pi / 2 + np.deg2rad(offset)
            hand = FloorPicks.grasp_pose(centre=centre, yaw=yaw, tilt_deg=tilt, roll_deg=roll)
            out.append(FloorHand(tilt_deg=max(abs(tilt), abs(roll)), hand=hand, opening=opening))
            closing = np.array([hand[0, 0], hand[1, 0], 0.0])
            closing /= np.linalg.norm(closing)
            along = np.array([-closing[1], closing[0], 0.0])
            side = self.neighbour_side(cube=cube, centre=centre, along=along, across=closing)
            for shift in FLOOR_PICK_OFFSETS if side else ():
                moved = hand.copy()
                moved[:3, 3] -= side * shift * along
                shifted.append(
                    FloorHand(tilt_deg=max(abs(tilt), abs(roll)), hand=moved, opening=opening)
                )
        return out + shifted

    @staticmethod
    def grasp_pose(
        *, centre: np.ndarray, yaw: float, tilt_deg: float, roll_deg: float
    ) -> np.ndarray:
        """Pinch-site pose: pads closing along the horizontal heading `yaw`; the approach
        tilted `tilt_deg` about the closing axis (`Frames.leaned`, upstream's tilt) or
        rolled `roll_deg` along it (`rolled`), the sign picking the side; the site
        `standoff` (or `rolled_standoff`) back from the cube's centre."""
        closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        if roll_deg:
            rotation = FloorPicks.rolled(
                closing=np.sign(roll_deg) * closing, roll_deg=abs(roll_deg)
            )
            standoff = FloorPicks.rolled_standoff(roll_deg=abs(roll_deg))
        else:
            across = np.array([-closing[1], closing[0], 0.0])
            rotation = Frames.leaned(
                closing=closing, lean_toward=np.sign(tilt_deg) * across, lean_deg=abs(tilt_deg)
            )
            standoff = FloorPicks.standoff(lean_deg=abs(tilt_deg))
        return Frames.pose(
            rotation=rotation, position=np.asarray(centre) - standoff * rotation[:, 2]
        )

    @staticmethod
    def rolled(*, closing: np.ndarray, roll_deg: float) -> np.ndarray:
        """Gripper rotation: the approach tilted `roll_deg` from straight down toward the
        horizontal `closing`, about the pads' width axis, the closing axis tilted with it
        (local x the closing axis, local z the approach). The pad on the `closing` side
        is the higher one."""
        roll = np.deg2rad(roll_deg)
        up = np.array([0.0, 0.0, 1.0])
        approach = np.sin(roll) * closing - np.cos(roll) * up
        tilted = np.cos(roll) * closing + np.sin(roll) * up
        return np.column_stack([tilted, np.cross(approach, tilted), approach])

    @staticmethod
    def rolled_standoff(*, roll_deg: float) -> float:
        """`standoff` for a rolled grasp: the lower closed pad's tip, `PAD_TIP_REACH`
        along the approach and a cube half-size down the tilted closing axis, stays
        `PAD_SUPPORT_MARGIN` above the cube's bottom face."""
        roll = np.deg2rad(roll_deg)
        drop = O.cube_half_size - PAD_SUPPORT_MARGIN - O.cube_half_size * np.sin(roll)
        return float(max(CUBE_STANDOFF, PAD_TIP_REACH - drop / np.cos(roll)))

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
            # The cube hangs below the site by up to the standoff plus its half size: the
            # standoff of a 45 deg lean, more than any grasp tried here has.
            needed = rim + BOX_RIM_CLEARANCE + O.cube_half_size + FloorPicks.standoff(lean_deg=45.0)
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
        self,
        *,
        pose: np.ndarray,
        base: np.ndarray,
        cube: str,
        seed: np.ndarray | None,
        jaw: dict[int, float],
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
                held=jaw,
            )
        except SkillFailed:
            return None
        if collision > 0:
            return None
        if seed is not None:
            jump = np.max(np.abs((conf - seed + np.pi) % (2 * np.pi) - np.pi))
            if jump > SHELF_PICK_MAX_JUMP:
                return None
        if not self.keeps_clearance(base=base, arm=conf, cube=cube, jaw=jaw):
            return None
        return self.kinder.wrapped(arm=conf)

    def keeps_clearance(
        self, *, base: np.ndarray, arm: np.ndarray, cube: str, jaw: dict[int, float]
    ) -> bool:
        """Every collision geom of the arm and gripper, the jaw shaped as `jaw`, at
        least `FLOOR_PICK_OBSTACLE_CLEARANCE` from the box, the cupboard and every cube
        but `cube`, at this base and arm."""
        shelf = self.shelf
        m, d = shelf.model, shelf.ik_data
        clearance = FLOOR_PICK_OBSTACLE_CLEARANCE
        d.qpos[:] = shelf.data.qpos
        for address, value in jaw.items():
            d.qpos[address] = value
        d.qpos[shelf.ba] = base
        d.qpos[shelf.qa] = arm
        shelf.mujoco.mj_kinematics(m, d)
        obstacles = self.obstacle_geoms(cube=cube)
        for g in self.arm_geoms():
            for f in obstacles:
                if shelf.mujoco.mj_geomDistance(m, d, g, f, clearance, None) < clearance:
                    return False
        return True

    def shaped_jaw(self, *, opening: float) -> dict[int, float]:
        """The gripper joints (qpos address: value) with the pads opened `opening` (m):
        `GRIPPER_CLOSED` scaled by upstream's command for that gap
        (`gripper_command_for_opening`), within 0.01 rad of where the jaw settles."""
        from kinder_models.dynamic3d.tossing.parameterized_skills import (
            gripper_command_for_opening,
        )

        command = gripper_command_for_opening(opening)
        m = self.shelf.model
        return {
            int(m.jnt_qposadr[m.joint(f"robot_{side}_{joint}").id]): closed * command
            for side in ("left", "right")
            for joint, closed in GRIPPER_CLOSED.items()
        }

    @staticmethod
    def shaped_fingers(*, robot: Any, opening: float) -> list[float]:
        """KINDER's PyBullet finger joints with the pads opened `opening` (m), by
        upstream's own gap table (`finger_state_for_opening`)."""
        from kinder_models.dynamic3d.tossing.parameterized_skills import (
            finger_state_for_opening,
        )

        return list(robot.finger_state_to_joints(finger_state_for_opening(opening)))

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
