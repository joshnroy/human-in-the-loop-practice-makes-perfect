"""Scripted skill controllers for SweepOffShelf3D, driven by the planner's skills.

Vendored from the scripted solvability check (hitl-pmp branch
`claude/sweep-off-shelf3d-solvability` @ 9b635c720, `scripts/sweep_off_shelf3d_skills.py`).
The one change: `ShelfSkills` attaches to an environment it is handed, already reset,
instead of making and resetting its own.

Control only: nothing here renders, captions or writes video. A caller that wants a
video passes a `Recorder`, which is told about every control step and every moment
worth a keyframe; with no recorder the skills run headless. Each of the six planner
skills is one call that runs to completion or raises `SkillFailed`:

- `pick_wiper()`: take the wiper off its home on top of the cupboard.
- `sweep_shelf(cube=, offset=, angle=)`: one stroke aimed at one cube on shelf index
  1, pulling it over the open front edge into the box. The stroke is turned `angle`
  deg ([-45, 45]) from the straight pull, on the line through the cube offset by
  `offset` m ([-0.10, 0.10]) along the head (`AngledLane`). The pull speed
  (SWEEP_SPEED) and the head's height over the board (SWEEP_HEIGHT) are fixed.
- `return_wiper()`: put the held wiper back where it started on top of the cupboard.
- `pick_cube_from_shelf(cube=)`, `pick_cube_from_floor(cube=)`,
  `place_cube_in_box(cube=)`.

Nothing here edits the environment. The friction grasp is the environment's own; the
only state ever set directly is `drop_cube_into_box`, a documented scene setup.

The wiper is only ever held at one fixed grasp, a constant hand pose in the wiper's own
frame (`ShelfSkills.grasp_in_wiper`): on the handle at its centre of mass, the wrist
over the handle's free end. Its home is the cupboard's top board with the handle
overhanging the open front, so the hand takes it, sweeps with it and puts it back at
that one grasp, from outside the cupboard. Base poses and arm routes adapt to where the
wiper lies; the grasp never does, and every grasp taken is recorded with its measured
hand-to-wiper pose (`grasps_taken`).

Four facts shape every motion. The cupboard slots are 0.254 m tall, too short for a
vertical gripper (the wrist hits the board above), so every grasp in the cupboard
leans the gripper about its closing axis; a lean keeps the pads on the same two faces.
The 2F-85's friction grasp resists rotation about its closing axis worst of all, so
the wiper grasp sits beside its centre of mass. Lowered behind the cubes, the flat
wiper's handle comes down on any cube in its line: where a stroke runs relative to its
target cube is the sweep's `offset`, and so what practice learns. And the box keeps
the base at least BASE_X back from the cupboard while the leaned hand only reaches
0.6 m or more in front of the base, so the base carries each stroke and the arm trims.
Near the cupboard the arm follows routes planned joint by joint on one IK branch, with
the carried wiper in the collision check.

Nothing here imports KINDER: the environment is built by `backend.py` and handed in.
"""

import contextlib
from collections.abc import Generator
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation, Slerp

ENV_ID = "kinder/SweepOffShelf3D-o5-v0"
CUBES = tuple(f"cube_{i}" for i in range(5))
SKILLS = (
    "pick_wiper",
    "sweep_shelf",
    "return_wiper",
    "pick_cube_from_shelf",
    "pick_cube_from_floor",
    "place_cube_in_box",
)
CUBE_SHELF = 1
# The wiper's home: the cupboard's top board, above the third shelf.
WIPER_SHELF = 3
CONTROL_DT = 0.1
# The base stays this far back so its front casters clear the box's near wall (x 1.11).
BASE_X = 0.80
# Lean of the gripper from vertical, about its closing axis, and how far the pinch site
# stands back from the gripped bar's centre line along the approach so the pad tips
# clear the board (kinematic probe: pads 2-4 mm above it).
WIPER_LEAN_DEG = 60.0
WIPER_STANDOFF = 0.035
CUBE_STANDOFF = 0.035
CUBE_LEANS_DEG = (30.0, 40.0, 50.0, 60.0)
PAD_HALF_LENGTH = 0.011
# The one fixed wiper grasp: on the handle at the centre of mass (None), wrist over the
# free end, the side it must be on to pull the head toward the robot. It becomes a pose
# in the wiper's own frame once, at reset, and is never searched over.
SWEEP_ALONG = None
SLOT_ENTRY_RISE = 0.02
# Joint-space jump allowed between neighbouring waypoints of a planned route: more is a
# change of IK branch, which swings the arm through whatever is between.
MAX_JOINT_JUMP = 0.5
ROUTE_STEP = 0.03
ROUTE_TURN_STEP = 0.1
# Penetration of the carried wiper into the cupboard a route may plan through: resting
# on the board it sets the wiper down on is contact, not collision.
CARRIED_WIPER_ALLOWANCE = 0.004
# How closely each intermediate waypoint is tracked (rad): looser lets the pads cut the
# corner onto the handle before the hand is over it.
ROUTE_TRACKING = 0.02
# Base x positions for setting a cube down on the open floor beside the box.
FLOOR_BASE_XS = (0.45, 0.35, 0.55, 0.25)
CUBE_FLOOR_XY = (0.95, 0.45)
SWEEP_STEM = (-1.0, 0.0, 0.0)
CARRY_Z = 0.65
HEAD_AHEADS = (0.80, 0.75, 0.85, 0.70, 0.90, 0.65)
HEAD_AHEAD = 0.80
# Stroke geometry: the head's centre starts behind the deepest cube and stops with the
# head just past the shelf's front edge; limit keeps the head off the cupboard's side
# walls.
STROKE_START_X = 1.60
STROKE_END_X = 1.355
STROKE_LIMIT = 0.17
# SweepShelf (Josh, 2026-10-24): one angled stroke aimed at one cube; the sampled
# parameters are [offset, angle]. `angle` (deg) turns the whole stroke -- path and head
# -- about z from the straight-out -x pull; > 0 pulls toward +y. The pull direction is
# d = (-cos a, sin a), the head lies along n = (sin a, cos a) (n = +y at a = 0), and the
# line passes the target cube at `offset` along n. The pull speed (m/s) and the head's
# height over the board (m; a cube is 2 cm tall, the head 1.5 cm) are fixed.
SWEEP_OFFSET_BOUNDS = (-0.10, 0.10)
SWEEP_ANGLE_BOUNDS = (-45.0, 45.0)
SWEEP_SPEED = 0.05
SWEEP_HEIGHT = 0.01
# The cupboard's inner faces the angled stroke keeps the wiper inside, less a margin:
# side panels at |y| 0.291, back panel at x 1.617 (the straight stroke's head reaches
# x 1.615 and |y| 0.271).
CUPBOARD_INNER_Y = 0.291
CUPBOARD_INNER_X = 1.617
ANGLED_Y_MARGIN = 0.02
ANGLED_X_MARGIN = 0.002
HEAD_HALF = (0.015, 0.1)
HANDLE_CENTRE = 0.115
HANDLE_HALF = (0.1, 0.0075)
ANGLED_MIN_STROKE = 0.05
# Half-width of the strip under the handle (and the fingers on it) that must be free of
# cubes where the wiper is laid down to reseat it: handle half-width + cube
# half-diagonal + margin.
HANDLE_BAND = 0.03
# Put the wiper down and take it again only once it has turned this far in the grasp.
RESEAT_TURN_DEG = 6.0
# Control steps in a row without both pads on the wiper before a sweep calls it lost.
WIPER_LOST_STEPS = 5
HOVER = 0.08
RESEAT_HEAD_X = 1.50
LOWER_TOLERANCE = 0.003
LOWER_STEPS = 20
CONTACT_STEP_LIMIT = 40
CONTACT_PENETRATION = 0.04
# A put-down onto the wiper's home that stops short is let go anyway once the wiper is
# this level (deg) and its head this close (m) above the board it rests on.
PUT_DOWN_LEVEL_DEG = 3.0
SLOT_PUT_DOWN_GAP = 0.003
PULL_CHUNK = 2
# How far below the requested height the pull aims the head, pressing it into the
# board so the cubes it pushes cannot wedge under its front edge and lift it.
PULL_PRESS = 0.005
REGRASP_TRIES = 3
CUBE_PICK_TRIES = 2
GRASP_TEST_LIFT = 0.02
PLACE_ON_SHELF_XY = (1.45, 0.0)
BOX_RELEASE_Z = 0.24


class SkillFailed(RuntimeError):
    """A skill could not reach its postcondition; the message says where and why."""


class WiperLost(RuntimeError):
    """The per-step `HoldingWiper` check failed mid-sweep. Deliberately not a
    `SkillFailed`: tolerant motions (`raise_hand`, `hand_to`, the IK fallbacks) catch
    `SkillFailed` and carry on, which swallowed the loss and let the stroke continue with
    no wiper in hand. `sweep_shelf` turns it into a `SkillFailed`."""


class StepBudgetExhausted(SkillFailed):
    """The run's hard control-step cap was reached; nothing here loops open."""


class Recorder(Protocol):
    def on_step(self, *, skills: "ShelfSkills", label: str) -> None: ...

    def mark(self, *, skills: "ShelfSkills", label: str) -> None: ...


class AngledLane(BaseModel):
    """The rotated stroke's plan: head-centre points (x, y) and the stroke's axes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    angle: float
    direction: np.ndarray
    normal: np.ndarray
    mouth: np.ndarray
    start: np.ndarray
    end: np.ndarray
    shift: float

    @staticmethod
    def axes(*, angle: float) -> tuple[np.ndarray, np.ndarray]:
        a = np.deg2rad(angle)
        return np.array([-np.cos(a), np.sin(a)]), np.array([np.sin(a), np.cos(a)])

    @staticmethod
    def footprint_ok(*, head: np.ndarray, angle: float, front_x: float) -> bool:
        """Head and handle, flat at head centre `head`, clear of the cupboard's side and
        back panels wherever they are inside the cupboard (x past the front face)."""
        d, n = AngledLane.axes(angle=angle)
        y_lim = CUPBOARD_INNER_Y - ANGLED_Y_MARGIN
        x_lim = CUPBOARD_INNER_X - ANGLED_X_MARGIN
        for centre, (hd, hn) in ((head, HEAD_HALF), (head + HANDLE_CENTRE * d, HANDLE_HALF)):
            for u in np.linspace(-1.0, 1.0, 9):
                for v in np.linspace(-1.0, 1.0, 9):
                    if abs(u) < 1 and abs(v) < 1:
                        continue
                    p = centre + u * hd * d + v * hn * n
                    if p[0] > x_lim:
                        return False
                    if p[0] > front_x - 0.01 and abs(p[1]) > y_lim:
                        return False
        return True

    @staticmethod
    def plan(*, cube_xy: Any, offset: float, angle: float, front_x: float) -> "AngledLane":
        """The stroke for `offset`, `angle`: on the line through cube + offset * n along
        d, from the furthest-back head centre at most STROKE_START_X with the wiper
        inside the cupboard, to where the head's point level with the cube reaches
        STROKE_END_X. Reach-in runs from the mouth (head centre at front_x - 0.12)
        back along the same line. Like the straight stroke's lane clip, if no stroke on
        that line keeps the wiper inside, the line is shifted along n, nearest first."""
        d, n = AngledLane.axes(angle=angle)
        cos_a, sin_a = -d[0], d[1]
        mouth_x = front_x - 0.12
        for shift in sorted(np.round(np.arange(-0.30, 0.30001, 0.005), 3), key=abs):
            lateral = float(np.clip(offset + shift, -HEAD_HALF[1], HEAD_HALF[1]))
            base = np.asarray(cube_xy, dtype=float)[:2] + (offset + shift) * n
            s_end = float((base[0] - (STROKE_END_X + lateral * sin_a)) / cos_a)
            s_mouth = float((base[0] - mouth_x) / cos_a)
            s_start = float((base[0] - STROKE_START_X) / cos_a)
            while s_start < s_end - ANGLED_MIN_STROKE and not AngledLane.footprint_ok(
                head=base + s_start * d, angle=angle, front_x=front_x
            ):
                s_start += 0.005
            if s_start >= s_end - ANGLED_MIN_STROKE:
                continue
            if all(
                AngledLane.footprint_ok(head=base + s * d, angle=angle, front_x=front_x)
                for s in np.arange(s_start, s_mouth + 1e-9, 0.01)
            ):
                return AngledLane(
                    angle=angle,
                    direction=d,
                    normal=n,
                    mouth=base + s_mouth * d,
                    start=base + s_start * d,
                    end=base + s_end * d,
                    shift=float(shift),
                )
        raise SkillFailed(f"angled stroke: no line at angle {angle:+.0f} keeps the wiper inside")


class SweepParams(BaseModel):
    """`sweep_shelf`'s two sampled parameters, validated against their bounds."""

    model_config = ConfigDict(frozen=True)

    offset: float = Field(ge=SWEEP_OFFSET_BOUNDS[0], le=SWEEP_OFFSET_BOUNDS[1])
    angle: float = Field(ge=SWEEP_ANGLE_BOUNDS[0], le=SWEEP_ANGLE_BOUNDS[1])


class Frames:
    @staticmethod
    def pose(*, rotation: np.ndarray, position: Any) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = rotation
        out[:3, 3] = position
        return out

    @staticmethod
    def leaned(*, closing: np.ndarray, lean_toward: np.ndarray, lean_deg: float) -> np.ndarray:
        """Gripper rotation: pads close along `closing`, approach leans from straight
        down toward the horizontal unit vector `lean_toward` (local z is the approach,
        local x the closing axis)."""
        lean = np.deg2rad(lean_deg)
        approach = np.sin(lean) * lean_toward + np.array([0.0, 0.0, -np.cos(lean)])
        return np.column_stack([closing, np.cross(approach, closing), approach])

    @staticmethod
    def line(*, start: np.ndarray, end: np.ndarray, step: float) -> list[np.ndarray]:
        """Poses along the straight line start -> end (slerped), end included."""
        turn = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()
        count = max(
            1,
            int(np.ceil(np.linalg.norm(end[:3, 3] - start[:3, 3]) / step)),
            int(np.ceil(turn / ROUTE_TURN_STEP)),
        )
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], end[:3, :3]]))
        return [
            Frames.pose(
                rotation=rotations(t).as_matrix(),
                position=(1 - t) * start[:3, 3] + t * end[:3, 3],
            )
            for t in np.linspace(0, 1, count + 1)[1:]
        ]


class ShelfSkills:
    """Controllers for the TidyBot in SweepOffShelf3D."""

    def __init__(
        self,
        *,
        env: Any,
        seed: int = 0,
        tick_limit: int = 12000,
        recorder: Recorder | None = None,
    ) -> None:
        """`env` is the KINDER env, already reset; the skills read its scene as it is."""
        self.seed = seed
        self.tick_limit = tick_limit
        self.recorder = recorder
        self.env = env
        self.core = self.env.unwrapped._object_centric_env  # noqa: SLF001
        self.sim = self.core._robot_env.sim  # noqa: SLF001
        self.model = self.sim.model.mj_model
        self.data = self.sim.data.mj_data
        self.ik_data = self.mujoco.MjData(self.model)
        self.site = self.model.site("robot_pinch_site").id
        self.joints = [self.model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
        self.qa = self.model.jnt_qposadr[self.joints]
        self.va = self.model.jnt_dofadr[self.joints]
        self.ba = [
            self.model.jnt_qposadr[self.model.joint(f"robot_joint_{a}").id]
            for a in ("x", "y", "th")
        ]
        owner = {g: self.root_name(body=self.model.geom_bodyid[g]) for g in range(self.model.ngeom)}
        self.robot_geoms = {g for g, r in owner.items() if r == "robot_base_link"}
        # The 2F-85's own links (pads, fingers, linkage, palm), as opposed to the arm.
        self.gripper_geoms = {
            g
            for g in self.robot_geoms
            if any(
                part in str(self.model.body(self.model.geom_bodyid[g]).name)
                for part in ("pad", "follower", "driver", "coupler", "spring", "robot_base")
            )
            and self.model.body(self.model.geom_bodyid[g]).name != "robot_base_link"
        }
        self.pad_geoms = {g for g in self.robot_geoms if "pad" in (self.model.geom(g).name or "")}
        sides: dict[int, set[int]] = {}
        for g in self.pad_geoms:
            sides.setdefault(int(self.model.geom_bodyid[g]), set()).add(g)
        self.pad_sides = [frozenset(geoms) for geoms in sides.values()]
        self.cupboard_geoms = {g for g, r in owner.items() if r == "cupboard_1"}
        self.box_geoms = {g for g, r in owner.items() if r == "box_0"}
        self.wiper_geoms = sorted(g for g, r in owner.items() if r == "wiper_0")
        sizes = self.model.geom_size[self.wiper_geoms]
        self.handle_geom = self.wiper_geoms[int(np.argmax(sizes[:, 0]))]
        self.head_geom = self.wiper_geoms[int(np.argmax(sizes[:, 1]))]
        self.wiper_body = self.model.body("wiper_0").id
        self.board = {
            shelf: self.core.shelf_top_z(shelf=shelf)
            for shelf in (CUBE_SHELF, CUBE_SHELF + 1, WIPER_SHELF)
        }
        self.front_x = self.core.shelf_front_x()
        self.wiper_start = self.wiper_pose()
        self.head_start = self.data.geom_xpos[self.head_geom].copy()
        self.stem_start = self.stem_direction()
        self.base_target = self.data.qpos[self.ba].copy()
        self.arm_reference = self.data.qpos[self.qa].copy()
        self.home_arm = self.arm_reference.copy()
        self.base_step_limit = 0.015
        self.grip = 0.0
        self.ticks = 0
        self.phase = "start"
        self.notes: list[str] = []
        self.cupboard_contact_steps = 0
        self.arm_cupboard_contact_steps = 0
        self.box_contact_steps = 0
        self._in_hand: np.ndarray | None = None
        self.held_cube: str | None = None
        self.slip: float | None = None
        self.slip_deg = 0.0
        self.contact_shortfalls = 0
        # While a sweep runs, `HoldingWiper` is re-evaluated after every control step.
        self.watch_wiper = False
        self.wiper_unheld_steps = 0
        # Hand pose in the wiper's frame for the fixed grasp, and the measured
        # hand-to-wiper pose every time it is taken.
        self.grasp_in_wiper = np.linalg.inv(self.wiper_start) @ self.leaned_handle_grasp(
            along=SWEEP_ALONG, lean_sign=-1.0
        )
        self.grasps_taken: list[dict[str, Any]] = []
        self.closed_in_hand = np.eye(4)
        # Head speed actually achieved over each pull (m/s); the base tops out at
        # 0.15 m/s, so a commanded speed above that is reached only with the arm.
        self.pull_speeds: list[float] = []

    @property
    def in_hand(self) -> np.ndarray | None:
        """The measured hand-to-wiper pose of the grasp held now; None while the wiper
        is out of the hand."""
        return self._in_hand

    @in_hand.setter
    def in_hand(self, value: np.ndarray | None) -> None:  # noqa: PLR0917 -- a setter
        # The slip counters measure how far the wiper has moved in the grasp held now,
        # so they start from zero each time the hand takes it. Every way of taking it
        # -- `take_wiper`, and `KinderSkills.sync` after a practice PickWiper --
        # sets this from None; re-measuring a grasp still held (`recalibrate_grasp`)
        # does not reset them, and letting go leaves the last grasp's turn readable.
        if value is not None and self._in_hand is None:
            self.slip, self.slip_deg = 0.0, 0.0
        self._in_hand = value

    @property
    def mujoco(self) -> Any:
        import mujoco

        return mujoco

    def root_name(self, *, body: int) -> str:
        while self.model.body_parentid[body] != 0:
            body = self.model.body_parentid[body]
        return str(self.model.body(body).name)

    def mark(self, *, label: str) -> None:
        if self.recorder is not None:
            self.recorder.mark(skills=self, label=label)

    # --- measurement -------------------------------------------------------

    def counts(self) -> dict[str, int]:
        in_box = on_floor = on_shelf = 0
        for name in CUBES:
            x, _, z = self.core.get_object(name).get_pose()[0]
            if self.core.cube_in_box(name):
                in_box += 1
            elif z < 0.05:
                on_floor += 1
            elif x > self.front_x and abs(z - self.board[CUBE_SHELF] - 0.01) < 0.01:
                on_shelf += 1
        return {"in_box": in_box, "on_floor": on_floor, "on_shelf": on_shelf}

    def cube_position(self, *, cube: str) -> np.ndarray:
        return np.asarray(self.core.get_object(cube).get_pose()[0], dtype=float).copy()

    def wiper_pose(self) -> np.ndarray:
        return Frames.pose(
            rotation=self.data.xmat[self.wiper_body].reshape(3, 3),
            position=self.data.xpos[self.wiper_body],
        )

    def site_pose(self) -> np.ndarray:
        return Frames.pose(
            rotation=self.data.site_xmat[self.site].reshape(3, 3),
            position=self.data.site_xpos[self.site],
        )

    def stem_direction(self) -> np.ndarray:
        """Unit vector along the handle, from the head toward the handle's free end."""
        axis = self.data.geom_xmat[self.handle_geom].reshape(3, 3)[:, 0].copy()
        offset = self.data.geom_xpos[self.handle_geom] - self.data.geom_xpos[self.head_geom]
        return axis if float(offset @ axis) > 0 else -axis

    def wiper_lift(self) -> float:
        return float(self.data.xpos[self.wiper_body][2] - self.wiper_start[2, 3])

    def head_bottom(self) -> float:
        return float(
            min(
                (
                    self.data.geom_xpos[self.head_geom]
                    + self.data.geom_xmat[self.head_geom].reshape(3, 3)
                    @ (np.array(sign) * self.model.geom_size[self.head_geom])
                )[2]
                for sign in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            )
        )

    def shelf_cubes(self) -> list[str]:
        return [
            name
            for name in CUBES
            if self.cube_position(cube=name)[0] > self.front_x
            and self.cube_position(cube=name)[2] > self.board[CUBE_SHELF]
            and self.cube_position(cube=name)[2] < self.board[CUBE_SHELF + 1]
        ]

    def tally_contacts(self) -> None:
        cupboard = box = arm = False
        for contact in self.data.contact[: self.data.ncon]:
            a, b = contact.geom1, contact.geom2
            if a not in self.robot_geoms and b not in self.robot_geoms:
                continue
            mine, other = (a, b) if a in self.robot_geoms else (b, a)
            cupboard |= other in self.cupboard_geoms
            box |= other in self.box_geoms
            arm |= other in self.cupboard_geoms and mine not in self.gripper_geoms
        self.cupboard_contact_steps += int(cupboard)
        self.arm_cupboard_contact_steps += int(arm)
        self.box_contact_steps += int(box)

    def track_slip(self) -> None:
        if self.in_hand is None:
            return
        now = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        drift = float(np.linalg.norm(now[:3, 3] - self.in_hand[:3, 3]))
        turn = float(
            np.rad2deg(Rotation.from_matrix(now[:3, :3] @ self.in_hand[:3, :3].T).magnitude())
        )
        self.slip = drift if self.slip is None else max(self.slip, drift)
        self.slip_deg = max(self.slip_deg, turn)

    # --- low-level control ----------------------------------------------------

    def step(self, *, arm: np.ndarray, base: np.ndarray | None = None, label: str = "move") -> None:
        if self.ticks >= self.tick_limit:
            raise StepBudgetExhausted(f"{self.tick_limit} control steps reached during {label}")
        self.arm_reference = np.asarray(arm).copy()
        action = np.zeros(11)
        error = (arm - self.data.qpos[self.qa] + np.pi) % (2 * np.pi) - np.pi
        action[3:10] = np.clip(error, -0.1, 0.1)
        base = self.base_target if base is None else np.asarray(base, dtype=float)
        action[:3] = np.clip(
            base - self.data.qpos[self.ba], -self.base_step_limit, self.base_step_limit
        )
        action[10] = self.grip
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.env.step(action)
        self.ticks += 1
        self.tally_contacts()
        self.track_slip()
        if self.recorder is not None:
            self.recorder.on_step(skills=self, label=label)
        if self.watch_wiper:
            self.check_wiper_held(label=label)

    def pads_hold_wiper(self) -> bool:
        """`HoldingWiper`: both gripper pads touch the wiper (the backend's own test)."""
        touched = [False] * len(self.pad_sides)
        for contact in self.data.contact[: self.data.ncon]:
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if self.model.geom_bodyid[other] != self.wiper_body:
                    continue
                for side, pads in enumerate(self.pad_sides):
                    touched[side] |= int(mine) in pads
        return all(touched)

    def check_wiper_held(self, *, label: str) -> None:
        if self.pads_hold_wiper():
            self.wiper_unheld_steps = 0
            return
        self.wiper_unheld_steps += 1
        if self.wiper_unheld_steps >= WIPER_LOST_STEPS:
            self.watch_wiper = False
            self.in_hand = None
            raise WiperLost(f"the wiper left the hand during {label}")

    @contextlib.contextmanager
    def wiper_unwatched(self) -> Generator[None, None, None]:
        """Suspend the per-step `HoldingWiper` check while a skill lets go of the wiper
        on purpose; on the way out, however it is left, the check is back as it was
        and counts afresh."""
        watching = self.watch_wiper
        self.watch_wiper = False
        try:
            yield
        finally:
            self.watch_wiper, self.wiper_unheld_steps = watching, 0

    def hold(self, *, count: int = 15, label: str = "hold") -> None:
        arm = self.arm_reference.copy()
        for _ in range(count):
            self.step(arm=arm, label=label)

    def move_base(self, *, target: Any, label: str = "move_base") -> None:
        self.base_target = np.asarray(target, dtype=float)
        arm = self.arm_reference.copy()
        distance = np.max(np.abs(self.base_target - self.data.qpos[self.ba]))
        for _ in range(max(150, int(np.ceil(distance / self.base_step_limit)) + 50)):
            self.step(arm=arm, base=self.base_target, label=label)
            if np.max(np.abs(self.data.qpos[self.ba] - self.base_target)) < 0.003:
                return
        raise SkillFailed(f"Base failed to converge during {label}")

    def touching(self, *, d: Any) -> list[str]:
        """Robot links in contact with the cupboard or box in `d`, for error messages."""
        self.mujoco.mj_collision(self.model, d)
        fixed = self.cupboard_geoms | self.box_geoms
        names = set()
        for contact in d.contact[: d.ncon]:
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if mine in self.robot_geoms and other in fixed:
                    names.add(
                        f"{self.model.body(self.model.geom_bodyid[mine]).name}"
                        f"/{self.model.geom(other).name or 'box'}"
                    )
        return sorted(names)

    def arm_touches(self, *, d: Any) -> bool:
        """Whether any arm link (not the gripper) touches the cupboard or box in `d`."""
        self.mujoco.mj_collision(self.model, d)
        fixed = self.cupboard_geoms | self.box_geoms
        for contact in d.contact[: d.ncon]:
            for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if mine in self.robot_geoms and mine not in self.gripper_geoms and other in fixed:
                    return True
        return False

    def collision_cost(self, *, d: Any, holding: bool) -> float:
        """Penetration of the robot into itself, the cupboard or the box, and (unless
        an object is held) of non-pad robot links into the wiper."""
        self.mujoco.mj_collision(self.model, d)
        cost = 0.0
        wiper = set(self.wiper_geoms)
        fixed = self.cupboard_geoms | self.box_geoms
        for contact in d.contact[: d.ncon]:
            a, b = contact.geom1, contact.geom2
            depth = max(0.0, -contact.dist)
            if a in self.robot_geoms and b in self.robot_geoms:
                cost += max(0.0, depth - 0.001)
            elif (a in self.robot_geoms and b in fixed) or (b in self.robot_geoms and a in fixed):
                cost += depth + 0.001
            elif not holding and (
                (a in self.robot_geoms and b in wiper and a not in self.pad_geoms)
                or (b in self.robot_geoms and a in wiper and b not in self.pad_geoms)
            ):
                cost += depth
        return cost

    def ik(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        base: Any = None,
        holding: bool = False,
        attempts: int = 24,
        seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """(arm joints, penetration) reaching the pose; the first attempt starts from
        `seed` (default: the current arm), so a route stays on one IK branch."""
        mujoco = self.mujoco
        d, m = self.ik_data, self.model
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        target = Rotation.from_matrix(rotation)
        rng = np.random.default_rng(0)
        initial = self.data.qpos[self.qa].copy() if seed is None else np.asarray(seed).copy()
        candidates = []
        err = np.full(6, np.inf)
        for attempt in range(attempts):
            d.qpos[:] = self.data.qpos
            d.qpos[self.qa] = initial
            if base is not None:
                d.qpos[self.ba] = base
            # Odd restarts start near the home arm pose: seeding only from the current
            # pose gets stuck in local minima once the arm is stretched into the slot.
            if attempt % 2:
                d.qpos[self.qa] = self.home_arm + rng.normal(0, 0.3 * (attempt > 1), 7)
            elif attempt:
                d.qpos[self.qa] += rng.normal(0, 0.9, 7)
            for _ in range(400):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                current = Rotation.from_matrix(d.site_xmat[self.site].reshape(3, 3))
                err = np.r_[
                    np.asarray(position) - d.site_xpos[self.site],
                    (target * current.inv()).as_rotvec() * 0.4,
                ]
                if np.linalg.norm(err) < 0.001:
                    break
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
                j = np.vstack([jacp[:, self.va], 0.4 * jacr[:, self.va]])
                dq = j.T @ np.linalg.solve(j @ j.T + 0.001 * np.eye(6), err)
                d.qpos[self.qa] += np.clip(dq, -0.12, 0.12)
                for i, joint in enumerate(self.joints):
                    if m.jnt_limited[joint]:
                        d.qpos[self.qa[i]] = np.clip(d.qpos[self.qa[i]], *m.jnt_range[joint])
            if np.linalg.norm(err) < 0.01:
                collision = self.collision_cost(d=d, holding=holding)
                q = d.qpos[self.qa].copy()
                score = np.linalg.norm((q - initial + np.pi) % (2 * np.pi) - np.pi)
                candidates.append((score + 1000 * collision, q, collision))
                # Holding something, a jump to another IK branch swings it into the
                # box; stay on the current branch unless it really penetrates.
                if attempt == 0 and (collision == 0 or (holding and collision < 0.01)):
                    return q, float(collision)
        if candidates:
            best = min(candidates, key=lambda c: c[0])
            return best[1], best[2]
        raise SkillFailed(
            f"IK failed at {np.round(position, 3)}: residual {np.linalg.norm(err):.3f}"
        )

    def move(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        label: str = "move",
        limit: int = 250,
        position_tolerance: float = 0.008,
        orientation_tolerance: float = 0.05,
        holding: bool = False,
    ) -> None:
        try:
            arm, _ = self.ik(position=position, rotation=rotation, holding=holding)
        except SkillFailed as exc:
            raise SkillFailed(f"{label}: {exc}") from exc
        for i in range(limit):
            if i and i % 8 == 0:
                # Keep the last reachable target; convergence is checked below.
                with contextlib.suppress(SkillFailed):
                    arm, _ = self.ik(position=position, rotation=rotation, holding=holding)
            self.step(arm=arm, label=label)
            actual = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3))
            angle = (Rotation.from_matrix(rotation) * actual.inv()).magnitude()
            if (
                np.linalg.norm(self.data.site_xpos[self.site] - position) < position_tolerance
                and angle < orientation_tolerance
            ):
                return
        raise SkillFailed(
            f"Physical move failed: {label}, target={np.round(position, 3)}, "
            f"actual={np.round(self.data.site_xpos[self.site], 3)}"
        )

    def path(
        self,
        *,
        position: Any,
        rotation: np.ndarray,
        label: str,
        count: int = 6,
        holding: bool = False,
        position_tolerance: float = 0.012,
    ) -> None:
        """Cartesian straight line with a slerped orientation, re-solving IK as it
        goes; for open space. Near the cupboard use a planned `route` instead."""
        start = self.site_pose()
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], rotation]))
        for t in np.linspace(0, 1, count + 1)[1:]:
            self.move(
                position=(1 - t) * start[:3, 3] + t * np.asarray(position),
                rotation=rotations(t).as_matrix(),
                label=label,
                holding=holding,
                position_tolerance=position_tolerance if t < 1 else 0.008,
                orientation_tolerance=0.08 if t < 1 else 0.05,
            )

    def site_at(self, *, base: np.ndarray, arm: np.ndarray) -> np.ndarray:
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.ba] = base
        d.qpos[self.qa] = arm
        self.mujoco.mj_kinematics(self.model, d)
        return Frames.pose(
            rotation=d.site_xmat[self.site].reshape(3, 3).copy(),
            position=d.site_xpos[self.site].copy(),
        )

    def plan_route(
        self,
        *,
        waypoints: list[np.ndarray],
        base: np.ndarray,
        holding: bool,
        start_arm: np.ndarray | None = None,
        allowance: float = 0.0,
    ) -> list[np.ndarray] | None:
        """Arm joints for straight Cartesian lines through `waypoints` (hand poses)
        from wherever the hand is with the arm at `start_arm` and the base at `base`,
        each IK seeded from the previous so the route stays on one branch; None if any
        pose penetrates by more than `allowance` or the branch would change."""
        q = self.data.qpos[self.qa].copy() if start_arm is None else start_arm.copy()
        here = self.site_at(base=base, arm=q)
        route = []
        for waypoint in waypoints:
            for pose in Frames.line(start=here, end=waypoint, step=ROUTE_STEP):
                try:
                    nxt, collision = self.ik(
                        position=pose[:3, 3],
                        rotation=pose[:3, :3],
                        base=base,
                        holding=holding,
                        seed=q,
                    )
                except SkillFailed:
                    return None
                jump = float(np.max(np.abs((nxt - q + np.pi) % (2 * np.pi) - np.pi)))
                if collision > allowance or jump > MAX_JOINT_JUMP:
                    return None
                if holding and self.in_hand is not None and self.carried_wiper_hits(pose=pose):
                    return None
                route.append(nxt)
                q = nxt
            here = waypoint
        return route

    def carried_wiper_drive_hits(self, *, base: np.ndarray) -> bool:
        """Whether driving the base from where it is to `base`, the arm held where it
        is, carries the held wiper into the cupboard or the box, the route's start pose
        at `base` included. Unchecked, the drive to the slot base put the wiper 7.7 mm
        into the back panel, the grasp slipped 22 mm there, and the put-down stalled
        (pomdp/0). Two paths are checked, since the base follows neither exactly: the
        straight line, and the one `move_base` commands (each axis stepped by at most
        `base_step_limit`, so the shorter axes arrive first). On pomdp/0 the first
        reaches 7.3 mm into the panel, the second 3.5 mm, and the base as it really
        drove, its heading wavering, 9.9 mm."""
        arm = self.data.qpos[self.qa].copy()
        start = self.data.qpos[self.ba].copy()
        target = np.asarray(base, dtype=float)
        steps = int(np.ceil(np.max(np.abs(target - start)) / self.base_step_limit))
        commanded = [start]
        for _ in range(steps):
            step = np.clip(target - commanded[-1], -self.base_step_limit, self.base_step_limit)
            commanded.append(commanded[-1] + step)
        straight = [start + t * (target - start) for t in np.linspace(0.0, 1.0, steps + 1)]
        return any(
            self.carried_wiper_hits(pose=self.site_at(base=here, arm=arm))
            for here in [*straight, *commanded, target]
        )

    def carried_wiper_hits(self, *, pose: np.ndarray) -> bool:
        """Whether the held wiper, carried rigidly at hand pose `pose` with the grasp as
        last measured, would cut into the cupboard or the box (in the IK model)."""
        assert self.in_hand is not None
        d = self.ik_data
        wiper = pose @ self.in_hand
        address = self.model.jnt_qposadr[self.model.joint("wiper_0_joint").id]
        d.qpos[address : address + 3] = wiper[:3, 3]
        d.qpos[address + 3 : address + 7] = Rotation.from_matrix(wiper[:3, :3]).as_quat()[
            [3, 0, 1, 2]
        ]
        self.mujoco.mj_kinematics(self.model, d)
        self.mujoco.mj_collision(self.model, d)
        geoms = set(self.wiper_geoms)
        fixed = self.cupboard_geoms | self.box_geoms
        for contact in d.contact[: d.ncon]:
            a, b = contact.geom1, contact.geom2
            if ((a in geoms and b in fixed) or (b in geoms and a in fixed)) and (
                -contact.dist > CARRIED_WIPER_ALLOWANCE
            ):
                return True
        return False

    def follow(self, *, route: list[np.ndarray], label: str, settle: float = 0.02) -> None:
        """Track a planned route joint by joint; the last configuration is reached to
        within `settle` rad or the move raises."""
        for index, q in enumerate(route):
            last = index == len(route) - 1
            for _ in range(80 if last else 20):
                self.step(arm=q, label=label)
                error = np.abs((q - self.data.qpos[self.qa] + np.pi) % (2 * np.pi) - np.pi)
                if np.max(error) < (settle if last else ROUTE_TRACKING):
                    break
        error = np.abs((route[-1] - self.data.qpos[self.qa] + np.pi) % (2 * np.pi) - np.pi)
        if np.max(error) > 0.1:
            raise SkillFailed(f"{label}: arm stopped {np.max(error):.2f} rad short (blocked)")

    def settle_at(self, *, pose: np.ndarray, label: str, holding: bool = False) -> None:
        """Finish a route on the exact pose: a route is tracked joint by joint to within
        a few hundredths of a radian, which leaves the pads a centimetre off a 15 mm
        handle."""
        self.move(
            position=pose[:3, 3],
            rotation=pose[:3, :3],
            label=label,
            limit=60,
            position_tolerance=0.003,
            orientation_tolerance=0.03,
            holding=holding,
        )

    def first_route(
        self,
        *,
        waypoints: list[np.ndarray],
        bases: list[np.ndarray],
        holding: bool,
        allowance: float = 0.0,
    ) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """First base pose with a collision-free single-branch route, the arm starting
        from where it is now. Holding the wiper, a base is also refused if the drive
        there, the arm held still, carries the wiper into the cupboard or the box
        (`carried_wiper_drive_hits`): the route is checked only from the base."""
        for base in bases:
            if holding and self.in_hand is not None and self.carried_wiper_drive_hits(base=base):
                self.notes.append(
                    f"route: base {np.round(base, 3).tolist()} refused, the drive there "
                    "carries the wiper into the cupboard or box"
                )
                continue
            route = self.plan_route(
                waypoints=waypoints, base=base, holding=holding, allowance=allowance
            )
            if route is not None:
                return base, route
        return None

    def raise_hand(self, *, z: float, label: str, holding: bool) -> None:
        """Straight up toward height `z`, 3 cm at a time, stopping at the arm's reach."""
        here = self.site_pose()
        for height in np.arange(here[2, 3] + 0.03, z + 1e-9, 0.03):
            try:
                self.move(
                    position=[*here[:2, 3], height],
                    rotation=here[:3, :3],
                    label=label,
                    holding=holding,
                    position_tolerance=0.012,
                    orientation_tolerance=0.08,
                )
            except SkillFailed:
                return

    # --- the wiper ---------------------------------------------------------------

    def wiper_grasp(self) -> np.ndarray:
        """World hand pose for the fixed grasp on the wiper as it lies now."""
        return self.wiper_pose() @ self.grasp_in_wiper

    def leaned_handle_grasp(
        self, *, along: float | None, lean_sign: float, lean_deg: float = WIPER_LEAN_DEG
    ) -> np.ndarray:
        """Pinch-site pose for a lean grasp on the handle, `along` metres from the head
        centre (None: as near the centre of mass as the pads fit beside the head).
        lean_sign +1 leans the approach toward the handle's free end, so the wrist sits
        over the head; -1 leans it toward the head, the wrist over the free end."""
        u = self.stem_direction()
        head = self.data.geom_xpos[self.head_geom]
        handle = self.data.geom_xpos[self.handle_geom]
        com = self.data.subtree_com[self.wiper_body]
        head_half_width = float(np.min(self.model.geom_size[self.head_geom][:2]))
        lo = head_half_width + PAD_HALF_LENGTH + 0.006
        hi = float(self.model.geom_size[self.handle_geom][0] * 2) - PAD_HALF_LENGTH
        s = float(np.clip((com - head) @ u if along is None else along, lo, hi))
        point = head + s * u
        point[2] = handle[2]
        closing = np.array([-u[1], u[0], 0.0])
        rotation = Frames.leaned(closing=closing, lean_toward=lean_sign * u, lean_deg=lean_deg)
        return Frames.pose(rotation=rotation, position=point - WIPER_STANDOFF * rotation[:, 2])

    def slot_bases(self, *, y: float) -> list[np.ndarray]:
        return [
            np.array([x, y + dy, 0.0])
            for x in (BASE_X, BASE_X - 0.08, BASE_X - 0.16, BASE_X - 0.24)
            for dy in (0.0, 0.1, -0.1, 0.2, -0.2)
        ]

    def close_and_check(self, *, label: str, count: int = 25) -> bool:
        """Close the gripper and test the grasp by lifting the hand GRASP_TEST_LIFT:
        the wiper must rise at least half as far."""
        self.grip = 1.0
        self.hold(count=count, label=f"{label}_close")
        self.closed_in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        wiper = float(self.data.xpos[self.wiper_body][2])
        hand = float(self.data.site_xpos[self.site][2])
        lift = self.site_pose()
        lift[2, 3] += GRASP_TEST_LIFT
        with contextlib.suppress(SkillFailed):
            self.path(
                position=lift[:3, 3],
                rotation=lift[:3, :3],
                label=f"{label}_test",
                count=2,
                holding=True,
            )
        rose = float(self.data.xpos[self.wiper_body][2]) - wiper
        moved = float(self.data.site_xpos[self.site][2]) - hand
        return moved > 0.005 and rose > 0.5 * moved

    def take_wiper(self, *, label: str) -> None:
        """Record the grasp just closed, and how far the measured hand-to-wiper pose is
        from the fixed one: the friction grasp is the same grasp every time only if
        this stays small. Measured as the gripper closed, before the test lift lets
        the wiper settle a few millimetres in the pads."""
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        nominal = np.linalg.inv(self.grasp_in_wiper)
        closed = self.closed_in_hand
        hand_in_wiper = np.linalg.inv(closed)
        turn = Rotation.from_matrix(closed[:3, :3] @ nominal[:3, :3].T).magnitude()
        self.grasps_taken.append({
            "label": label,
            "tick": self.ticks,
            "hand_in_wiper_xyz": [round(float(v), 4) for v in hand_in_wiper[:3, 3]],
            "hand_in_wiper_rotvec_deg": [
                round(float(v), 2)
                for v in np.rad2deg(Rotation.from_matrix(hand_in_wiper[:3, :3]).as_rotvec())
            ],
            "offset_mm": round(float(np.linalg.norm(closed[:3, 3] - nominal[:3, 3])) * 1000, 1),
            "offset_deg": round(float(np.rad2deg(turn)), 2),
        })

    def let_go_of_wiper(self, *, label: str) -> None:
        self.grip = 0.0
        self.in_hand = None
        self.hold(count=20, label=f"{label}_release")

    def pick_wiper(self) -> None:
        """Take the wiper off its home on top of the cupboard by friction, at the
        grasp it sweeps with, and lift it clear. The hand comes in level, pads 2 cm
        above the handle, along a route planned joint by joint so the arm never changes
        IK branch (and swings through the cupboard) on the way."""
        if self.in_hand is not None:
            raise SkillFailed("pick_wiper: already holding the wiper")
        self.grip = 0.0
        found = self.shelf_wiper_route(target=None)
        if found is None:
            raise SkillFailed("pick_wiper: no collision-free route into the wiper slot")
        base, route, grasp = found
        self.move_base(target=base, label="drive_to_cupboard")
        self.mark(label="at cupboard")
        self.follow(route=route, label="into_wiper_slot")
        self.settle_at(pose=grasp, label="onto_handle")
        self.mark(label="pads around handle")
        if not self.close_and_check(label="pick_wiper"):
            raise SkillFailed("pick_wiper: the wiper did not come with the hand")
        self.take_wiper(label="pick_wiper")
        self.leave_wiper_slot(grasp=grasp, holding=True, label="lift_wiper")
        self.mark(label=f"lifted, wiper +{self.wiper_lift():.2f} m")
        if self.wiper_lift() < 0.05:
            raise SkillFailed(f"pick_wiper: wiper not lifted ({self.wiper_lift():.3f} m)")

    def shelf_wiper_route(
        self, *, target: np.ndarray | None
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray] | None:
        """(base, route, final hand pose) into the wiper slot: to the handle as the
        wiper lies (target None), or, holding it, to the hand pose that sets it down
        at `target` (a wiper body pose)."""
        options = []
        if target is None:
            options.append(self.wiper_grasp())
        else:
            assert self.in_hand is not None
            options.append(target @ np.linalg.inv(self.in_hand))
        for grasp in options:
            pre = grasp.copy()
            pre[2, 3] += SLOT_ENTRY_RISE
            porch = pre.copy()
            porch[0, 3] = self.front_x - 0.12
            found = self.first_route(
                waypoints=[porch, pre, grasp],
                bases=self.slot_bases(y=float(grasp[1, 3])),
                holding=target is not None,
            )
            if found is not None:
                return found[0], found[1], grasp
        return None

    def leave_wiper_slot(self, *, grasp: np.ndarray, holding: bool, label: str) -> None:
        lift = grasp.copy()
        lift[2, 3] += SLOT_ENTRY_RISE
        out = lift.copy()
        out[0, 3] = self.front_x - 0.15
        clear = out.copy()
        clear[:3, 3] += [-0.05, 0.0, 0.15]
        route = self.plan_route(
            waypoints=[lift, out, clear],
            base=self.data.qpos[self.ba].copy(),
            holding=holding,
            allowance=0.01,
        )
        if route is None:
            for waypoint, count in ((lift, 4), (out, 10), (clear, 4)):
                self.path(
                    position=waypoint[:3, 3],
                    rotation=waypoint[:3, :3],
                    label=label,
                    count=count,
                    holding=holding,
                )
            return
        self.follow(route=route, label=label, settle=0.05)

    def wiper_target(self, *, head: Any, stem: Any = SWEEP_STEM) -> np.ndarray:
        """Wiper body pose: flat as it started, handle pointing along `stem` (by default
        at the robot, -x), head centre at `head`."""
        u = self.stem_start
        yaw = np.arctan2(stem[1], stem[0]) - np.arctan2(u[1], u[0])
        turn = Frames.pose(rotation=Rotation.from_euler("z", yaw).as_matrix(), position=head)
        return turn @ Frames.pose(rotation=np.eye(3), position=-self.head_start) @ self.wiper_start

    def move_wiper(self, *, head: Any, label: str, stem: Any = SWEEP_STEM) -> None:
        """Open loop on the last measured grasp: the hand goes where the wiper would be
        at `head` if it did not move in the grasp."""
        assert self.in_hand is not None
        desired = self.wiper_target(head=head, stem=stem)
        target = desired @ np.linalg.inv(self.in_hand)
        distance = float(np.linalg.norm(target[:3, 3] - self.data.site_xpos[self.site]))
        self.path(
            position=target[:3, 3],
            rotation=target[:3, :3],
            label=label,
            count=max(1, int(np.ceil(distance / 0.03))),
            holding=True,
            position_tolerance=0.02,
        )
        error = float(np.linalg.norm(desired[:3, 3] - self.wiper_pose()[:3, 3]))
        if error > 0.05:
            raise SkillFailed(
                f"{label}: wiper {error:.3f} m off target with the hand on target "
                f"(turned {self.slip_deg:.0f} deg in the grasp)"
            )

    def wiper_home_support(self) -> float:
        """Height of the board the wiper rests on at home. The wiper's body origin is
        its bottom face, so this is not its start less the head's half-height, which is
        7.5 mm below the board."""
        return float(self.board[WIPER_SHELF])

    def put_down_reached(self, *, support: float, gap: float) -> bool:
        """Whether the held wiper is level (`PUT_DOWN_LEVEL_DEG`) with its head within
        `gap` above `support`: close enough to let go, though the hand stopped short."""
        tilt = float(np.rad2deg(np.arccos(np.clip(self.wiper_pose()[2, 2], -1.0, 1.0))))
        return tilt < PUT_DOWN_LEVEL_DEG and self.head_bottom() - support < gap

    def level_target(self, *, x: float, above_board: float) -> np.ndarray:
        """Hand pose that would take the wiper from where it is now to flat, head centre
        at `x`, head bottom `above_board` over the cube shelf: the measured wiper, not
        the grasp at closing, so tilt in the grasp is taken out."""
        now = self.wiper_pose()
        head = self.data.geom_xpos[self.head_geom].copy()
        stem = self.stem_direction()
        level = np.array([stem[0], stem[1], 0.0])
        level /= np.linalg.norm(level)
        axis = np.cross(stem, level)
        angle = float(np.arcsin(np.clip(np.linalg.norm(axis), 0.0, 1.0)))
        turn = (
            Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
            if angle > 1e-6
            else np.eye(3)
        )
        desired = (
            Frames.pose(rotation=turn, position=head)
            @ Frames.pose(rotation=np.eye(3), position=-head)
            @ now
        )
        half = float(self.model.geom_size[self.head_geom][2])
        drop = head[2] - half - (self.board[CUBE_SHELF] + above_board)
        desired[:3, 3] += [x - head[0], 0.0, -float(np.clip(drop, -0.01, 0.01))]
        return desired @ np.linalg.inv(now) @ self.site_pose()

    def base_for(self, *, target: np.ndarray, ahead_of: float) -> tuple[np.ndarray, np.ndarray]:
        """(base, arm) about HEAD_AHEAD behind the head that reaches `target`."""
        tried = []
        # Heading 0 first, as before the angled stroke; the turned headings only where
        # none of those reaches (an angled wiper, e.g. reseated at its angle).
        for heading, ahead in [(h, a) for h in (0.0, 0.4, -0.4, 0.8, -0.8) for a in HEAD_AHEADS]:
            base = np.array([min(ahead_of - ahead, BASE_X), target[1, 3], heading])
            try:
                arm, collision = self.ik(
                    position=target[:3, 3], rotation=target[:3, :3], base=base, holding=True
                )
            except SkillFailed:
                tried.append(f"{base[0]:.2f}: no IK")
                continue
            # With the head near the board the pad tips can touch it too; the arm itself
            # must stay clear.
            if collision <= CONTACT_PENETRATION and not self.arm_touches(d=self.ik_data):
                return base, arm
            tried.append(f"{base[0]:.2f}: {collision:.3f} m {self.touching(d=self.ik_data)}")
        raise SkillFailed(f"no base pose reaches the hand target: {'; '.join(tried)}")

    def wiper_to(self, *, head: Any, label: str) -> None:
        """Head to `head`, the wiper flat with its handle pointing at the robot."""
        assert self.in_hand is not None
        target = self.wiper_target(head=head) @ np.linalg.inv(self.in_hand)
        base, _ = self.base_for(target=target, ahead_of=float(head[0]))
        if np.max(np.abs(base - self.data.qpos[self.ba])) > 0.002:
            self.move_base(target=base, label=label)
        self.move_wiper(head=head, label=label)

    def hand_to(
        self,
        *,
        target: np.ndarray,
        label: str,
        tolerance: float = 0.006,
        limit: int = CONTACT_STEP_LIMIT,
    ) -> None:
        """Near or on the board: a bounded move of the hand with the base following,
        the shortfall counted rather than treated as a failure."""
        base, _ = self.base_for(
            target=target, ahead_of=float(self.data.geom_xpos[self.head_geom][0])
        )
        if np.max(np.abs(base - self.data.qpos[self.ba])) > 0.002:
            self.move_base(target=base, label=label)
        try:
            self.move(
                position=target[:3, 3],
                rotation=target[:3, :3],
                label=label,
                limit=limit,
                holding=True,
                position_tolerance=tolerance,
                orientation_tolerance=0.1,
            )
        except SkillFailed:
            self.contact_shortfalls += 1

    def recalibrate_grasp(self) -> None:
        """Re-measure the hand-to-wiper transform while the wiper hangs free: it droops
        a little in the friction grasp under its own weight."""
        assert self.in_hand is not None
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()

    def reseat_wiper(self, *, label: str) -> None:
        """Lay the wiper flat on the swept strip of the cube shelf, let go and take it
        again at the sweep grasp: each stroke's drag turns it 10-30 deg in the friction
        grasp, and carried like that the wrist no longer clears the board above."""
        lift = self.site_pose()
        lift[2, 3] += HOVER
        self.hand_to(target=lift, label=label)
        back = self.site_pose()
        head = self.data.geom_xpos[self.head_geom]
        back[0, 3] += RESEAT_HEAD_X - float(head[0])
        back[1, 3] += self.reseat_y(near=float(head[1])) - float(head[1])
        self.hand_to(target=back, label=label)
        for _ in range(3):
            self.hand_to(target=self.level_target(x=RESEAT_HEAD_X, above_board=HOVER), label=label)
        for _ in range(LOWER_STEPS // 5):
            if self.head_bottom() <= self.board[CUBE_SHELF] + 0.002:
                break
            self.hand_to(target=self.level_target(x=RESEAT_HEAD_X, above_board=0.0), label=label)
        self.notes.append(f"{label}: set down after turning {self.slip_deg:.1f} deg in the grasp")
        # From the deliberate release until the wiper is held again, the sweep's
        # per-step `HoldingWiper` check would call the empty hand a lost wiper.
        with self.wiper_unwatched():
            self.let_go_of_wiper(label=label)
            up = self.site_pose()
            up[2, 3] += 0.02
            self.hand_to(target=up, label=f"{label}_release")
            self.regrasp_on_shelf(label=label)
        self.hand_to(
            target=self.level_target(x=RESEAT_HEAD_X, above_board=HOVER),
            label=f"{label}_lift",
        )

    def regrasp_on_shelf(self, *, label: str) -> None:
        for attempt in range(REGRASP_TRIES):
            grasp = self.wiper_grasp()
            pre = grasp.copy()
            pre[2, 3] += 0.025
            self.hand_to(target=pre, label=f"{label}_regrasp")
            self.hand_to(target=grasp, label=f"{label}_regrasp", tolerance=0.003)
            if self.close_and_check(label=f"{label}_regrasp"):
                self.take_wiper(label=label)
                return
            self.notes.append(f"{label}: regrasp {attempt + 1} missed, retrying")
            self.grip = 0.0
            self.hold(count=15, label=label)
            self.hand_to(target=pre, label=f"{label}_regrasp")
        raise SkillFailed(f"{label}: {REGRASP_TRIES} regrasps on the shelf missed")

    def reseat_y(self, *, near: float) -> float:
        """Nearest head y at which the wiper, laid flat with its head at RESEAT_HEAD_X,
        rests on bare board: no cube under the head or the handle."""
        cubes = [self.cube_position(cube=c) for c in self.shelf_cubes()]
        half = float(self.model.geom_size[self.head_geom][1]) + 0.015
        for y in sorted(
            np.arange(-STROKE_LIMIT, STROKE_LIMIT + 1e-9, 0.005), key=lambda v: abs(v - near)
        ):
            under_head = any(
                abs(p[0] - RESEAT_HEAD_X) < 0.035 and abs(p[1] - y) < half for p in cubes
            )
            under_handle = any(p[0] < RESEAT_HEAD_X and abs(p[1] - y) < HANDLE_BAND for p in cubes)
            if not under_head and not under_handle:
                return float(y)
        raise SkillFailed("reseat: no bare strip of board to lay the wiper on")

    def head_over_cupboard(self) -> bool:
        """Whether the wiper head's footprint, turned as it is, reaches past the
        cupboard's open front face."""
        rotation = self.data.geom_xmat[self.head_geom].reshape(3, 3)
        reach = float(np.abs(rotation[0]) @ self.model.geom_size[self.head_geom])
        return float(self.data.geom_xpos[self.head_geom][0]) + reach > self.front_x

    def level_target_at(self, *, head: Any, above_board: float) -> np.ndarray:
        """`level_target` with the head moved to (x, y): the measured wiper levelled
        about its head, its yaw kept, head bottom `above_board` over the cube shelf."""
        here = self.data.geom_xpos[self.head_geom].copy()
        target = self.level_target(x=float(here[0]), above_board=above_board)
        target[0, 3] += float(head[0]) - float(here[0])
        target[1, 3] += float(head[1]) - float(here[1])
        return target

    @staticmethod
    def angled_base_options(*, hand: np.ndarray, lane: "AngledLane") -> list[np.ndarray]:
        """Base poses tried for the lowered stroke start. First the straight stroke's
        own base turned with the path -- HEAD_AHEAD behind the head along the pull,
        heading turned by the stroke angle, so the arm holds the wiper as it does for
        a straight stroke -- with x capped at BASE_X by sliding further along the
        pull. Then the linear-plough probe's search: x at most BASE_X and 0.55 m
        behind the hand, y beside it, heading turned up to 0.8 rad."""
        d = lane.direction
        heading = float(np.arctan2(-d[1], -d[0]))
        turned = []
        for ahead in HEAD_AHEADS:
            xy = lane.start + ahead * d
            if xy[0] > BASE_X:
                xy = lane.start + (lane.start[0] - BASE_X) / -d[0] * d
            turned.append(np.array([xy[0], xy[1], heading]))
        return turned + [
            np.array([min(bx, hand[0] - 0.55), hand[1] + dy, th])
            for bx in (BASE_X, 0.72, 0.64)
            for dy in (0.0, -0.1, 0.1, -0.2, 0.2)
            for th in (0.0, -0.4, 0.4, -0.8, 0.8)
        ]

    def arm_clear(self, *, target: np.ndarray, base: np.ndarray, seed: Any = None) -> Any:
        """Arm joints reaching `target` from `base` with the arm clear, or None."""
        try:
            q, collision = self.ik(
                position=target[:3, 3],
                rotation=target[:3, :3],
                base=base,
                holding=True,
                attempts=6,
                seed=seed,
            )
        except SkillFailed:
            return None
        if collision > CONTACT_PENETRATION or self.arm_touches(d=self.ik_data):
            return None
        # `ik` accepts a centimetre of residual; at full stretch near the board that
        # centimetre is the difference between the head on the board and over a cube.
        reached = self.site_at(base=np.asarray(base, dtype=float), arm=q)
        if float(np.linalg.norm(reached[:3, 3] - target[:3, 3])) > 0.004:
            return None
        return q

    @staticmethod
    def follow_base(*, base0: np.ndarray, head0: np.ndarray, head: Any) -> np.ndarray:
        """The base moved with the head (heading fixed), x capped at BASE_X."""
        b = np.asarray(base0, dtype=float).copy()
        b[0] = min(base0[0] + head[0] - head0[0], BASE_X)
        b[1] = base0[1] + head[1] - head0[1]
        return b

    @staticmethod
    def segment(*, a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
        """Points from a to b at most `step` apart, a excluded, b included."""
        count = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step)))
        return [a + t * (b - a) for t in np.linspace(0.0, 1.0, count + 1)[1:]]

    def angled_base(self, *, lane: AngledLane, low: float, hover: float) -> np.ndarray:
        """First base pose (`angled_base_options`) from which the hand, the base
        following the head, reaches every point of the hover reach-in and the lowered
        stroke on one IK branch; raises if none does."""
        assert self.in_hand is not None
        in_hand = self.in_hand
        stem = np.array([*lane.direction, 0.0])

        def hand(*, xy: np.ndarray, z: float) -> np.ndarray:
            return self.wiper_target(head=[xy[0], xy[1], z], stem=stem) @ np.linalg.inv(in_hand)

        start = hand(xy=lane.start, z=low)
        checks = [(lane.mouth, hover)]
        checks += [(p, hover) for p in self.segment(a=lane.mouth, b=lane.start, step=0.05)]
        checks += [(p, low) for p in self.segment(a=lane.start, b=lane.end, step=0.03)]
        reached = 0
        for base0 in self.angled_base_options(hand=start[:3, 3], lane=lane):
            if self.arm_clear(target=start, base=base0) is None:
                continue
            reached += 1
            q = None
            for xy, z in checks:
                b = self.follow_base(base0=base0, head0=lane.start, head=xy)
                q = self.arm_clear(target=hand(xy=xy, z=z), base=b, seed=q)
                if q is None:
                    break
            else:
                return base0
            if reached >= 6:
                break
        raise SkillFailed(
            f"angled stroke: no base reaches the whole stroke at angle {lane.angle:+.0f} "
            f"({reached} bases reached its start)"
        )

    def wiper_along(self, *, head: Any, stem: Any, base: np.ndarray, label: str) -> None:
        """Base to `base` (if it moved), then the head to `head`, handle along `stem`."""
        if np.max(np.abs(base - self.data.qpos[self.ba])) > 0.002:
            self.move_base(target=base, label=label)
        self.move_wiper(head=head, stem=stem, label=label)

    def angled_stroke(self, *, cube_xy: Any, offset: float, angle: float, name: str) -> AngledLane:
        """The straight `stroke` turned by `angle`: raise, base to the mouth, head over
        the box then to the mouth at hover, reach in along the line to the start,
        lower, pull along the line at SWEEP_SPEED and SWEEP_HEIGHT (closed loop on the
        measured wiper, the base following the head), lift, withdraw along the line to
        the mouth, turn the wiper back straight there, reseat if it turned in the
        grasp, and out."""
        lane = AngledLane.plan(cube_xy=cube_xy, offset=offset, angle=angle, front_x=self.front_x)
        if lane.shift:
            self.notes.append(f"{name}: line shifted {lane.shift:+.3f} m to stay inside")
        stem = np.array([*lane.direction, 0.0])
        flat = self.board[CUBE_SHELF] + float(self.model.geom_size[self.head_geom][2])
        hover = flat + HOVER
        low = flat + SWEEP_HEIGHT - PULL_PRESS
        self.phase = "raise"
        self.raise_hand(z=CARRY_Z, label=f"{name}_raise", holding=True)
        self.phase = "plan_base"
        base0 = self.angled_base(lane=lane, low=low, hover=hover)
        mouth_base = self.follow_base(base0=base0, head0=lane.start, head=lane.mouth)
        self.phase = "approach"
        self.move_base(target=mouth_base, label=f"{name}_base")
        if self.head_over_cupboard():
            head_z = float(self.data.geom_xpos[self.head_geom][2])
            self.move_wiper(head=[*lane.mouth, head_z], stem=stem, label=f"{name}_to_mouth")
        self.move_wiper(head=[*lane.mouth, hover + 0.12], stem=stem, label=f"{name}_above_box")
        self.move_wiper(head=[*lane.mouth, hover], stem=stem, label=f"{name}_mouth")
        self.recalibrate_grasp()
        for xy in self.segment(a=lane.mouth, b=lane.start, step=0.05):
            b = self.follow_base(base0=base0, head0=lane.start, head=xy)
            self.wiper_along(head=[*xy, hover], stem=stem, base=b, label=f"{name}_reach_in")
        self.recalibrate_grasp()
        self.phase = "lower"
        for _ in range(LOWER_STEPS):
            if abs(self.head_bottom() - self.board[CUBE_SHELF] - SWEEP_HEIGHT) < LOWER_TOLERANCE:
                break
            self.arm_to(
                target=self.level_target_at(head=lane.start, above_board=SWEEP_HEIGHT),
                label=f"{name}_lower",
                tolerance=0.003,
                limit=15,
            )
        self.mark(label=f"{name}: head behind cubes")
        self.phase = "pull"
        self.angled_pull(lane=lane, base0=base0, label=f"{name}_pull")
        self.mark(label=f"{name}: stroke end")
        self.phase = "lift"
        here = self.data.geom_xpos[self.head_geom][:2].copy()
        for _ in range(3):
            self.arm_to(
                target=self.level_target_at(head=here, above_board=HOVER),
                label=f"{name}_lift",
                tolerance=0.006,
                limit=CONTACT_STEP_LIMIT,
            )
        # Withdraw along the wiper's measured stem to the mouth (outside the cupboard),
        # then turn the whole robot -- base and arm rigidly, the arm's joints held --
        # about the head until the stem points at the robot again: the hand never moves
        # relative to the base, so nothing loads the friction grasp (turning the hand
        # instead dropped the wiper in 213/1960 grid strokes). From there it is the
        # straight stroke's pose at the mouth; the base slides (still rigidly) to a lane
        # the straight stroke uses and, if the wiper turned in the grasp, forward to the
        # straight stroke's end to reseat from there.
        self.recalibrate_grasp()
        self.phase = "withdraw"
        here = self.data.geom_xpos[self.head_geom][:2].copy()
        base_here = self.data.qpos[self.ba].copy()
        now = self.stem_direction()
        flat_stem = np.array([now[0], now[1], 0.0]) / max(1e-9, float(np.hypot(now[0], now[1])))
        mouth_x = self.front_x - 0.12
        mouth = np.array([
            mouth_x,
            float(here[1] + (mouth_x - here[0]) * flat_stem[1] / min(-1e-6, flat_stem[0])),
        ])
        for xy in self.segment(a=here, b=mouth, step=0.05):
            b = self.follow_base(base0=base_here, head0=here, head=xy)
            self.wiper_along(head=[*xy, hover], stem=flat_stem, base=b, label=f"{name}_withdraw")
        self.phase = "unturn"
        self.unturn_rigidly(label=f"{name}_unturn")
        head = self.data.geom_xpos[self.head_geom][:2].copy()
        lane_y = float(np.clip(head[1], -STROKE_LIMIT, STROKE_LIMIT))
        self.slide_base(dx=0.0, dy=lane_y - float(head[1]), label=f"{name}_to_lane")
        if self.slip_deg > RESEAT_TURN_DEG:
            self.phase = "reseat"
            head = self.data.geom_xpos[self.head_geom][:2].copy()
            self.slide_base(dx=STROKE_END_X - float(head[0]), dy=0.0, label=f"{name}_reseat_from")
            self.reseat_wiper(label=f"{name}_reseat")
        else:
            self.recalibrate_grasp()
        self.phase = "out"
        head = self.data.geom_xpos[self.head_geom]
        self.wiper_to(head=[mouth_x, float(head[1]), hover + 0.12], label=f"{name}_out")
        self.hold(count=15, label="settle")
        self.notes.append(f"{name}: {self.counts()['in_box']}/5 in box after the stroke")
        return lane

    def angled_pull(self, *, lane: AngledLane, base0: np.ndarray, label: str) -> None:
        """Pull the lowered head from `lane.start` to `lane.end` at SWEEP_SPEED,
        SWEEP_HEIGHT over the board, closed loop on the measured wiper every PULL_CHUNK
        control steps, the base following the head."""
        began = self.ticks
        q = None
        advance = SWEEP_SPEED * CONTROL_DT * PULL_CHUNK
        for xy in self.segment(a=lane.start, b=lane.end, step=advance):
            target = self.level_target_at(head=xy, above_board=SWEEP_HEIGHT - PULL_PRESS)
            b = self.follow_base(base0=base0, head0=lane.start, head=xy)
            try:
                q, _ = self.ik(
                    position=target[:3, 3],
                    rotation=target[:3, :3],
                    base=b,
                    holding=True,
                    attempts=6,
                    seed=q,
                )
            except SkillFailed:
                self.contact_shortfalls += 1
                continue
            self.base_target = b
            for _ in range(PULL_CHUNK):
                self.step(arm=q, base=b, label=label)
        here = self.data.geom_xpos[self.head_geom][:2].copy()
        travelled = float(np.linalg.norm(here - lane.start))
        self.pull_speeds.append(travelled / max(1, self.ticks - began) / CONTROL_DT)
        self.hold(count=10, label=label)

    @staticmethod
    def rotated_base(*, base: np.ndarray, pivot: Any, angle: float) -> np.ndarray:
        """The base pose turned by `angle` (rad) about the vertical through `pivot`."""
        c, s = np.cos(angle), np.sin(angle)
        rel = np.asarray(base[:2], dtype=float) - np.asarray(pivot[:2], dtype=float)
        xy = np.asarray(pivot[:2], dtype=float) + np.array([
            c * rel[0] - s * rel[1],
            s * rel[0] + c * rel[1],
        ])
        return np.array([xy[0], xy[1], float(base[2]) + angle])

    @staticmethod
    def stem_yaw(*, stem: Any) -> float:
        """The stem's turn from pointing at the robot (-x), > 0 toward +y; turning the
        robot by this much about the head brings it back to -x."""
        return float(np.arctan2(stem[1], -stem[0]))

    def unturn_rigidly(self, *, label: str, step: float = 0.05) -> None:
        """Turn base and arm together (arm joints held) about the head until the stem
        points at -x, `step` rad at a time."""
        for _ in range(40):
            left = self.stem_yaw(stem=self.stem_direction())
            if abs(left) < np.deg2rad(1.5):
                return
            pivot = self.data.geom_xpos[self.head_geom][:2].copy()
            base = self.rotated_base(
                base=self.data.qpos[self.ba].copy(),
                pivot=pivot,
                angle=float(np.clip(left, -step, step)),
            )
            self.move_base(target=base, label=label)
        raise SkillFailed(f"{label}: the stem did not come back to -x")

    def slide_base(self, *, dx: float, dy: float, label: str) -> None:
        """Translate the base (arm joints held), so the hand and wiper move rigidly,
        x capped at BASE_X."""
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return
        base = self.data.qpos[self.ba].copy()
        base[0] = min(base[0] + dx, BASE_X)
        base[1] += dy
        self.move_base(target=base, label=label)

    def arm_to(self, *, target: np.ndarray, label: str, tolerance: float, limit: int) -> None:
        """`hand_to` with the base held where it is (the angled stroke's base is chosen
        for the whole stroke): a shortfall is counted, not a failure."""
        try:
            self.move(
                position=target[:3, 3],
                rotation=target[:3, :3],
                label=label,
                limit=limit,
                holding=True,
                position_tolerance=tolerance,
                orientation_tolerance=0.1,
            )
        except SkillFailed:
            self.contact_shortfalls += 1

    def sweep_shelf(self, *, cube: str, offset: float, angle: float) -> None:
        """One stroke (`angled_stroke`) on the line `offset` beside `cube`, turned by
        `angle` deg, at SWEEP_HEIGHT and SWEEP_SPEED. Needs the wiper in hand (from
        `pick_wiper`) and `cube` on the shelf; raises unless `cube` ends in the box.
        Other cubes the stroke moves are not this skill's concern."""
        params = SweepParams(offset=offset, angle=angle)
        if self.in_hand is None:
            raise SkillFailed("sweep_shelf: the wiper is not in hand")
        if cube not in self.shelf_cubes():
            raise SkillFailed(f"sweep_shelf: {cube} is not on the shelf")
        self.watch_wiper, self.wiper_unheld_steps = True, 0
        try:
            self.angled_stroke(
                cube_xy=self.cube_position(cube=cube)[:2],
                offset=params.offset,
                angle=params.angle,
                name=f"{cube}_o{params.offset:+.3f}_a{params.angle:+.0f}",
            )
        except WiperLost as exc:
            raise SkillFailed(f"sweep_shelf [{self.phase}]: {exc}") from exc
        except SkillFailed as exc:
            raise SkillFailed(f"[{self.phase}] {exc}") from exc
        finally:
            self.watch_wiper = False
        if not self.core.cube_in_box(cube):
            raise SkillFailed(f"sweep_shelf: {cube} is not in the box after the stroke")

    def return_wiper(self) -> None:
        """Put the held wiper back where it started on top of the cupboard, at the
        grasp it is held by."""
        if self.in_hand is None:
            raise SkillFailed("return_wiper: the wiper is not in hand")
        grasp = self.into_wiper_slot()
        if grasp is None:
            # The grasp moved on the way in, and from the grasp as it is now the
            # put-down would collide: back out with the wiper rather than stall
            # against the board above.
            here = self.site_pose()
            self.leave_wiper_slot(grasp=here, holding=True, label="return_back_out")
            raise SkillFailed("return_wiper: the put-down collides after the way in")
        try:
            self.settle_at(pose=grasp, label="return_onto_shelf", holding=True)
        except SkillFailed as exc:
            if not self.put_down_reached(support=self.wiper_home_support(), gap=SLOT_PUT_DOWN_GAP):
                raise
            self.notes.append(f"return_wiper: let go on the shelf short of the target ({exc})")
        self.let_go_of_wiper(label="return")
        self.leave_wiper_slot(grasp=grasp, holding=False, label="return_withdraw")
        error = float(np.linalg.norm(self.wiper_pose()[:3, 3] - self.wiper_start[:3, 3]))
        self.mark(label=f"wiper returned, {error:.3f} m from its start")
        if error > 0.05 or abs(self.wiper_lift()) > 0.01:
            raise SkillFailed(f"return_wiper: wiper ended {error:.3f} m from where it started")

    def into_wiper_slot(self) -> np.ndarray | None:
        """Raise, plan into the wiper slot, drive and follow the route in, then re-aim
        the put-down with the grasp as it is now (`recalibrate_grasp`). Returns that
        put-down hand pose, or None if a straight move from here to it would collide,
        which nothing checked before: from a grasp that slipped on the way in it
        stalled against the board above."""
        self.raise_hand(z=CARRY_Z, label="return_raise", holding=True)
        # Aim with the grasp as it is now: 13 cm from the centre of mass the head droops
        # in the friction grasp, and aimed with the grasp at closing it meets the board
        # edge before the slot.
        self.recalibrate_grasp()
        found = self.shelf_wiper_route(target=self.wiper_start)
        if found is None:
            raise SkillFailed("return_wiper: no collision-free route into the wiper slot")
        base, route, _ = found
        self.move_base(target=base, label="return_to_cupboard")
        self.follow(route=route, label="return_into_slot", settle=0.04)
        self.recalibrate_grasp()
        assert self.in_hand is not None
        grasp = self.wiper_start @ np.linalg.inv(self.in_hand)
        here = self.data.qpos[self.ba].copy()
        if self.plan_route(waypoints=[grasp], base=here, holding=True) is None:
            return None
        return grasp

    # --- cubes -------------------------------------------------------------------

    def box_drop_xy(self) -> tuple[float, float]:
        """The box's inner centre in the live scene: where a cube is let go over it."""
        (x0, y0, x1, y1), _ = self.core.box_interior()
        return (float((x0 + x1) / 2), float((y0 + y1) / 2))

    def drop_cube_into_box(self, *, cube: str = "cube_0") -> None:
        """Documented scene setup, not a skill: teleport one cube above the box."""
        state = self.core.get_object(cube)
        state.set_pose([*self.box_drop_xy(), 0.10], [1.0, 0.0, 0.0, 0.0])
        state.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.hold(count=30, label="cube_settles_in_box")
        if not self.core.cube_in_box(cube):
            raise SkillFailed("Dropped cube did not settle in the box")

    def cube_grasp(
        self, *, cube: str, centre: Any, lean_deg: float, face: int, sign: float
    ) -> np.ndarray:
        """Pads on the cube's `face` (0 or 1) side faces; the approach leans `lean_deg`
        from vertical toward `sign` times the horizontal normal of the other faces."""
        quaternion = self.core.get_object(cube).get_pose()[1]
        axes = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        closing = axes[:, face].copy()
        closing[2] = 0.0
        closing /= np.linalg.norm(closing)
        toward = sign * np.array([-closing[1], closing[0], 0.0])
        rotation = Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean_deg)
        return Frames.pose(
            rotation=rotation, position=np.asarray(centre) - CUBE_STANDOFF * rotation[:, 2]
        )

    def reachable_base(
        self,
        *,
        poses: list[np.ndarray],
        holding: bool,
        xs: tuple[float, ...] = (BASE_X, BASE_X - 0.1, BASE_X - 0.2),
        ys: tuple[float, ...] = (0.0, -0.15, 0.15, -0.3, 0.3),
    ) -> np.ndarray | None:
        """First base pose from which every pose is reached collision-free."""
        for x in xs:
            for y in ys:
                base = np.array([x, y, 0.0])
                clean = True
                for pose in poses:
                    try:
                        _, collision = self.ik(
                            position=pose[:3, 3],
                            rotation=pose[:3, :3],
                            base=base,
                            holding=holding,
                        )
                    except SkillFailed:
                        clean = False
                        break
                    if collision > 0:
                        clean = False
                        break
                if clean:
                    return base
        return None

    def grip_cube(self, *, cube: str, label: str) -> None:
        self.grip = 1.0
        self.hold(count=25, label=f"{label}_close")
        self.held_cube = cube

    def release_cube(self, *, label: str) -> None:
        self.grip = 0.0
        self.hold(count=25, label=f"{label}_release")
        self.held_cube = None

    def check_lifted(self, *, cube: str, above: float, label: str) -> None:
        z = float(self.cube_position(cube=cube)[2])
        self.mark(label=f"{cube} lifted to z={z:.3f}")
        if z < above:
            self.release_cube(label=label)
            raise SkillFailed(f"{label}: {cube} not lifted (z={z:.3f})")

    def pick_cube_from_box(self, *, cube: str) -> None:
        """Out of the box only a steep grasp clears the box walls."""
        grasp, base = self.open_space_cube_grasp(cube=cube, leans=(30.0, 45.0), toward_robot=None)
        self.take_cube_in_open_space(cube=cube, grasp=grasp, base=base, label="from_box")
        self.check_lifted(cube=cube, above=0.12, label="pick_cube_from_box")

    def pick_cube_from_floor(self, *, cube: str) -> None:
        """Pick a cube off the floor. The gripper leans toward the cupboard (wrist on
        the robot's side), the one orientation that can later place it on the shelf."""
        if self.in_hand is not None or self.held_cube is not None:
            raise SkillFailed("pick_cube_from_floor: the hand is not empty")
        position = self.cube_position(cube=cube)
        xs = tuple(float(position[0] - d) for d in (0.45, 0.55, 0.65))
        ys = tuple(float(position[1] + d) for d in (0.0, -0.15, 0.15))
        grasp, base = self.open_space_cube_grasp(
            cube=cube, leans=CUBE_LEANS_DEG, toward_robot=True, xs=xs, ys=ys
        )
        self.take_cube_in_open_space(cube=cube, grasp=grasp, base=base, label="from_floor")
        self.check_lifted(cube=cube, above=0.05, label="pick_cube_from_floor")

    def open_space_cube_grasp(
        self, *, cube: str, leans: tuple[float, ...], toward_robot: bool | None, **bases: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        centre = self.cube_position(cube=cube)
        for lean in leans:
            for face in (0, 1):
                for sign in (1.0, -1.0):
                    grasp = self.cube_grasp(
                        cube=cube, centre=centre, lean_deg=lean, face=face, sign=sign
                    )
                    approach = grasp[:3, 2]
                    if toward_robot is not None and (approach[0] > 0.2) != toward_robot:
                        continue
                    base = self.reachable_base(
                        poses=[self.cube_pre(grasp=grasp), grasp], holding=False, **bases
                    )
                    if base is not None:
                        return grasp, base
        raise SkillFailed(f"No collision-free grasp on {cube} at {np.round(centre, 3)}")

    @staticmethod
    def cube_pre(*, grasp: np.ndarray) -> np.ndarray:
        pre = grasp.copy()
        pre[:3, 3] += -0.10 * grasp[:3, 2] + np.array([0.0, 0.0, 0.08])
        return pre

    def take_cube_in_open_space(
        self, *, cube: str, grasp: np.ndarray, base: np.ndarray, label: str
    ) -> None:
        if self.held_cube is not None or self.in_hand is not None:
            raise SkillFailed(f"{label}: the hand is not empty")
        pre = self.cube_pre(grasp=grasp)
        self.grip = 0.0
        self.raise_hand(z=0.35, label=f"{label}_raise", holding=False)
        self.move_base(target=base, label=f"{label}_drive")
        self.path(position=pre[:3, 3], rotation=pre[:3, :3], label=f"{label}_above", count=8)
        self.path(position=grasp[:3, 3], rotation=grasp[:3, :3], label=f"{label}_down", count=6)
        self.grip_cube(cube=cube, label=label)
        self.path(
            position=pre[:3, 3], rotation=pre[:3, :3], label=f"{label}_lift", count=6, holding=True
        )

    def set_cube_down_on_floor(self, *, cube: str) -> None:
        """Put a held cube down upright on the open floor beside the box."""
        here = self.site_pose()
        offset = here[:3, 3] - self.cube_position(cube=cube)
        down = here.copy()
        down[:3, 3] = np.array([*CUBE_FLOOR_XY, 0.012]) + offset
        above = down.copy()
        above[2, 3] += 0.10
        base = self.reachable_base(
            poses=[above, down],
            holding=True,
            xs=FLOOR_BASE_XS,
            ys=(CUBE_FLOOR_XY[1], CUBE_FLOOR_XY[1] - 0.15, CUBE_FLOOR_XY[1] + 0.15),
        )
        if base is None:
            raise SkillFailed("set_cube_down_on_floor: no base pose reaches the floor spot")
        self.raise_hand(z=0.45, label="cube_raise", holding=True)
        self.move_base(target=base, label="to_floor_spot")
        self.path(
            position=above[:3, 3], rotation=above[:3, :3], label="over_floor", count=6, holding=True
        )
        self.path(position=down[:3, 3], rotation=down[:3, :3], label="lower_to_floor", holding=True)
        self.release_cube(label="floor")
        self.path(position=above[:3, 3], rotation=above[:3, :3], label="back_off_floor", count=3)
        self.mark(label=f"{cube} set down on the floor")

    def shelf_cube_pose(self, *, cube: str, centre: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """Candidate (grasp, mouth) hand poses for a cube at `centre` on shelf index 1,
        leaning toward the cupboard so the wrist stays below the board above."""
        out = []
        for lean in CUBE_LEANS_DEG:
            for face in (0, 1):
                for sign in (1.0, -1.0):
                    grasp = self.cube_grasp(
                        cube=cube, centre=centre, lean_deg=lean, face=face, sign=sign
                    )
                    if grasp[0, 2] <= 0.2:
                        continue
                    mouth = grasp.copy()
                    mouth[:3, 3] = [self.front_x - 0.12, grasp[1, 3], grasp[2, 3] + 0.04]
                    out.append((grasp, mouth))
        return out

    def pick_cube_from_shelf(self, *, cube: str) -> None:
        """Take a cube off shelf index 1 from the front, along a planned route."""
        if self.held_cube is not None or self.in_hand is not None:
            raise SkillFailed("pick_cube_from_shelf: the hand is not empty")
        centre = self.cube_position(cube=cube)
        self.grip = 0.0
        self.raise_hand(z=0.35, label="shelf_cube_raise", holding=False)
        attempts = 0
        for grasp, mouth in self.shelf_cube_pose(cube=cube, centre=centre):
            if attempts == CUBE_PICK_TRIES:
                break
            over = grasp.copy()
            over[2, 3] += 0.03
            found = self.first_route(
                waypoints=[mouth, over, grasp],
                bases=[
                    np.array([x, centre[1] + dy, 0.0])
                    for x in (BASE_X, BASE_X - 0.08, BASE_X - 0.16, BASE_X - 0.24)
                    for dy in (0.0, -0.1, 0.1, -0.2, 0.2)
                ],
                holding=False,
            )
            if found is None:
                continue
            base, route = found
            self.move_base(target=base, label="to_shelf_cube")
            self.follow(route=route, label="into_cube_slot")
            # The pads may rest on the board a millimetre short of the exact pose.
            with contextlib.suppress(SkillFailed):
                self.settle_at(pose=grasp, label="onto_cube")
            self.grip_cube(cube=cube, label="shelf_cube")
            back = self.plan_route(waypoints=[over, mouth], base=base, holding=True, allowance=0.01)
            if back is None:
                self.path(
                    position=over[:3, 3],
                    rotation=over[:3, :3],
                    label="lift_cube",
                    holding=True,
                    count=3,
                )
                self.path(
                    position=mouth[:3, 3], rotation=mouth[:3, :3], label="out_of_slot", holding=True
                )
            else:
                self.follow(route=back, label="out_of_slot", settle=0.05)
            attempts += 1
            try:
                self.check_lifted(
                    cube=cube, above=self.board[CUBE_SHELF] + 0.02, label="pick_cube_from_shelf"
                )
            except SkillFailed as exc:
                self.notes.append(f"pick_cube_from_shelf: attempt {attempts} missed ({exc})")
                continue
            return
        if attempts:
            raise SkillFailed(f"pick_cube_from_shelf: {attempts} grasps on {cube} missed")
        raise SkillFailed(f"pick_cube_from_shelf: no collision-free route to {cube}")

    def place_cube_on_shelf(
        self, *, cube: str, xy: tuple[float, float] = PLACE_ON_SHELF_XY
    ) -> None:
        """Place the held cube upright on shelf index 1 at `xy`."""
        here = self.site_pose()
        offset = here[:3, 3] - self.cube_position(cube=cube)
        drop = here.copy()
        drop[:3, 3] = np.array([*xy, self.board[CUBE_SHELF] + 0.012]) + offset
        over = drop.copy()
        over[2, 3] += 0.03
        mouth = over.copy()
        mouth[:3, 3] = [self.front_x - 0.12, over[1, 3], over[2, 3] + 0.01]
        self.raise_hand(z=0.30, label="cube_raise", holding=True)
        found = None
        for base in [
            np.array([x, y, 0.0])
            for x in (BASE_X, BASE_X - 0.1)
            for y in (0.0, -0.15, 0.15, -0.3, 0.3)
        ]:
            route = self.plan_route(
                waypoints=[mouth, over, drop],
                base=base,
                holding=True,
                start_arm=self.data.qpos[self.qa].copy(),
            )
            if route is not None:
                found = (base, route)
                break
        if found is None:
            raise SkillFailed("place_cube_on_shelf: no collision-free route to the shelf")
        base, route = found
        self.move_base(target=base, label="drive_to_shelf")
        self.follow(route=route, label="cube_into_slot")
        self.settle_at(pose=drop, label="cube_onto_shelf", holding=True)
        self.release_cube(label="shelf")
        back = self.plan_route(waypoints=[over, mouth], base=base, holding=False, allowance=0.01)
        if back is not None:
            self.follow(route=back, label="withdraw", settle=0.05)
        else:
            self.path(position=mouth[:3, 3], rotation=mouth[:3, :3], label="withdraw", count=4)
        placed = self.cube_position(cube=cube)
        self.mark(label=f"{cube} at ({placed[0]:.2f},{placed[1]:.2f},{placed[2]:.3f})")
        if not (
            placed[0] > self.front_x + 0.01
            and abs(placed[2] - self.board[CUBE_SHELF] - 0.01) < 0.004
            and not self.core.cube_in_box(cube)
        ):
            raise SkillFailed(f"{cube} not resting on shelf {CUBE_SHELF}: {np.round(placed, 3)}")

    def place_cube_in_box(self, *, cube: str) -> None:
        """Lower the held cube into the box and let go above its floor."""
        if self.held_cube != cube:
            raise SkillFailed(f"place_cube_in_box: not holding {cube}")
        here = self.site_pose()
        offset = here[:3, 3] - self.cube_position(cube=cube)
        low = here.copy()
        # Let go just above the rim: lower, the leaned wrist meets the box's near wall.
        low[:3, 3] = np.array([*self.box_drop_xy(), BOX_RELEASE_Z]) + offset
        high = low.copy()
        high[2, 3] += 0.12
        self.raise_hand(z=0.45, label="cube_raise", holding=True)
        found = self.first_route(
            waypoints=[high, low],
            bases=[
                np.array([x, y, 0.0])
                for x in (BASE_X, BASE_X - 0.1, BASE_X - 0.2)
                for y in (0.0, -0.15, 0.15, -0.3, 0.3)
            ],
            holding=True,
        )
        if found is None:
            raise SkillFailed("place_cube_in_box: no route reaches over the box")
        base, route = found
        self.move_base(target=base, label="to_box")
        self.follow(route=route, label="over_box", settle=0.04)
        self.release_cube(label="box")
        self.path(position=high[:3, 3], rotation=high[:3, :3], label="out_of_box", count=4)
        self.hold(count=15, label="settle")
        self.mark(label=f"{cube} in box: {self.core.cube_in_box(cube)}")
        if not self.core.cube_in_box(cube):
            raise SkillFailed(f"place_cube_in_box: {cube} not in the box")
