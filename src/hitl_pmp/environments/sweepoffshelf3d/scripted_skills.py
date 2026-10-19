"""Scripted skill controllers for SweepOffShelf3D, driven by the planner's skills.

Vendored from the scripted solvability check (hitl-pmp branch
`claude/sweep-off-shelf3d-solvability` @ 9b635c720, `scripts/sweep_off_shelf3d_skills.py`).
The one change: `ShelfSkills` attaches to an environment it is handed, already reset,
instead of making and resetting its own.

Control only: nothing here renders, captions or writes video. A caller that wants a
video passes a `Recorder`, which is told about every control step and every moment
worth a keyframe; with no recorder the skills run headless. Each of the six planner
skills is one call that runs to completion or raises `SkillFailed`:

- `pick_wiper()`: take the wiper off shelf index 2.
- `sweep_shelf(height=, speed=)`: pull every cube on shelf index 1 over the open front
  edge into the box. `height` is the head's clearance over the board in metres,
  [0, 0.04]; `speed` the pull speed in m/s, [0.05, 0.25].
- `return_wiper()`: put the held wiper back where it started on shelf index 2.
- `pick_cube_from_shelf(cube=)`, `pick_cube_from_floor(cube=)`,
  `place_cube_in_box(cube=)`.

Nothing here edits the environment. The friction grasp is the environment's own; the
only state ever set directly is `drop_cube_into_box`, a documented scene setup.

The wiper is only ever held at one of two fixed grasps, each a constant hand pose in
the wiper's own frame (`ShelfSkills.grasp_in_wiper`): "shelf" on the head, used to take
it off and put it back on its shelf, and "sweep" on the handle at its centre of mass,
used to sweep. Between them it is put down on the floor and taken again. Base poses
and arm routes adapt to where the wiper lies; the grasp never does, and every grasp
taken is recorded with its measured hand-to-wiper pose (`grasps_taken`).

Four facts shape every motion. The cupboard slots are 0.254 m tall, too short for a
vertical gripper (the wrist hits the board above), so every grasp in the cupboard
leans the gripper about its closing axis; a lean keeps the pads on the same two faces.
The 2F-85's friction grasp resists rotation about its closing axis worst of all, so
both wiper grasps sit beside its centre of mass. Lowered behind the cubes, the flat
wiper's handle comes down on any cube in its line, so strokes are placed with no cube
under the handle (`Strokes.plan`). And the box keeps the base at least BASE_X back
from the cupboard while the leaned hand only reaches 0.6 m or more in front of the
base, so the base carries each stroke and the arm trims. Near the cupboard the arm
follows routes planned joint by joint on one IK branch, with the carried wiper in the
collision check.

Nothing here imports KINDER: the environment is built by `backend.py` and handed in.
"""

import contextlib
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
WIPER_SHELF = 2
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
# The two fixed wiper grasps. SHELF: on the head, SHELF_HEAD_OFFSET from its centre
# toward the open front, pads closing along the handle, wrist leaning out of the slot;
# it reaches the wiper wherever it lies on its shelf (a grasp on the handle puts the
# wrist into the cupboard's side panel when the wiper lies toward that side). SWEEP:
# on the handle at the centre of mass (None), wrist over the free end, the side it must
# be on to pull the head toward the robot. Each becomes a pose in the wiper's own frame
# once, at reset, and is never searched over.
SHELF_HEAD_OFFSET = 0.035
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
# Put-down spot on the open floor beside the box, for every regrasp.
FLOOR_HEAD_XY = (0.95, 0.55)
FLOOR_BASE_XS = (0.45, 0.35, 0.55, 0.25)
CUBE_FLOOR_XY = (0.95, 0.45)
SWEEP_STEM = (-1.0, 0.0, 0.0)
CARRY_Z = 0.65
HEAD_AHEADS = (0.80, 0.75, 0.85, 0.70, 0.90, 0.65)
HEAD_AHEAD = 0.80
# Stroke geometry: the head's centre starts behind the deepest cube and stops with the
# head just past the shelf's front edge; reach is the head half-length less a margin,
# limit keeps the head off the cupboard's side walls.
STROKE_START_X = 1.60
STROKE_END_X = 1.355
STROKE_REACH = 0.08
STROKE_LIMIT = 0.17
# Half-width of the strip under the handle (and the fingers on it) that must be free of
# cubes when the wiper is lowered: handle half-width + cube half-diagonal + margin.
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
PULL_CHUNK = 2
# How far below the requested height the pull aims the head, pressing it into the
# board so the cubes it pushes cannot wedge under its front edge and lift it.
PULL_PRESS = 0.005
SWEEP_PASSES = 5
REGRASP_TRIES = 3
CUBE_PICK_TRIES = 2
GRASP_TEST_LIFT = 0.02
PLACE_ON_SHELF_XY = (1.45, 0.0)
BOX_DROP_XY = (1.235, 0.0)
BOX_RELEASE_Z = 0.24
DROP_XY = (1.235, 0.0)


class SkillFailed(RuntimeError):
    """A skill could not reach its postcondition; the message says where and why."""


class WiperLost(RuntimeError):
    """The per-step `HoldingWiper` check failed mid-sweep. Deliberately not a
    `SkillFailed`: tolerant motions (`raise_hand`, `hand_to`, the IK fallbacks) catch
    `SkillFailed` and carry on, which swallowed the loss and let the stroke continue with
    no wiper in hand. `sweep_shelf` turns it into a `SkillFailed`."""


class StepBudgetExhausted(SkillFailed):
    """The run's hard control-step cap was reached; nothing here loops open."""


class SweepSettings(BaseModel):
    """`sweep_shelf`'s two parameters, validated against the documented ranges."""

    model_config = ConfigDict(frozen=True)

    height: float = Field(ge=0.0, le=0.04)
    speed: float = Field(ge=0.05, le=0.25)


class Recorder(Protocol):
    def on_step(self, *, skills: "ShelfSkills", label: str) -> None: ...

    def mark(self, *, skills: "ShelfSkills", label: str) -> None: ...


class Strokes:
    @staticmethod
    def plan(*, ys: list[float], reach: float, band: float, limit: float) -> list[float]:
        """Fewest stroke centres covering every cube with the head (|dy| <= reach)
        while no cube lies under the handle (|dy| > band): lowered behind the cubes,
        the flat wiper's handle would otherwise come down on top of one. Greedy cover
        over centres 5 mm apart within +/- limit."""
        candidates = np.round(np.arange(-limit, limit + 1e-9, 0.005), 4)
        remaining = set(range(len(ys)))
        centres: list[float] = []
        while remaining:
            best: float | None = None
            covered: set[int] = set()
            for c in candidates:
                if any(abs(y - c) <= band for y in ys):
                    continue
                hits = {i for i in remaining if abs(ys[i] - c) <= reach}
                if len(hits) > len(covered):
                    best, covered = float(c), hits
            if best is None:
                break
            centres.append(best)
            remaining -= covered
        return sorted(centres)


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
            shelf: self.core.shelf_top_z(shelf=shelf) for shelf in (CUBE_SHELF, WIPER_SHELF)
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
        self.in_hand: np.ndarray | None = None
        self.grasp_kind: str | None = None
        self.held_cube: str | None = None
        self.slip: float | None = None
        self.slip_deg = 0.0
        self.contact_shortfalls = 0
        # While a sweep runs, `HoldingWiper` is re-evaluated after every control step.
        self.watch_wiper = False
        self.wiper_unheld_steps = 0
        # Hand pose in the wiper's frame for each fixed grasp, and the measured
        # hand-to-wiper pose every time one is taken.
        self.grasp_in_wiper = {
            "shelf": np.linalg.inv(self.wiper_start)
            @ self.leaned_head_grasp(offset=SHELF_HEAD_OFFSET),
            "sweep": np.linalg.inv(self.wiper_start)
            @ self.leaned_handle_grasp(along=SWEEP_ALONG, lean_sign=-1.0),
        }
        self.grasps_taken: list[dict[str, Any]] = []
        self.closed_in_hand = np.eye(4)
        # Head speed actually achieved over each pull (m/s); the base tops out at
        # 0.15 m/s, so a commanded speed above that is reached only with the arm.
        self.pull_speeds: list[float] = []

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
            and self.cube_position(cube=name)[2] < self.board[WIPER_SHELF]
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
            self.in_hand, self.grasp_kind = None, None
            raise WiperLost(f"the wiper left the hand during {label}")

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
        from where it is now."""
        for base in bases:
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

    def wiper_grasp(self, *, kind: str) -> np.ndarray:
        """World hand pose for fixed grasp `kind` on the wiper as it lies now."""
        return self.wiper_pose() @ self.grasp_in_wiper[kind]

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

    def leaned_head_grasp(self, *, offset: float, lean_deg: float = WIPER_LEAN_DEG) -> np.ndarray:
        """Pinch-site pose for a lean grasp on the head, `offset` metres from its centre
        toward the cupboard's open front, pads closing along the handle. The wrist leans
        out of the slot toward the robot, and the grasp sits beside the centre of mass,
        so the head barely droops: on the handle, 13 cm from the centre of mass, it
        drooped 15 deg and fell out while being put back."""
        u = self.stem_direction()
        head = self.data.geom_xpos[self.head_geom].copy()
        along_head = self.data.geom_xmat[self.head_geom].reshape(3, 3)[:, 1].copy()
        front = along_head if along_head[0] < 0 else -along_head
        point = head + offset * front
        rotation = Frames.leaned(closing=u, lean_toward=-front, lean_deg=lean_deg)
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

    def take_wiper(self, *, kind: str, label: str) -> None:
        """Record the grasp just closed, and how far the measured hand-to-wiper pose is
        from the fixed one: the friction grasp is the same grasp every time only if
        this stays small. Measured as the gripper closed, before the test lift lets
        the wiper settle a few millimetres in the pads."""
        self.in_hand = np.linalg.inv(self.site_pose()) @ self.wiper_pose()
        self.grasp_kind = kind
        self.slip, self.slip_deg = 0.0, 0.0
        nominal = np.linalg.inv(self.grasp_in_wiper[kind])
        closed = self.closed_in_hand
        hand_in_wiper = np.linalg.inv(closed)
        turn = Rotation.from_matrix(closed[:3, :3] @ nominal[:3, :3].T).magnitude()
        self.grasps_taken.append({
            "kind": kind,
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
        self.grasp_kind = None
        self.hold(count=20, label=f"{label}_release")

    def pick_wiper(self) -> None:
        """Take the wiper off shelf index 2 by friction and lift it clear of the
        cupboard. On the shelf only a lean toward the handle's free end clears the
        board above, so the grasp is on the handle's middle; the hand comes in level,
        pads 2 cm above the handle, along a route planned joint by joint so the arm
        never changes IK branch (and swings through the cupboard) on the way."""
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
        self.take_wiper(kind="shelf", label="pick_wiper")
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
            options.append(self.wiper_grasp(kind="shelf"))
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

    def set_wiper_down_on_floor(self, *, label: str) -> None:
        """Lay the held wiper flat on the floor beside the box, handle along +y."""
        assert self.in_hand is not None
        half = float(self.model.geom_size[self.head_geom][2])
        head = np.array([*FLOOR_HEAD_XY, half + 0.004])
        stem = self.stem_start
        target = self.wiper_target(head=head, stem=stem) @ np.linalg.inv(self.in_hand)
        above = self.wiper_target(head=head + [0.0, 0.0, 0.30], stem=stem) @ np.linalg.inv(
            self.in_hand
        )
        self.raise_hand(z=CARRY_Z, label=label, holding=True)
        found = None
        for base in [
            np.array([x, head[1] + dy, 0.0]) for x in FLOOR_BASE_XS for dy in (0.0, -0.15, 0.15)
        ]:
            route = self.plan_route(waypoints=[above], base=base, holding=True)
            # The descent from there must be reachable too (planned, not followed).
            if route is not None and self.plan_route(
                waypoints=[target],
                base=base,
                holding=True,
                start_arm=route[-1],
                allowance=CONTACT_PENETRATION,
            ):
                found = (base, route)
                break
        if found is None:
            raise SkillFailed(f"{label}: no route sets the wiper down on the floor")
        base, route = found
        self.move_base(target=base, label=label)
        # To 30 cm above the spot along a planned route, then down in short lines.
        self.follow(route=route, label=label, settle=0.04)
        for z in (0.15, 0.05, 0.015):
            self.move_wiper(head=head + [0.0, 0.0, z], label=label, stem=stem)
        self.move_wiper(head=head, label=label, stem=stem)
        self.let_go_of_wiper(label=label)
        approach = self.site_pose()[:3, 2]
        self.path(
            position=self.data.site_xpos[self.site] - 0.06 * approach + [0.0, 0.0, 0.06],
            rotation=self.site_pose()[:3, :3],
            label=f"{label}_back_off",
            count=3,
        )
        self.mark(label="wiper on the floor")

    def take_wiper_from_floor(self, *, kind: str, label: str) -> None:
        for attempt in range(REGRASP_TRIES):
            grasp = self.wiper_grasp(kind=kind)
            pre = grasp.copy()
            pre[2, 3] += 0.04
            lifted = grasp.copy()
            lifted[2, 3] += 0.25
            self.raise_hand(z=0.30, label=label, holding=False)
            bases = [
                np.array([x, grasp[1, 3] + dy, 0.0])
                for x in FLOOR_BASE_XS
                for dy in (0.0, -0.15, 0.15)
            ]
            # A later attempt starts its search further down the list of base poses.
            found = self.first_route(
                waypoints=[lifted, pre, grasp],
                bases=bases[attempt:] + bases[:attempt],
                holding=False,
            )
            if found is None:
                raise SkillFailed(f"{label}: no route reaches the wiper on the floor")
            base, route = found
            self.move_base(target=base, label=label)
            try:
                self.follow(route=route, label=label)
                self.settle_at(pose=grasp, label=label)
            except SkillFailed as exc:
                self.notes.append(f"{label}: approach {attempt + 1} blocked ({exc}), retrying")
                continue
            if self.close_and_check(label=label):
                self.take_wiper(kind=kind, label=label)
                self.path(
                    position=lifted[:3, 3],
                    rotation=lifted[:3, :3],
                    label=label,
                    count=5,
                    holding=True,
                )
                self.mark(label=f"wiper taken again ({kind} grasp)")
                return
            self.notes.append(f"{label}: grasp {attempt + 1} missed, retrying")
            self.grip = 0.0
            self.hold(count=15, label=label)
            self.path(position=pre[:3, 3], rotation=pre[:3, :3], label=label, count=3)
        raise SkillFailed(f"{label}: {REGRASP_TRIES} grasps on the floor missed")

    def regrasp_for_sweep(self) -> None:
        """The shelf grasp holds the head, the part that does the sweeping. Put the
        wiper down on the open floor and take it again at the sweep grasp."""
        self.notes.append(
            f"shelf grasp held with max slip {self.slip or 0:.3f} m / {self.slip_deg:.1f} deg"
        )
        self.set_wiper_down_on_floor(label="regrasp")
        self.take_wiper_from_floor(kind="sweep", label="regrasp")

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
        for ahead in HEAD_AHEADS:
            base = np.array([min(ahead_of - ahead, BASE_X), target[1, 3], 0.0])
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

    def lower_head(self, *, height: float, label: str) -> None:
        """Bring the head down to `height` over the board, flat, closed loop on the
        measured wiper. (Lowering the hand by pure translation from wherever it is
        compounds a small droop each step: measured 6 deg of tilt and 40 control steps
        per 2 mm.)"""
        x = float(self.data.geom_xpos[self.head_geom][0])
        for _ in range(LOWER_STEPS):
            if abs(self.head_bottom() - self.board[CUBE_SHELF] - height) < LOWER_TOLERANCE:
                return
            self.hand_to(
                target=self.level_target(x=x, above_board=height),
                label=label,
                tolerance=0.003,
                limit=15,
            )

    def pull(self, *, distance: float, settings: SweepSettings, label: str) -> None:
        """Pull the head toward the robot at `settings.speed`, `settings.height` over the
        board, closed loop on the measured wiper (`level_target`) every PULL_CHUNK
        control steps. The base follows the head; it moves at most 0.15 m/s."""
        y = float(self.data.qpos[self.ba][1])
        x = float(self.data.geom_xpos[self.head_geom][0])
        start, began = x, self.ticks
        end = x - distance
        advance = settings.speed * CONTROL_DT * PULL_CHUNK
        while x > end + 1e-9:
            x = max(end, x - advance)
            target = self.level_target(x=x, above_board=settings.height - PULL_PRESS)
            base = np.array([min(x - HEAD_AHEAD, BASE_X), y, 0.0])
            try:
                arm, _ = self.ik(
                    position=target[:3, 3], rotation=target[:3, :3], base=base, holding=True
                )
            except SkillFailed:
                self.contact_shortfalls += 1
                continue
            self.base_target = base
            for _ in range(PULL_CHUNK):
                self.step(arm=arm, base=base, label=label)
        travelled = start - float(self.data.geom_xpos[self.head_geom][0])
        self.pull_speeds.append(travelled / max(1, self.ticks - began) / CONTROL_DT)
        self.hold(count=10, label=label)

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
        self.let_go_of_wiper(label=label)
        up = self.site_pose()
        up[2, 3] += 0.02
        self.hand_to(target=up, label=f"{label}_release")
        for attempt in range(REGRASP_TRIES):
            grasp = self.wiper_grasp(kind="sweep")
            pre = grasp.copy()
            pre[2, 3] += 0.025
            self.hand_to(target=pre, label=f"{label}_regrasp")
            self.hand_to(target=grasp, label=f"{label}_regrasp", tolerance=0.003)
            if self.close_and_check(label=f"{label}_regrasp"):
                self.take_wiper(kind="sweep", label=label)
                self.hand_to(
                    target=self.level_target(x=RESEAT_HEAD_X, above_board=HOVER),
                    label=f"{label}_lift",
                )
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

    def stroke(self, *, y: float, settings: SweepSettings, name: str) -> None:
        flat = self.board[CUBE_SHELF] + float(self.model.geom_size[self.head_geom][2])
        hover = flat + HOVER
        mouth = self.front_x - 0.12
        self.raise_hand(z=CARRY_Z, label=f"{name}_raise", holding=True)
        self.move_base(target=[mouth - HEAD_AHEAD, y, 0.0], label=f"{name}_base")
        self.wiper_to(head=[mouth, y, hover + 0.12], label=f"{name}_above_box")
        self.wiper_to(head=[mouth, y, hover], label=f"{name}_mouth")
        self.recalibrate_grasp()
        for x in np.arange(mouth + 0.05, STROKE_START_X + 1e-9, 0.05):
            self.wiper_to(head=[x, y, hover], label=f"{name}_reach_in")
        self.wiper_to(head=[STROKE_START_X, y, hover], label=f"{name}_reach_in")
        self.recalibrate_grasp()
        self.lower_head(height=settings.height, label=f"{name}_lower")
        self.mark(label=f"{name}: head behind cubes")
        self.pull(distance=STROKE_START_X - STROKE_END_X, settings=settings, label=f"{name}_pull")
        self.mark(label=f"{name}: stroke end")
        if self.slip_deg > RESEAT_TURN_DEG:
            self.reseat_wiper(label=f"{name}_reseat")
        else:
            x = float(self.data.geom_xpos[self.head_geom][0])
            for _ in range(3):
                self.hand_to(target=self.level_target(x=x, above_board=HOVER), label=f"{name}_lift")
            self.recalibrate_grasp()
        self.wiper_to(head=[mouth, y, hover + 0.12], label=f"{name}_out")
        self.hold(count=15, label="settle")
        self.notes.append(f"{name}: {self.counts()['in_box']}/5 in box after the stroke")

    def sweep_shelf(self, *, height: float, speed: float) -> None:
        """Pull every cube on shelf index 1 over the front edge with the wiper head:
        strokes across the shelf's width, up to SWEEP_PASSES passes. Needs the wiper in
        hand (from `pick_wiper`); raises if any cube is left on the shelf."""
        settings = SweepSettings(height=height, speed=speed)
        if self.in_hand is None:
            raise SkillFailed("sweep_shelf: the wiper is not in hand")
        if self.grasp_kind != "sweep":
            self.regrasp_for_sweep()
        self.watch_wiper, self.wiper_unheld_steps = True, 0
        try:
            for sweep_pass in range(SWEEP_PASSES):
                ys = [float(self.cube_position(cube=c)[1]) for c in self.shelf_cubes()]
                if not ys:
                    return
                centres = Strokes.plan(
                    ys=ys, reach=STROKE_REACH, band=HANDLE_BAND, limit=STROKE_LIMIT
                )
                if not centres:
                    break
                for y in centres:
                    self.stroke(y=y, settings=settings, name=f"pass{sweep_pass}_y{y:+.2f}")
        except WiperLost as exc:
            raise SkillFailed(f"sweep_shelf: {exc}") from exc
        finally:
            self.watch_wiper = False
        left = self.shelf_cubes()
        if left:
            raise SkillFailed(f"sweep_shelf: {len(left)} cube(s) still on the shelf: {left}")

    def return_wiper(self) -> None:
        """Put the held wiper back where it started on shelf index 2. Only the shelf
        grasp fits the wiper slot, so a wiper held at the sweep grasp is put down on the
        floor and taken again first."""
        if self.in_hand is None:
            raise SkillFailed("return_wiper: the wiper is not in hand")
        if self.grasp_kind != "shelf":
            self.set_wiper_down_on_floor(label="return")
            self.take_wiper_from_floor(kind="shelf", label="return")
        self.raise_hand(z=CARRY_Z, label="return_raise", holding=True)
        # Aim with the grasp as it is now: 13 cm from the centre of mass the head droops
        # in the friction grasp, and aimed with the grasp at closing it meets the board
        # edge before the slot.
        self.recalibrate_grasp()
        found = self.shelf_wiper_route(target=self.wiper_start)
        if found is None:
            raise SkillFailed("return_wiper: no collision-free route into the wiper slot")
        base, route, grasp = found
        self.move_base(target=base, label="return_to_cupboard")
        self.follow(route=route, label="return_into_slot", settle=0.04)
        self.recalibrate_grasp()
        grasp = self.wiper_start @ np.linalg.inv(self.in_hand)
        self.settle_at(pose=grasp, label="return_onto_shelf", holding=True)
        self.let_go_of_wiper(label="return")
        self.leave_wiper_slot(grasp=grasp, holding=False, label="return_withdraw")
        error = float(np.linalg.norm(self.wiper_pose()[:3, 3] - self.wiper_start[:3, 3]))
        self.mark(label=f"wiper returned, {error:.3f} m from its start")
        if error > 0.05 or abs(self.wiper_lift()) > 0.01:
            raise SkillFailed(f"return_wiper: wiper ended {error:.3f} m from where it started")

    # --- cubes -------------------------------------------------------------------

    def drop_cube_into_box(self, *, cube: str = "cube_0") -> None:
        """Documented scene setup, not a skill: teleport one cube above the box."""
        state = self.core.get_object(cube)
        state.set_pose([*DROP_XY, 0.10], [1.0, 0.0, 0.0, 0.0])
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
        low[:3, 3] = np.array([*BOX_DROP_XY, BOX_RELEASE_Z]) + offset
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
