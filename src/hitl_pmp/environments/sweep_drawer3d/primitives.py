"""The reset's primitives: park the wiper, open/close the drawer, pick, place, push.

Each primitive searches a small set of base stances and grasps, checks every candidate
against the planning scene (IK, collision-free approach, straight-line descent), and
only then moves the robot. A primitive that finds nothing raises ExecutionError without
having touched the world.
"""

from typing import Any, ClassVar, Literal

import numpy as np
from pybullet_helpers.geometry import Pose
from pydantic import BaseModel, ConfigDict, PrivateAttr
from shapely.affinity import translate
from shapely.geometry import Polygon

from .footprints import Footprints
from .motion import ExecutionError, Motion
from .planning_scene import Orientations, PlanningScene
from .session import CubeHeading, SweepDrawerSession
from .types import GripperGeometry, SweepDrawerScene

S = SweepDrawerScene
G = GripperGeometry
HOVER = 0.12
DRAWER_HOVER_Z = 0.53  # above the drawer walls (0.444) with the pads hanging below the EE


class Candidate(BaseModel):
    """A grasp or push: EE pose(s) plus the approach axis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose: Any
    approach: np.ndarray
    yaw: float
    score: float
    end: Any = None
    # gripper opening for this grasp: MuJoCo command and the PyBullet finger state
    cmd: float = G.CUBE_OPEN_CMD
    pb: float = G.CUBE_OPEN_PB
    partner: str | None = None
    kind: Literal["single", "squeeze", "row"] = "single"


class PickPlan(BaseModel):
    """A pick the planning scene has passed end to end, not yet carried out."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grasp: Candidate
    held: tuple[str, ...]
    path: Any
    base: tuple[float, float, float]
    # how far back along the approach axis the hover sits
    back: float
    reach: Any
    down: Any
    rejected: dict[str, int]


class Opening:
    """Gripper command <-> inner pad gap, measured in both models."""

    MJ_CMD = (0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
    MJ_GAP = (0.0851, 0.0699, 0.0615, 0.0572, 0.0529, 0.0485, 0.044, 0.0349)
    PB_STATE = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
    PB_GAP = (0.084, 0.065, 0.054, 0.044, 0.034, 0.022)

    @staticmethod
    def for_gap(*, gap: float) -> tuple[float, float, float]:
        """(MuJoCo command, PyBullet finger state, pad-centre depth below the EE)."""
        cmd = float(np.interp(-gap, [-g for g in Opening.MJ_GAP], Opening.MJ_CMD))
        pb = float(np.interp(-gap, [-g for g in Opening.PB_GAP], Opening.PB_STATE))
        depth = float(np.interp(cmd, [0.0, G.CUBE_OPEN_CMD], [0.0253, G.PAD_CENTER_PARTIAL]))
        return cmd, pb, depth


class SqueezeGrasp:
    """A grasp that closes across a cube turned off the closing axis.

    The pads meet two of the cube's corners and turn it square between them as they
    close. A face grasp takes its closing axis from the cube; this one takes it from the
    surroundings, so a cube turned against a wall can be closed on along the wall, with
    both fingers beside it, where its own face directions would put a finger in the wall.
    """

    # Room between the open pads and the cube's widest extent, both sides together.
    SPARE = 0.006
    # Closing axes tried, evenly over half a turn.
    COUNT = 8
    # A heading this close to a face direction is that face grasp over again.
    SAME = float(np.radians(10.0))
    # Ranked below every face grasp: the cube moves in the hand as the pads close.
    HANDICAP = 0.02

    @staticmethod
    def span(*, cube_yaw: float, closing_yaw: float) -> float:
        """The cube's widest extent along the closing axis."""
        u = np.array([np.cos(closing_yaw), np.sin(closing_yaw)])
        return 2.0 * Footprints.support_distance(yaw=cube_yaw, direction=u)

    @staticmethod
    def gap(*, cube_yaw: float, closing_yaw: float) -> float:
        return SqueezeGrasp.span(cube_yaw=cube_yaw, closing_yaw=closing_yaw) + SqueezeGrasp.SPARE

    @staticmethod
    def headings(*, face_yaws: list[float]) -> list[float]:
        quarter = np.pi / 2
        out = []
        for k in range(SqueezeGrasp.COUNT):
            heading = k * np.pi / SqueezeGrasp.COUNT
            off = [abs((heading - f + quarter / 2) % quarter - quarter / 2) for f in face_yaws]
            if not off or min(off) > SqueezeGrasp.SAME:
                out.append(float(heading))
        return out


class WiperHold:
    """Whether the wiper is in the hand."""

    # Held, the wiper's origin is 4 to 5 cm from the gripper (measured on five seeds).
    REACH = 0.15

    @staticmethod
    def in_hand(*, gripper: np.ndarray, wiper: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(gripper) - np.asarray(wiper))) < WiperHold.REACH


class ShutHand:
    """The hand once it has closed on what it holds.

    The descent is checked with the fingers open to let the cube in. Shut, the pads and
    the links above them stand closer to whatever lies beside the cube; on a leaning
    hand they reach over a touching neighbour's top edge, pinch it, and lift it too.
    """

    # The planning model's hand stood 1 to 2 mm from the neighbour in the one grasp
    # measured to take it, and over 4 mm in grasps beside a neighbour that did not.
    MARGIN: ClassVar[float] = 0.003
    # The planning model's fingers shut no further than this: pads 2.2 cm apart.
    STATE: ClassVar[float] = 0.6

    @staticmethod
    def state(*, held: tuple[str, ...]) -> float:
        """One cube is 2 cm across; a row is wider, and the hand shuts that much less."""
        width = SweepDrawerScene.CUBE_HALF * 2 * len(held) + 0.002
        return min(Opening.for_gap(gap=width)[1], ShutHand.STATE)


class DrawerStroke:
    """How far the base drives to slide the drawer, and whether the drawer got there."""

    # Half the reset's own "drawer closed" criterion (ResetOutcome.drawer_closed, 1 cm).
    CLOSED_TOLERANCE = 0.005
    OPEN_TOLERANCE = 0.03
    # A closing stroke exactly as long as the remaining slide does not shut the drawer:
    # slack in the grip and the arm's compliance absorb part of it (a 1.5 cm stroke moved
    # the drawer 4 mm). Driving past shut lets the joint limit end the stroke instead.
    CLOSE_OVERSHOOT = 0.02

    @staticmethod
    def closing(*, target: float) -> bool:
        return target <= DrawerStroke.CLOSED_TOLERANCE

    @staticmethod
    def base_travel(*, start: float, target: float) -> float:
        travel = target - start
        return (
            travel - DrawerStroke.CLOSE_OVERSHOOT if DrawerStroke.closing(target=target) else travel
        )

    @staticmethod
    def reached(*, end: float, target: float) -> bool:
        """A close that stops short of shut is a failure, not 'within 3 cm of closed', so
        the reset hears about it and can close again."""
        if DrawerStroke.closing(target=target):
            return end < DrawerStroke.CLOSED_TOLERANCE
        return abs(end - target) <= DrawerStroke.OPEN_TOLERANCE


class Primitives(BaseModel):
    """All reset primitives for one session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    scene: PlanningScene
    motion: Motion
    # Both True in use; False restores what was there before, to measure the difference.
    squeeze: bool = True
    finger_board: bool = True

    _last_pick: PickPlan | None = PrivateAttr(default=None)

    @property
    def last_pick(self) -> PickPlan | None:
        """The plan the most recent `pick` carried out."""
        return self._last_pick

    # ================================================================ geometry queries
    def obstacles_2d(
        self, *, cube: str, moved: dict[str, np.ndarray] | None = None
    ) -> list[Polygon]:
        """Footprints at `cube`'s level: the other cubes, plus drawer walls if it is inside."""
        moved = moved or {}
        z = self.session.position(name=cube)[2]
        out = []
        for other in S.CUBES:
            if other == cube:
                continue
            p = moved.get(other, self.session.position(name=other))
            if abs(p[2] - z) < 0.05:
                out.append(Footprints.cube(center=p[:2], yaw=self.session.yaw(name=other)))
        out += self.finger_walls(cube=cube)
        return out

    def walls(self, *, cube: str) -> list[Polygon]:
        """Vertical surfaces at the height of the palm around `cube`: the drawer's walls
        for a cube inside it, the island's drawer faces for a cube on the floor."""
        where = self.session.location(cube=cube)
        if where == "drawer":
            return Footprints.drawer_walls(drawer_pos=self.session.drawer_pos())
        if where == "floor":
            return [Footprints.island_faces(drawer_pos=self.session.drawer_pos())]
        return []

    def finger_walls(self, *, cube: str) -> list[Polygon]:
        """Vertical surfaces at the height of the fingers: the same walls in the drawer,
        but on the floor the island's bottom board, not the drawer faces above it."""
        if self.finger_board and self.session.location(cube=cube) == "floor":
            return [Footprints.island_board()]
        return self.walls(cube=cube)

    def support(self, *, cube: str) -> float:
        return float(self.session.position(name=cube)[2] - S.CUBE_HALF)

    def face_yaws(self, *, cube: str) -> list[float]:
        from kinder_models.dynamic3d.utils import upright_grasp_rotations

        yaws = []
        for q in upright_grasp_rotations(self.session.quaternion(name=cube)):
            yaws.append(float(2 * np.arctan2(q[2], q[3])))
        return yaws

    def grasp_candidates(
        self,
        *,
        cube: str,
        at: np.ndarray | None = None,
        obstacles: list[Polygon] | None = None,
    ) -> list[Candidate]:
        """Face grasps, then squeeze grasps: closing axes x tilts that lean the palm away
        from walls x small offsets, best first. Kept only if the fingers clear everything
        in 2D and, beside a wall, the palm clears it."""
        c = self.session.position(name=cube) if at is None else at
        obst = self.obstacles_2d(cube=cube) if obstacles is None else obstacles
        faces = self.face_yaws(cube=cube)
        out: list[Candidate] = []
        for yaw in faces:
            out += self._grasps_along(cube=cube, centre=c, yaw=yaw, obstacles=obst)
        turned = self.session.yaw(name=cube)
        for yaw in SqueezeGrasp.headings(face_yaws=faces) if self.squeeze else ():
            out += self._grasps_along(
                cube=cube,
                centre=c,
                yaw=yaw,
                obstacles=obst,
                gap=SqueezeGrasp.gap(cube_yaw=turned, closing_yaw=yaw),
                handicap=SqueezeGrasp.HANDICAP,
                kind="squeeze",
            )
        out.sort(key=lambda k: -k.score)
        return out

    def _grasps_along(
        self,
        *,
        cube: str,
        centre: np.ndarray,
        yaw: float,
        obstacles: list[Polygon],
        gap: float = 0.032,
        handicap: float = 0.0,
        kind: Literal["single", "squeeze", "row"] = "single",
    ) -> list[Candidate]:
        """Grasps closing along `yaw` with the pads `gap` apart on the way down."""
        from pybullet_helpers.geometry import Pose

        cmd, pb, depth = Opening.for_gap(gap=gap)
        support = self.support(cube=cube)
        walls = self.walls(cube=cube)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-np.sin(yaw), np.cos(yaw)])
        tries = [
            (alpha, lean, off_v, off_u)
            for alpha in (0.0, 0.45, 0.7)
            for lean in ((0.0,) if alpha == 0 else (1.0, -1.0))
            for off_v in (0.0, 0.003, -0.003, 0.005, -0.005)
            for off_u in (0.0, 0.002, -0.002)
        ]
        centres = np.array([centre[:2] + off_v * v + off_u * u for _, _, off_v, off_u in tries])
        # a tilted finger reaches toward the side the palm leans away from
        reach = np.array([0.019 * np.sin(alpha) for alpha, _, _, _ in tries])
        lean = np.array([lean for _, lean, _, _ in tries])
        clear = Footprints.finger_clearances(
            centres=centres - 0.5 * (reach * lean)[:, None] * v,
            yaw=yaw,
            obstacles=obstacles,
            gap=gap,
            widen=reach,
            inner=SqueezeGrasp.span(cube_yaw=self.session.yaw(name=cube), closing_yaw=yaw) / 2,
        )
        out: list[Candidate] = []
        for k, (alpha, side, off_v, off_u) in enumerate(tries):
            if clear[k] < 0.002:
                continue
            ctr = centres[k]
            if walls:
                palm = Footprints.palm(center=ctr, yaw=yaw, lean=side, alpha=alpha)
                if Footprints.clearance(shape=palm, obstacles=walls) < 0.003:
                    continue
            q, a = Orientations.tilted(yaw=yaw, lean=side, alpha=alpha)
            pad = np.array([ctr[0], ctr[1], support + 0.0124 + 0.012 * np.sin(alpha)])
            score = min(float(clear[k]), 0.01) - 0.3 * (abs(off_v) + abs(off_u)) - 0.004 * alpha
            out.append(
                Candidate(
                    pose=Pose(tuple(pad - a * depth), q),
                    approach=a,
                    yaw=yaw,
                    score=score - handicap,
                    cmd=cmd,
                    pb=pb,
                    kind=kind,
                )
            )
        return out

    def row_between(self, *, cube: str, partner: str) -> tuple[str, ...] | None:
        """Cubes lying in a face-to-face row between two end cubes (None if the row is
        broken: a gap too wide for friction to bridge, or a cube half on the line)."""
        a, b = self.session.position(name=cube)[:2], self.session.position(name=partner)[:2]
        axis = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        members = [(0.0, cube), (float(np.dot(b - a, axis)), partner)]
        for o in S.CUBES:
            if o in (cube, partner) or self.session.location(cube=o) != self.session.location(
                cube=cube
            ):
                continue
            p = self.session.position(name=o)[:2] - a
            along, off = float(np.dot(p, axis)), float(abs(np.cross(axis, p)))
            if 0.0 < along < members[1][0] and off < 0.02:
                if off > 0.006:
                    return None
                members.append((along, o))
        members.sort()
        if any(n[0] - m[0] > 0.034 for m, n in zip(members, members[1:], strict=False)):
            return None
        return tuple(o for _, o in members[1:-1])

    def pair_candidates(self, *, cube: str, partner: str) -> list[Candidate]:
        """Grasp a face-to-face row (two or three cubes) at once: the pads close on its
        outer faces. For cubes packed too tightly for a finger between them, this is the
        only grasp there is."""
        from pybullet_helpers.geometry import Pose

        middle = self.row_between(cube=cube, partner=partner)
        if middle is None:
            return []
        group = (cube, *middle, partner)
        a, b = self.session.position(name=cube), self.session.position(name=partner)
        d = b[:2] - a[:2]
        sep = float(np.linalg.norm(d))
        if abs(a[2] - b[2]) > 0.005:
            return []
        yaw = float(np.arctan2(d[1], d[0]))
        for o in group:
            off = (self.session.yaw(name=o) - yaw) % (np.pi / 2)
            if min(off, np.pi / 2 - off) > np.radians(12):
                return []
        gap = sep + 2 * S.CUBE_HALF + 0.011
        if gap > 0.0845:
            return []
        cmd, pb, depth = Opening.for_gap(gap=gap)
        mid = (a[:2] + b[:2]) / 2
        obst = [
            p
            for o in S.CUBES
            if o not in group and abs(self.session.position(name=o)[2] - a[2]) < 0.05
            for p in [
                Footprints.cube(
                    center=self.session.position(name=o)[:2], yaw=self.session.yaw(name=o)
                )
            ]
        ] + self.finger_walls(cube=cube)
        walls = self.walls(cube=cube)
        v = np.array([-np.sin(yaw), np.cos(yaw)])
        support = self.support(cube=cube)
        out = []
        for alpha in (0.0, 0.45, 0.7):
            for lean in (0.0,) if alpha == 0 else (1.0, -1.0):
                reach = 0.019 * np.sin(alpha)
                clr = Footprints.finger_clearance(
                    center=mid - 0.5 * reach * lean * v,
                    yaw=yaw,
                    obstacles=obst,
                    gap=gap,
                    widen=reach,
                )
                if clr < 0.002:
                    continue
                palm = Footprints.palm(center=mid, yaw=yaw, lean=lean, alpha=alpha)
                if walls and Footprints.clearance(shape=palm, obstacles=walls) < 0.003:
                    continue
                q, ax = Orientations.tilted(yaw=yaw, lean=lean, alpha=alpha)
                pad = np.array([mid[0], mid[1], support + 0.0124 + 0.012 * np.sin(alpha)])
                pose = Pose(tuple(pad - ax * depth), q)
                out.append(
                    Candidate(
                        pose=pose,
                        approach=ax,
                        yaw=yaw,
                        score=min(clr, 0.01) - 0.004 * alpha,
                        cmd=cmd,
                        pb=pb,
                        partner=partner,
                        kind="row",
                    )
                )
        out.sort(key=lambda k: -k.score)
        return out

    def distinct(self, *, candidates: list[Candidate], limit: int = 8) -> list[Candidate]:
        keys, kept = set(), []
        for k in candidates:
            key = (
                round(float(np.mod(k.yaw, np.pi)), 2),
                round(float(k.approach[0]), 2),
                round(float(k.approach[1]), 2),
            )
            if key not in keys:
                keys.add(key)
                kept.append(k)
        return kept[:limit]

    def stances(self, *, target: np.ndarray, where: str) -> list[tuple[float, float, float]]:
        """Base poses to try: facing the island from beyond the (open) drawer, or around a
        floor cube at arm's length."""
        out = []
        if where in ("drawer", "counter", "other"):
            xmin = (
                S.DRAWER_HANDLE_FRONT_X
                + max(self.session.drawer_pos(), 0.0)
                + S.CHASSIS_HALF[0]
                + 0.03
            )
            for x in (xmin, xmin + 0.05, xmin + 0.1, xmin + 0.15):
                for y in (target[1], target[1] + 0.15, target[1] - 0.15, 0.0):
                    for th in (np.pi, np.pi - 0.3, np.pi + 0.3):
                        out.append((float(x), float(y), float((th + np.pi) % (2 * np.pi) - np.pi)))
        else:
            for dist in (0.55, 0.5, 0.6, 0.65):
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    out.append((
                        float(target[0] - dist * np.cos(ang)),
                        float(target[1] - dist * np.sin(ang)),
                        float((ang + np.pi) % (2 * np.pi) - np.pi),
                    ))
        return out

    def graspable(self, *, cube: str) -> bool:
        return bool(self.grasp_candidates(cube=cube))

    def resolve_at_actual_base(
        self,
        *,
        hover: Any,
        target: Any,
        bodies: set[int],
        descent_bodies: set[int],
        finger_state: float = 0.0,
        margin: float = 0.0,
        held: int | None = None,
        held_tf: Any = None,
        step: float = 0.01,
        ik_seed: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """Re-solve hover IK, the arm plan and the straight descent at the base pose the
        robot actually reached (the base stops within millimetres, not exactly)."""
        self.scene.sync()
        q_h = self.scene.ik(pose=hover, seed=S.HOME if ik_seed is None else ik_seed)
        if q_h is None:
            return None
        plan = self.scene.plan_arm(goal=q_h, bodies=bodies, held=held, held_tf=held_tf)
        if plan is None:
            return None
        down = self.scene.linear_path(
            start=q_h,
            target=target,
            step=step,
            bodies=descent_bodies,
            finger_state=finger_state,
            margin=margin,
        )
        return None if down is None else (plan, down)

    def pad_centers(self) -> list[np.ndarray]:
        import mujoco

        m, d = self.session.mj_model, self.session.mj_data
        out = []
        for n in ("robot_left_pad1", "robot_right_pad1"):
            out.append(
                np.array(
                    d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)], dtype=float
                )
            )
        return out

    # ================================================================ wiper
    def choose_wiper_parking_pose(self) -> str:
        """Keep a valid initial pose; otherwise plan inside the declared start region."""
        import pybullet
        from pybullet_helpers.geometry import Pose, set_pose
        from scipy.spatial.transform import Rotation

        initial, quat = self.session.initial_pose(name=S.WIPER)
        if S.COUNTER_TOP - 0.01 < initial[2] < S.COUNTER_TOP + 0.02:
            return "valid initial wiper pose retained"
        fixture, region = self.session.wiper_initial_region()
        self.scene.sync()
        obstacles = self.scene.bodies()
        rng = np.random.default_rng(0)
        try:
            for _ in range(256):
                x, y, z, yaw = fixture.sample_pose_in_region(region, rng)
                # Initialization samples a volume, then gravity settles the object;
                # physical placement targets the region's supporting surface.
                z = min(float(r.bbox[2]) for r in fixture.region_objects[region])
                xyzw = Rotation.from_euler("z", yaw).as_quat()
                q = (float(xyzw[0]), float(xyzw[1]), float(xyzw[2]), float(xyzw[3]))
                pos = np.array([x, y, z + 0.003])
                set_pose(self.scene.wiper_body, Pose(tuple(pos), q), self.scene.cid)
                low, high = pybullet.getAABB(self.scene.wiper_body, physicsClientId=self.scene.cid)
                corners = [
                    np.array([a, b, z], dtype=np.float32)
                    for a in (low[0], high[0])
                    for b in (low[1], high[1])
                ]
                if not all(fixture.check_in_region(c, region) for c in corners):
                    continue
                if any(
                    pybullet.getClosestPoints(
                        self.scene.wiper_body, other, distance=0.001, physicsClientId=self.scene.cid
                    )
                    for other in obstacles
                ):
                    continue
                self.session.set_wiper_parking_pose(position=np.array([x, y, z]), quaternion=q)
                coords = [round(v, 4) for v in (x, y, z)]
                return f"planned wiper target inside {region}: {coords}"
        finally:
            set_pose(
                self.scene.wiper_body,
                Pose(
                    tuple(self.session.position(name=S.WIPER)),
                    self.session.quaternion(name=S.WIPER),
                ),
                self.scene.cid,
            )
        raise ExecutionError(f"no collision-free wiper parking pose inside {region}")

    def wiper_handle_geometry(self) -> tuple[int, int]:
        import mujoco

        m = self.session.mj_model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, S.WIPER)
        handle = max(
            (g for g in range(m.ngeom) if m.geom_bodyid[g] == body),
            key=lambda g: float(m.geom_size[g][0] / m.geom_size[g][1]),
        )
        return handle, 0

    def wiper_in_hand(self, *, gripper: np.ndarray, wiper: np.ndarray) -> bool:
        return WiperHold.in_hand(gripper=gripper, wiper=wiper)

    def wiper_grasp_point(
        self, *, center: np.ndarray, axis: np.ndarray, along: float
    ) -> np.ndarray:
        return center + along * axis

    def wiper_grasp_standoff(self) -> float:
        """Distance behind the grasp point along negative end-effector approach."""
        return 0.035

    def wiper_grasp_offsets(self) -> tuple[float, ...]:
        return 0.0, 0.03, 0.06

    def wiper_approach_angles(self) -> tuple[float, ...]:
        return 0.0, 0.4, 0.7, 1.0, 1.57, 1.9

    def wiper_stow_goal(self) -> np.ndarray:
        return np.asarray(S.HOME)

    def wiper_pickup_carry_goal(self) -> np.ndarray:
        """Keep shared pickup completion at its existing stow posture."""
        return self.wiper_stow_goal()

    def wiper_grasp_yaw(self, *, axis: np.ndarray) -> float:
        return float(np.arctan2(axis[1], axis[0]) + np.pi / 2)

    def wiper_pick_targets_after_navigation(
        self, *, hover: Pose, target: Pose, along: float, approach: np.ndarray
    ) -> tuple[Pose, Pose]:
        """Shared default preserves the already checked native pickup targets."""
        del along, approach
        return hover, target

    def recover_wiper(self) -> str:
        """Grasp the observed handle and verify that it follows a physical lift."""
        from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        data = self.session.mj_data
        # The long, narrow handle is distinguishable from the wide blade by local x.
        handle, handle_axis = self.wiper_handle_geometry()
        center = np.array(data.geom_xpos[handle])
        handle_axes = np.array(data.geom_xmat[handle]).reshape(3, 3)
        axis = handle_axes[:, handle_axis]
        yaw = self.wiper_grasp_yaw(axis=axis)
        before = self.session.position(name=S.WIPER).copy()
        wiper = Pose(tuple(before), self.session.quaternion(name=S.WIPER))
        set_pose(self.scene.wiper_body, wiper, self.scene.cid)
        rejected: dict[str, int] = {}
        orientations = []
        for azimuth in np.linspace(yaw, yaw + 2 * np.pi, 8, endpoint=False):
            for angle in self.wiper_approach_angles():
                closing = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
                toward = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
                approach = np.sin(angle) * toward + np.array([0.0, 0.0, -np.cos(angle)])
                orientations.append((closing, approach))
        for along in self.wiper_grasp_offsets():
            grasp_point = self.wiper_grasp_point(center=center, axis=axis, along=along)
            for closing, approach in orientations:
                goal = Pose(
                    tuple(grasp_point - approach * self.wiper_grasp_standoff()),
                    Orientations.from_axes(closing=closing, approach=approach),
                )
                hover = Pose(tuple(np.asarray(goal.position) - approach * 0.08), goal.orientation)
                for stance in self.stances(target=grasp_point, where="floor"):
                    path = self.scene.plan_base(target=stance)
                    if path is None:
                        rejected["base"] = rejected.get("base", 0) + 1
                        continue
                    base = path[-1]
                    self.scene.sync(base=base)
                    bodies = self.scene.bodies()
                    self.scene.robot.set_joints(self.scene.planning_fingers(arm=S.HOME))
                    solutions = ikfast_closest_inverse_kinematics(
                        self.scene.robot,
                        world_from_target=hover,
                    )
                    q_h, down, reach = None, None, None
                    for solution in solutions[:4]:
                        q_h = np.asarray(solution[:7], dtype=float)
                        if self.scene.in_collision(
                            joints=self.scene.planning_fingers(arm=q_h),
                            bodies=bodies,
                        ):
                            continue
                        down = self.scene.linear_path(
                            start=q_h,
                            target=goal,
                            bodies=bodies,
                            max_jump=0.6,
                        )
                        if down is None:
                            continue
                        reach = self.scene.plan_arm(
                            goal=q_h,
                            bodies=bodies | {self.scene.wiper_body},
                            start=S.HOME,
                            base=base,
                        )
                        if reach is not None:
                            break
                    if reach is None:
                        why = "hover IK" if q_h is None else "descent" if down is None else "reach"
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    self.motion.go_home(grip=0.0)
                    self.motion.set_gripper(command=0.0)
                    if not self.motion.drive(path=path, grip=0.0):
                        raise ExecutionError("wiper pickup base did not converge")
                    hover, goal = self.wiper_pick_targets_after_navigation(
                        hover=hover, target=goal, along=along, approach=approach
                    )
                    redo = self.resolve_at_actual_base(
                        hover=hover,
                        target=goal,
                        bodies=bodies | {self.scene.wiper_body},
                        descent_bodies=bodies,
                        ik_seed=q_h,
                    )
                    if redo is None:
                        raise ExecutionError("wiper pickup unavailable at actual base pose")
                    reach, down = redo
                    if not self.motion.follow(path=reach, grip=0.0):
                        raise ExecutionError("wiper pickup approach did not converge")
                    if not self.motion.follow(path=down, grip=0.0, final_tol=0.006):
                        raise ExecutionError("wiper pickup descent did not converge")
                    self.motion.set_gripper(command=1.0)
                    ee = self.scene.ee_now()
                    grasped = Pose(
                        tuple(self.session.position(name=S.WIPER)),
                        self.session.quaternion(name=S.WIPER),
                    )
                    held_tf = multiply_poses(ee.invert(), grasped)
                    up = None
                    # A blade beneath a cabinet overhang cannot rise vertically.
                    # Retreat along the observed approach before lifting, checking
                    # both the arm and the carried blade at every waypoint.
                    for retreat in (0.0, 0.05, 0.10, 0.16, 0.22):
                        back_pos = np.asarray(ee.position) - retreat * approach
                        targets = [Pose(tuple(back_pos), ee.orientation)] if retreat else []
                        targets.append(Pose(tuple(back_pos + [0.0, 0.0, 0.16]), ee.orientation))
                        candidate: list[np.ndarray] = []
                        start = self.session.arm()
                        for target in targets:
                            segment = self.scene.linear_path(
                                start=start,
                                target=target,
                                bodies=bodies,
                                finger_state=G.CLOSED_PB,
                                max_jump=0.6,
                            )
                            if segment is None or any(
                                self.scene.in_collision(
                                    joints=self.scene.planning_fingers(arm=q, state=G.CLOSED_PB),
                                    bodies=bodies,
                                    held=self.scene.wiper_body,
                                    held_tf=held_tf,
                                )
                                for q in segment
                            ):
                                candidate = []
                                break
                            candidate.extend(segment)
                            start = segment[-1]
                        if candidate:
                            up = candidate
                            break
                    if up is None:
                        raise ExecutionError("no collision-free lift for the grasped wiper")
                    if not self.motion.follow(path=up, grip=1.0):
                        raise ExecutionError("wiper pickup lift did not converge")
                    after = self.session.position(name=S.WIPER)
                    ee = np.asarray(self.scene.ee_now().position)
                    if after[2] < before[2] + 0.05 or not self.wiper_in_hand(
                        gripper=ee, wiper=after
                    ):
                        raise ExecutionError(
                            f"wiper did not follow the lift: rise {after[2] - before[2]:.3f} m;"
                            f" gripper distance {np.linalg.norm(ee - after):.3f} m"
                        )
                    # Base navigation assumes the arm is stowed. A floor grasp
                    # leaves it extended below the counter, so stow while checking
                    # the actual carried blade before any navigation begins.
                    held_now = multiply_poses(
                        self.scene.ee_now().invert(),
                        Pose(tuple(after), self.session.quaternion(name=S.WIPER)),
                    )
                    stow = self.scene.plan_arm(
                        goal=self.wiper_pickup_carry_goal(),
                        bodies=self.scene.bodies(),
                        held=self.scene.wiper_body,
                        held_tf=held_now,
                    )
                    if stow is None:
                        raise ExecutionError("no collision-free stow for the recovered wiper")
                    if not self.motion.follow(path=stow, grip=1.0):
                        raise ExecutionError("recovered wiper stow did not converge")
                    if not self.wiper_in_hand(
                        gripper=np.asarray(self.scene.ee_now().position),
                        wiper=self.session.position(name=S.WIPER),
                    ):
                        raise ExecutionError("wiper lost during stow")
                    return f"observed wiper rise {after[2] - before[2]:.3f} m; rejected {rejected}"
        raise ExecutionError(f"no collision-free wiper grasp; rejected {rejected}")

    def park_wiper(self) -> str:
        """Physically place the held tool in its declared start region and verify it."""
        from pybullet_helpers.geometry import Pose, multiply_poses
        from scipy.spatial.transform import Rotation

        home_pos, home_q = self.session.wiper_parking_pose()
        ee = self.scene.ee_now()
        lying = self.session.position(name=S.WIPER)
        if not WiperHold.in_hand(gripper=np.asarray(ee.position), wiper=lying):
            raise ExecutionError("the wiper is not in the hand")
        held_tf = multiply_poses(
            ee.invert(), Pose(tuple(lying), self.session.quaternion(name=S.WIPER))
        )
        # Navigation's footprint assumes a stowed arm, including the carried blade.
        if np.max(np.abs(np.asarray(S.HOME) - self.session.arm())) > 0.04:
            stow = self.scene.plan_arm(
                goal=S.HOME,
                bodies=self.scene.bodies(),
                held=self.scene.wiper_body,
                held_tf=held_tf,
            )
            if stow is None:
                # Sweep can end with the blade contacting the counter or drawer.
                # Leave that contact with a checked lift before folding the arm.
                for rise in (0.08, 0.05, 0.03, 0.015, 0.12, 0.16):
                    ee = self.scene.ee_now()
                    bodies = self.scene.bodies()
                    lift = self.scene.linear_path(
                        start=self.session.arm(),
                        target=Pose(
                            tuple(np.asarray(ee.position) + [0.0, 0.0, rise]), ee.orientation
                        ),
                        bodies=bodies,
                        finger_state=G.CLOSED_PB,
                        max_jump=0.6,
                    )
                    if lift is None or any(
                        self.scene.in_collision(
                            joints=self.scene.planning_fingers(arm=q, state=G.CLOSED_PB),
                            bodies=bodies,
                            held=self.scene.wiper_body,
                            held_tf=held_tf,
                        )
                        for q in lift
                    ):
                        continue
                    if not self.motion.follow(path=lift, grip=1.0):
                        raise ExecutionError("held wiper lift before stow did not converge")
                    held_tf = multiply_poses(
                        self.scene.ee_now().invert(),
                        Pose(
                            tuple(self.session.position(name=S.WIPER)),
                            self.session.quaternion(name=S.WIPER),
                        ),
                    )
                    stow = self.scene.plan_arm(
                        goal=S.HOME,
                        bodies=self.scene.bodies(),
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    if stow is not None:
                        break
            if stow is None or not self.motion.follow(path=stow, grip=1.0):
                raise ExecutionError("held wiper could not be stowed for return")
            held_tf = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name=S.WIPER)),
                    self.session.quaternion(name=S.WIPER),
                ),
            )
        goal = multiply_poses(Pose(tuple(home_pos + [0.0, 0.0, 0.013]), home_q), held_tf.invert())
        pre = Pose(tuple(np.asarray(goal.position) + [0.0, 0.0, 0.10]), goal.orientation)
        rejected: dict[str, int] = {}
        for distance in (0.7, 0.6, 0.8, 0.65, 0.75):
            for dy in (0.0, 0.1, -0.1):
                path = self.scene.plan_base(
                    target=(float(home_pos[0] + distance), float(home_pos[1] + dy), np.pi)
                )
                if path is None:
                    rejected["base"] = rejected.get("base", 0) + 1
                    continue
                base = path[-1]
                self.scene.sync(base=base)
                bodies = self.scene.bodies()
                q_pre = self.scene.ik(pose=pre, seed=S.HOME)
                if q_pre is None:
                    rejected["IK"] = rejected.get("IK", 0) + 1
                    continue
                down = self.scene.linear_path(
                    start=q_pre,
                    target=goal,
                    bodies=bodies,
                    finger_state=G.CLOSED_PB,
                    max_jump=0.6,
                )
                if down is None or any(
                    self.scene.in_collision(
                        joints=self.scene.planning_fingers(arm=q, state=G.CLOSED_PB),
                        bodies=bodies,
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    for q in [q_pre, *down]
                ):
                    rejected["descent"] = rejected.get("descent", 0) + 1
                    continue
                carry = self.scene.plan_arm(
                    goal=q_pre,
                    bodies=bodies,
                    base=base,
                    held=self.scene.wiper_body,
                    held_tf=held_tf,
                )
                if carry is None:
                    rejected["carry"] = rejected.get("carry", 0) + 1
                    continue
                if not self.motion.drive(path=path, grip=1.0):
                    raise ExecutionError("wiper return base did not converge")
                if not WiperHold.in_hand(
                    gripper=np.asarray(self.scene.ee_now().position),
                    wiper=self.session.position(name=S.WIPER),
                ):
                    raise ExecutionError("wiper lost during return navigation")
                redo = self.resolve_at_actual_base(
                    hover=pre,
                    target=goal,
                    bodies=bodies,
                    descent_bodies=bodies,
                    held=self.scene.wiper_body,
                    held_tf=held_tf,
                    finger_state=G.CLOSED_PB,
                    ik_seed=q_pre,
                )
                if redo is None:
                    rejected["actual base"] = rejected.get("actual base", 0) + 1
                    continue
                carry, down = redo
                if any(
                    self.scene.in_collision(
                        joints=self.scene.planning_fingers(arm=q, state=G.CLOSED_PB),
                        bodies=bodies,
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    for q in down
                ):
                    rejected["actual descent"] = rejected.get("actual descent", 0) + 1
                    continue
                if not self.motion.follow(path=carry, grip=1.0):
                    raise ExecutionError("wiper carry did not converge")
                if not self.motion.follow(path=down, grip=1.0, final_tol=0.005):
                    raise ExecutionError("wiper placement did not converge")
                self.motion.set_gripper(command=0.0)
                self.motion.hold(ticks=40, grip=0.0)
                ee = self.scene.ee_now()
                approach = Rotation.from_quat(ee.orientation).as_matrix()[:, 2]
                away = np.array([base[0] - home_pos[0], base[1] - home_pos[1], 0.0])
                away *= 0.10 / np.linalg.norm(away)
                withdrawal = None
                for offset in (
                    -approach * 0.10,
                    away,
                    away + [0.0, 0.0, -0.05],
                    away + [0.0, 0.0, 0.05],
                    np.array([0.0, 0.0, 0.10]),
                ):
                    self.scene.sync()
                    withdrawal = self.scene.linear_path(
                        start=self.session.arm(),
                        target=Pose(tuple(np.asarray(ee.position) + offset), ee.orientation),
                        bodies=self.scene.bodies() | {self.scene.wiper_body},
                        max_jump=0.6,
                    )
                    if withdrawal is not None:
                        break
                if withdrawal is None or not self.motion.follow(path=withdrawal, grip=0.0):
                    raise ExecutionError("no converged collision-free withdrawal from the wiper")
                self.motion.hold(ticks=40, grip=0.0)
                pos = self.session.position(name=S.WIPER)
                yaw = Rotation.from_quat(self.session.quaternion(name=S.WIPER)).as_euler("zyx")[0]
                target_yaw = Rotation.from_quat(home_q).as_euler("zyx")[0]
                yaw_error = abs((yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi)
                fixture, region = self.session.wiper_initial_region()
                in_region = any(
                    fixture.check_in_region(
                        np.asarray(pos + [0.0, 0.0, dz], dtype=np.float32), region
                    )
                    for dz in (0.0, 0.01, -0.01)
                )
                if not (
                    S.COUNTER_TOP - 0.01 < pos[2] < S.COUNTER_TOP + 0.02
                    and np.linalg.norm(pos[:2] - home_pos[:2]) < 0.02
                    and yaw_error < np.radians(5.0)
                    and in_region
                ):
                    raise ExecutionError(
                        "released wiper did not settle at the declared start target"
                    )
                stow = self.scene.plan_arm(
                    goal=S.HOME,
                    bodies=self.scene.bodies() | {self.scene.wiper_body},
                )
                if stow is None or not self.motion.follow(path=stow, grip=0.0):
                    raise ExecutionError("empty-hand stow after placement did not converge")
                return f"observed supported placement and withdrawal; rejected {rejected}"
        raise ExecutionError(f"no physical placement plan for the wiper; rejected {rejected}")

    # ================================================================ drawer
    def handle_position(self) -> np.ndarray:
        import mujoco

        m, d = self.session.mj_model, self.session.mj_data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, S.DRAWER + "_handle")
        bar = max(
            (g for g in range(m.ngeom) if m.geom_bodyid[g] == b),
            key=lambda g: m.geom_size[g][0] + m.geom_size[g][1],
        )
        return np.array(d.geom_xpos[bar], dtype=float)

    def grasp_handle(self) -> float:
        """Drive in front of the drawer and close the open gripper on the handle bar, the
        approach pitched down so the fingers close vertically around it."""
        from pybullet_helpers.geometry import Pose

        rejected: dict[str, int] = {}
        for pitch in (0.35, 0.0, 0.6):
            for dx in (0.62, 0.66, 0.70, 0.58, 0.74):
                h = self.handle_position()
                a = np.array([-np.cos(pitch), 0.0, -np.sin(pitch)])
                closing = np.array([-np.sin(pitch), 0.0, np.cos(pitch)])
                q = Orientations.from_axes(closing=closing, approach=a)
                ee = Pose(tuple(h - a * G.PAD_CENTER_OPEN), q)
                pre = Pose(tuple(h - a * (G.PAD_CENTER_OPEN + 0.09)), q)
                path = self.scene.plan_base(target=(float(h[0] + dx), float(h[1]), np.pi))
                if path is None:
                    rejected["base"] = rejected.get("base", 0) + 1
                    continue
                base = path[-1]
                self.scene.sync(base=base)
                everything = self.scene.bodies()
                q_pre = self.scene.ik(pose=pre, seed=S.HOME)
                if q_pre is None or self.scene.in_collision(
                    joints=self.scene.planning_fingers(arm=q_pre), bodies=everything
                ):
                    rejected["pre"] = rejected.get("pre", 0) + 1
                    continue
                approach = self.scene.linear_path(
                    start=q_pre, target=ee, bodies=self.scene.bodies(without_drawer=True)
                )
                plan = (
                    None
                    if approach is None
                    else self.scene.plan_arm(goal=q_pre, bodies=everything, start=S.HOME, base=base)
                )
                if plan is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                self.motion.go_home(grip=0.0)
                self.motion.set_gripper(command=0.0)
                if not self.motion.drive(path=path, grip=0.0):
                    raise ExecutionError("base did not reach the drawer stance")
                redo = self.resolve_at_actual_base(
                    hover=pre,
                    target=ee,
                    bodies=self.scene.bodies(),
                    descent_bodies=self.scene.bodies(without_drawer=True),
                )
                if redo is not None:
                    plan, approach = redo
                self.motion.follow(path=plan, grip=0.0)
                self.motion.follow(path=approach, grip=0.0, final_tol=0.008)
                g = self.motion.set_gripper(command=1.0)
                if g < 0.2:
                    raise ExecutionError(f"handle grasp blocked (gripper {g:.2f})")
                return pitch
        raise ExecutionError(f"no handle grasp; rejected {rejected}")

    def move_drawer(self, *, target: float, speed: float = 0.012) -> str:
        """Grasp the handle and slide the drawer to `target` by driving the base with the
        arm held rigid -- the base, not the arm, supplies the stroke."""
        start = self.session.drawer_pos()
        pitch = self.grasp_handle()
        x, y, th = self.session.base()
        hold = self.session.arm()
        travel = DrawerStroke.base_travel(start=start, target=target)
        ticks = int(abs(travel) / speed) + 60
        self.motion.drive_straight(
            target=(x + travel, y, th), grip=1.0, arm=hold, speed=speed, max_ticks=ticks
        )
        self.motion.hold(ticks=5, grip=1.0, arm=hold)
        self.release_handle(hold=hold)
        end = self.session.drawer_pos()
        if not DrawerStroke.reached(end=end, target=target):
            raise ExecutionError(f"drawer at {end:.3f} after moving it toward {target}")
        return f"drawer {start:.3f} -> {end:.3f} (target {target}); handle pitch {pitch:.2f}"

    def release_handle(self, *, hold: np.ndarray) -> None:
        """Open the hand, back straight off the handle along the approach axis, retract."""
        from pybullet_helpers.geometry import Pose
        from scipy.spatial.transform import Rotation

        self.motion.set_gripper(command=0.0, arm=hold)
        ee = self.scene.ee_now()
        a = Rotation.from_quat(ee.orientation).as_matrix()[:, 2]
        back = self.scene.linear_path(
            start=self.session.arm(),
            target=Pose(tuple(np.array(ee.position) - a * 0.10), ee.orientation),
        )
        if back:
            self.motion.follow(path=back, grip=0.0)
        self.motion.go_home(grip=0.0)

    # ================================================================ cubes
    def find_pick(
        self, *, cube: str, partner: str | None = None, max_stances: int = 40
    ) -> PickPlan:
        """Search stances and grasps for a pick the planning scene passes end to end:
        base path, hover, straight descent, arm plan. Moves nothing."""
        from pybullet_helpers.geometry import Pose

        where = self.session.location(cube=cube)
        c = self.session.position(name=cube)
        held = (
            (cube,)
            if partner is None
            else (cube, *(self.row_between(cube=cube, partner=partner) or ()), partner)
        )
        grasps = self.distinct(
            candidates=self.grasp_candidates(cube=cube)
            if partner is None
            else self.pair_candidates(cube=cube, partner=partner)
        )
        if not grasps:
            raise ExecutionError(f"{cube} ({where}) is boxed in: no grasp clears its neighbours")
        rejected: dict[str, int] = {}
        for tried, stance in enumerate(self.stances(target=c, where=where)):
            if tried >= max_stances:
                break
            path = self.scene.plan_base(target=stance)
            if path is None:
                rejected["base"] = rejected.get("base", 0) + 1
                continue
            base = path[-1]
            self.scene.sync(base=base)
            everything = self.scene.bodies()
            others = self.scene.bodies(without_cubes=held)
            bystanders = {self.scene.cube_body(cube=c) for c in S.CUBES if c not in held}
            for k in grasps:
                ez = k.pose.position[2]
                floor = DRAWER_HOVER_Z if where == "drawer" else 0.0
                back = max(HOVER, (floor - ez) / max(-k.approach[2], 0.3))
                hover = Pose(
                    tuple(np.array(k.pose.position) - k.approach * back), k.pose.orientation
                )
                q_h = self.scene.ik(pose=hover, seed=S.HOME)
                if q_h is None or self.scene.in_collision(
                    joints=self.scene.planning_fingers(arm=q_h, state=k.pb), bodies=everything
                ):
                    rejected["hover"] = rejected.get("hover", 0) + 1
                    continue
                down = self.scene.linear_path(
                    start=q_h,
                    target=k.pose,
                    bodies=others,
                    finger_state=k.pb,
                    margin=0.001,
                )
                if down is None:
                    rejected["descent"] = rejected.get("descent", 0) + 1
                    continue
                if self.scene.in_collision(
                    joints=self.scene.planning_fingers(
                        arm=down[-1], state=ShutHand.state(held=held)
                    ),
                    bodies=bystanders,
                    margin=ShutHand.MARGIN,
                ):
                    rejected["closing"] = rejected.get("closing", 0) + 1
                    continue
                plan = self.scene.plan_arm(goal=q_h, bodies=everything, start=S.HOME, base=base)
                if plan is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                return PickPlan(
                    grasp=k,
                    held=held,
                    path=path,
                    base=base,
                    back=back,
                    reach=plan,
                    down=down,
                    rejected=rejected,
                )
        raise ExecutionError(f"no feasible pick for {cube} ({where}); rejected {rejected}")

    def pick_feasible(self, *, cube: str) -> bool:
        """Whether the planner finds a pick for `cube` where it lies. Moves nothing."""
        try:
            self.find_pick(cube=cube)
        except ExecutionError:
            return False
        return True

    def pick(
        self, *, cube: str, partner: str | None = None, max_stances: int = 40
    ) -> tuple[Any, str]:
        """Top-down (possibly tilted) grasp -- or, with `partner`, a whole row at once;
        returns EE->cube."""
        from pybullet_helpers.geometry import Pose, multiply_poses

        c = self.session.position(name=cube)
        found = self.find_pick(cube=cube, partner=partner, max_stances=max_stances)
        self._last_pick = found
        k, held, back, base = found.grasp, found.held, found.back, found.base
        plan, down = found.reach, found.down
        self.motion.go_home(grip=0.0)
        if not self.motion.drive(path=found.path, grip=0.0):
            raise ExecutionError("base did not converge")
        moved = self.session.position(name=cube) - c
        target = Pose(tuple(np.array(k.pose.position) + moved), k.pose.orientation)
        hover_now = Pose(tuple(np.array(target.position) - k.approach * back), k.pose.orientation)
        redo = self.resolve_at_actual_base(
            hover=hover_now,
            target=target,
            bodies=self.scene.bodies(),
            descent_bodies=self.scene.bodies(without_cubes=held),
            finger_state=k.pb,
            margin=0.001,
            step=0.008,
        )
        if redo is not None:
            plan, down = redo
        self.motion.follow(path=plan, grip=0.0)
        self.motion.set_gripper(command=k.cmd)
        reached = self.motion.follow(path=down, grip=k.cmd, final_tol=0.006)
        ee_at = np.array(self.scene.ee_now().position)
        aim = np.array(target.position)
        cube_at = self.session.position(name=cube)
        pads = self.pad_centers()
        g = self.motion.set_gripper(command=1.0)
        ee = self.scene.ee_now()
        cube_pose = Pose(
            tuple(self.session.position(name=cube)), self.session.quaternion(name=cube)
        )
        ee_to_cube = multiply_poses(ee.invert(), cube_pose)
        up = self.scene.linear_path(
            start=self.session.arm(),
            target=Pose(tuple(np.array(ee.position) - k.approach * back), ee.orientation),
        )
        # the descent run backwards is a known-feasible way out
        self.motion.follow(path=up if up else list(reversed(down)), grip=1.0)
        lifted = all(self.session.position(name=h)[2] > c[2] + 0.06 for h in held)
        if not lifted:
            self.motion.set_gripper(command=0.0)
            self.motion.go_home(grip=0.0)
            raise ExecutionError(
                f"grasp missed or slipped (fingers {g:.2f});"
                f" stance {np.round(base, 2).tolist()}; descent reached {reached};"
                f" EE minus aim {np.round(ee_at - aim, 4).tolist()};"
                f" cube moved {np.round(cube_at - c, 4).tolist()};"
                f" approach {np.round(k.approach, 2).tolist()};"
                f" pads minus cube {[np.round(p - cube_at, 4).tolist() for p in pads]}"
            )
        note = (
            f"stance {np.round(base, 2).tolist()};"
            f" {k.kind} grasp, approach {np.round(k.approach, 2).tolist()},"
            f" closing at {np.degrees(k.yaw) % 180:.0f} deg;"
            f" gripper {g:.2f}; rejected {found.rejected}"
        )
        return ee_to_cube, note

    def place(
        self,
        *,
        cube: str,
        target_xy: tuple[float, float],
        ee_to_cube: Any,
        target_yaw: float | None = None,
        drop: float = 0.004,
        release: float = G.CUBE_OPEN_CMD,
    ) -> str:
        """Carry the held cube to `target_xy` on the countertop and release it just above,
        turned to `target_yaw` if given (the stock Sweep is anchored on cube_0's full pose).
        `release` is the gripper command that lets go: the single-cube opening unless a
        whole row is held."""
        from pybullet_helpers.geometry import Pose, multiply_poses
        from scipy.spatial.transform import Rotation

        tz = S.COUNTER_TOP + S.CUBE_HALF + drop
        ee_now = self.scene.ee_now()
        held_q = multiply_poses(ee_now, ee_to_cube).orientation
        cube_body = self.scene.cube_body(cube=cube)
        rejected: dict[str, int] = {}
        xmin = max(
            S.DRAWER_HANDLE_FRONT_X
            + max(self.session.drawer_pos(), 0.0)
            + S.CHASSIS_HALF[0]
            + 0.03,
            1.2,
        )
        turns = [0.0, np.pi / 2, -np.pi / 2, np.pi]
        if target_yaw is not None:
            held_yaw = CubeHeading.of(quaternion=held_q)
            first = float((target_yaw - held_yaw + np.pi) % (2 * np.pi) - np.pi)
            turns = [first] + [first + t for t in turns[1:]]
        for dyaw in turns:
            rot = Rotation.from_euler("z", dyaw)
            goal_cube = Pose(
                (target_xy[0], target_xy[1], tz),
                tuple((rot * Rotation.from_quat(held_q)).as_quat()),
            )
            goal = multiply_poses(goal_cube, ee_to_cube.invert())
            pre = Pose(
                (goal.position[0], goal.position[1], goal.position[2] + 0.10), goal.orientation
            )
            for x in (xmin, xmin + 0.08, xmin + 0.16):
                for y in (target_xy[1], target_xy[1] - 0.15, target_xy[1] + 0.15):
                    path = self.scene.plan_base(target=(x, y, np.pi))
                    if path is None:
                        rejected["base"] = rejected.get("base", 0) + 1
                        continue
                    base = path[-1]
                    self.scene.sync(base=base)
                    q_pre = self.scene.ik(pose=pre, seed=S.HOME)
                    down = (
                        None if q_pre is None else self.scene.linear_path(start=q_pre, target=goal)
                    )
                    carry = None
                    if down is not None:
                        carry = self.scene.plan_arm(
                            goal=q_pre,
                            bodies=self.scene.bodies(without_cubes=(cube,)),
                            start=S.HOME,
                            base=base,
                            held=cube_body,
                            held_tf=ee_to_cube,
                        )
                    if carry is None:
                        rejected["arm"] = rejected.get("arm", 0) + 1
                        continue
                    self.motion.go_home(grip=1.0, held=cube_body, held_tf=ee_to_cube)
                    if not self.motion.drive(path=path, grip=1.0):
                        raise ExecutionError("base did not converge")
                    redo = self.resolve_at_actual_base(
                        hover=pre,
                        target=goal,
                        bodies=self.scene.bodies(without_cubes=(cube,)),
                        descent_bodies=set(),
                        held=cube_body,
                        held_tf=ee_to_cube,
                    )
                    if redo is not None:
                        carry, down = redo
                    self.motion.follow(path=carry, grip=1.0)
                    self.motion.follow(path=down, grip=1.0, final_tol=0.006)
                    # Open only as far as letting go takes. Opened fully, a pad swings
                    # 5 cm out and shoves the cube in the next slot out of the pile.
                    self.motion.set_gripper(command=release)
                    up = self.scene.linear_path(start=self.session.arm(), target=pre)
                    if up:
                        self.motion.follow(path=up, grip=release)
                    self.motion.go_home(grip=0.0)
                    self.motion.hold(ticks=5, grip=0.0)
                    fin = self.session.position(name=cube)
                    err = float(np.hypot(fin[0] - target_xy[0], fin[1] - target_xy[1]))
                    if not (S.COUNTER_TOP < fin[2] < S.COUNTER_TOP + 0.03 and err < 0.02):
                        raise ExecutionError(
                            f"cube not at the slot after release"
                            f" (xy error {err:.3f}, z {fin[2]:.3f})"
                        )
                    return (
                        f"xy error {err:.3f}; stance ({base[0]:.2f},{base[1]:.2f});"
                        f" rejected {rejected}"
                    )
        raise ExecutionError(f"no feasible place; rejected {rejected}")

    # ================================================================ push (declutter)
    def _push_geometry(
        self, *, cube: str, direction: np.ndarray
    ) -> tuple[np.ndarray, Polygon, Polygon, Polygon]:
        """Wide-open push: one finger just behind the cube, the other 8.5 cm ahead of it,
        the palm (centred between them) clear of the drawer walls."""
        c = self.session.position(name=cube)[:2]
        sd = Footprints.support_distance(yaw=self.session.yaw(name=cube), direction=direction)
        ang = float(np.arctan2(direction[1], direction[0]))
        g = c + direction * (G.OPEN_HALF_GAP - sd - 0.003)
        t, w = G.FINGER_THICK, G.FINGER_HALF_WIDTH
        behind = Footprints.rect(
            center=g, yaw=ang, u0=-G.OPEN_HALF_GAP - t, u1=-G.OPEN_HALF_GAP, v0=-w, v1=w
        )
        ahead = Footprints.rect(
            center=g, yaw=ang, u0=G.OPEN_HALF_GAP, u1=G.OPEN_HALF_GAP + t, v0=-w, v1=w
        )
        hu, hv = G.PALM_HALF
        palm = Footprints.rect(center=g, yaw=ang, u0=-hu, u1=hu, v0=-hv, v1=hv)
        return g, behind, ahead, palm

    def push_candidates(self, *, cube: str) -> list[tuple[float, np.ndarray, float]]:
        """(score, direction, distance): pushes into free floor, scored by how many drawer
        cubes are graspable afterwards."""
        obst = self.obstacles_2d(cube=cube)
        walls = self.walls(cube=cube)
        c3 = self.session.position(name=cube)
        c = c3[:2]
        front = S.DRAWER_FRONT_INNER_X + self.session.drawer_pos()
        out = []
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            d = np.array([np.cos(ang), np.sin(ang)])
            _, behind, ahead, palm = self._push_geometry(cube=cube, direction=d)
            if (
                min(
                    Footprints.clearance(shape=behind, obstacles=obst),
                    Footprints.clearance(shape=ahead, obstacles=obst),
                )
                < 0.002
            ):
                continue
            if Footprints.clearance(shape=palm, obstacles=walls) < 0.003:
                continue
            sd = Footprints.support_distance(yaw=self.session.yaw(name=cube), direction=d)
            for dist in (0.05, 0.07, 0.035, 0.09):
                end = c + d * dist
                where = self.session.location(cube=cube)
                if where == "drawer" and not (
                    S.COUNTER_EDGE_X + 0.06 <= end[0] <= front - 0.02 and abs(end[1]) <= 0.28
                ):
                    continue
                if where == "floor" and end[0] < S.DRAWER_FACE_X + 0.06:
                    continue
                sweep = Footprints.swept(start=c, end=end, radius=sd + 0.003)
                ahead_sweep = ahead.union(translate(ahead, *(d * dist))).convex_hull
                moved = {cube: np.array([end[0], end[1], c3[2]])}
                if Footprints.clearance(shape=sweep, obstacles=obst) < 0.0005:
                    # Cubes in the path get shoved along too. Allowed only away from the
                    # drawer front (-x), where the whole cluster comes off the wall.
                    train = self.push_train(cube=cube, direction=d, dist=dist)
                    if d[0] > -0.7 or train is None:
                        continue
                    moved.update(train)
                if Footprints.clearance(shape=ahead_sweep, obstacles=obst) < 0.001:
                    continue
                if (
                    Footprints.clearance(shape=translate(palm, *(d * dist)), obstacles=walls)
                    < 0.003
                ):
                    continue
                n = self.graspable_after(moved=moved)
                if not n and d[0] < -0.7 and self.session.location(cube=cube) == "drawer":
                    # Nothing becomes graspable in the rigid 2D model, but shoving the
                    # cluster off the front wall is still progress: the real cubes spread.
                    out.append((-1.0 - 10 * dist, d, dist))
                    break
                if n:
                    # pushing along the slide axis drags the (undamped-only) drawer with it
                    out.append((n - 10 * dist - 0.5 * abs(d[0]), d, dist))
                    break
        out.sort(key=lambda t: -t[0])
        return out

    def push_train(
        self, *, cube: str, direction: np.ndarray, dist: float
    ) -> dict[str, np.ndarray] | None:
        """Cubes the pushed cube would shove ahead of it, moved rigidly by the same stroke;
        None if the train would leave the exposed drawer floor or hit a cube outside it."""
        front = S.DRAWER_FRONT_INNER_X + self.session.drawer_pos()
        members = {cube}
        grew = True
        while grew:
            grew = False
            for o in S.CUBES:
                if o in members or self.session.location(cube=o) != "drawer":
                    continue
                fo = Footprints.cube(
                    center=self.session.position(name=o)[:2], yaw=self.session.yaw(name=o)
                )
                for mbr in members:
                    p = self.session.position(name=mbr)[:2]
                    path = Footprints.swept(start=p, end=p + direction * dist, radius=0.0142)
                    if path.intersects(fo):
                        members.add(o)
                        grew = True
                        break
        out = {}
        for mbr in members - {cube}:
            p = self.session.position(name=mbr)
            end = p[:2] + direction * dist
            if not (S.COUNTER_EDGE_X + 0.06 <= end[0] <= front - 0.02 and abs(end[1]) <= 0.28):
                return None
            out[mbr] = np.array([end[0], end[1], p[2]])
        return out

    def graspable_after(self, *, moved: dict[str, np.ndarray]) -> int:
        n = 0
        for other in S.CUBES:
            if self.session.location(cube=other) != "drawer":
                continue
            obst = self.obstacles_2d(cube=other, moved=moved)
            if self.grasp_candidates(cube=other, at=moved.get(other), obstacles=obst):
                n += 1
        return n

    def push(self, *, cube: str, max_stances: int = 24) -> str:
        from pybullet_helpers.geometry import Pose

        where = self.session.location(cube=cube)
        c3 = self.session.position(name=cube)
        options = self.push_candidates(cube=cube)
        if not options:
            raise ExecutionError(f"no push frees {cube}")
        ez = self.support(cube=cube) + G.TIP_OPEN + 0.004
        hz = max(ez + HOVER, DRAWER_HOVER_Z if where == "drawer" else 0.0)
        rejected: dict[str, int] = {}
        for _, d, dist in options[:4]:
            g, *_ = self._push_geometry(cube=cube, direction=d)
            ang = float(np.arctan2(d[1], d[0]))
            q = Orientations.top_down(yaw=ang)
            start = Pose((g[0], g[1], ez), q)
            stop = Pose((g[0] + d[0] * (dist + 0.003), g[1] + d[1] * (dist + 0.003), ez), q)
            for tried, stance in enumerate(self.stances(target=c3, where=where)):
                if tried >= max_stances:
                    break
                path = self.scene.plan_base(target=stance)
                if path is None:
                    rejected["base"] = rejected.get("base", 0) + 1
                    continue
                base = path[-1]
                self.scene.sync(base=base)
                everything = self.scene.bodies()
                q_h = self.scene.ik(pose=Pose((g[0], g[1], hz), q), seed=S.HOME)
                if q_h is None or self.scene.in_collision(
                    joints=self.scene.planning_fingers(arm=q_h), bodies=everything
                ):
                    rejected["hover"] = rejected.get("hover", 0) + 1
                    continue
                down = self.scene.linear_path(
                    start=q_h, target=start, bodies=everything, margin=0.001
                )
                stroke = (
                    None
                    if down is None
                    else self.scene.linear_path(
                        start=down[-1],
                        target=stop,
                        step=0.005,
                        bodies=self.scene.bodies(without_cubes=(cube,)),
                    )
                )
                if stroke is None:
                    rejected["descent"] = rejected.get("descent", 0) + 1
                    continue
                plan = self.scene.plan_arm(goal=q_h, bodies=everything, start=S.HOME, base=base)
                if plan is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                self.motion.go_home(grip=0.0)
                self.motion.set_gripper(command=0.0)
                if not self.motion.drive(path=path, grip=0.0):
                    raise ExecutionError("base did not converge")
                redo = self.resolve_at_actual_base(
                    hover=Pose((g[0], g[1], hz), q),
                    target=start,
                    bodies=everything,
                    descent_bodies=everything,
                    margin=0.001,
                )
                if redo is not None:
                    stroke2 = self.scene.linear_path(
                        start=redo[1][-1],
                        target=stop,
                        step=0.005,
                        bodies=self.scene.bodies(without_cubes=(cube,)),
                    )
                    if stroke2 is not None:
                        (plan, down), stroke = redo, stroke2
                self.motion.follow(path=plan, grip=0.0)
                self.motion.follow(path=down, grip=0.0, final_tol=0.006)
                self.motion.follow(path=stroke, grip=0.0, tol=0.01, final_tol=0.006)
                up = self.scene.linear_path(
                    start=self.session.arm(),
                    target=Pose((stop.position[0], stop.position[1], hz), q),
                )
                if up:
                    self.motion.follow(path=up, grip=0.0)
                self.motion.go_home(grip=0.0)
                moved = float(np.linalg.norm(self.session.position(name=cube)[:2] - c3[:2]))
                return (
                    f"pushed {cube} toward {np.degrees(ang):.0f} deg"
                    f" by {moved:.3f} (planned {dist})"
                )
        raise ExecutionError(f"no executable push for {cube}; rejected {rejected}")
