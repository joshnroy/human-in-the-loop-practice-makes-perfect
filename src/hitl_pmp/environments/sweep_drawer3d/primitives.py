"""The reset's primitives: park the wiper, open/close the drawer, pick, place, push.

Each primitive searches a small set of base stances and grasps, checks every candidate
against the planning scene (IK, collision-free approach, straight-line descent), and
only then moves the robot. A primitive that finds nothing raises ExecutionError without
having touched the world.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from shapely.affinity import translate
from shapely.geometry import Polygon

from .footprints import Footprints
from .motion import ExecutionError, Motion
from .planning_scene import Orientations, PlanningScene
from .session import SweepDrawerSession
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


class Opening:
    """Gripper command <-> inner pad gap, measured in both models (probe_grip.py)."""

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


class Primitives(BaseModel):
    """All reset primitives for one session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    scene: PlanningScene
    motion: Motion

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
        out += self.walls(cube=cube)
        return out

    def walls(self, *, cube: str) -> list[Polygon]:
        """Vertical surfaces taller than the palm around `cube`: the drawer's walls for a
        cube inside it, the island's front (drawer faces) for a cube on the floor."""
        from shapely.geometry import box

        where = self.session.location(cube=cube)
        if where == "drawer":
            return Footprints.drawer_walls(drawer_pos=self.session.drawer_pos())
        if where == "floor":
            front = S.DRAWER_FACE_X + max(self.session.drawer_pos(), 0.0)
            return [box(-1.0, -1.1, front, 1.1)]
        return []

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
        """Face grasps: both closing axes x tilts that lean the palm away from walls x small
        offsets, best first. Kept only if the fingers clear everything in 2D and, in the
        drawer, the palm clears the walls."""
        from pybullet_helpers.geometry import Pose

        c = self.session.position(name=cube) if at is None else at
        support = self.support(cube=cube)
        obst = self.obstacles_2d(cube=cube) if obstacles is None else obstacles
        walls = self.walls(cube=cube)
        out: list[Candidate] = []
        # Face grasps at a 3.2 cm opening, and corner-to-corner (diagonal) grasps at 3.8 cm:
        # squeezing two opposite vertical edges turns the cube face-on between the pads,
        # and it is the only way to get a finger past a cube lying corner-first at a wall.
        faces = [(y, 0.032, 0.0) for y in self.face_yaws(cube=cube)]
        diagonals = [(y + np.pi / 4, 0.038, -0.004) for y in self.face_yaws(cube=cube)[:2]]
        # Beside a wall, also close parallel to it whatever the cube's heading: the pads
        # land on its corners and square it up, and neither finger goes near the wall.
        along_wall = []
        if walls:
            th = self.session.yaw(name=cube)
            for w in (np.pi / 2, -np.pi / 2):
                width = 2 * S.CUBE_HALF * (abs(np.cos(th - w)) + abs(np.sin(th - w)))
                along_wall.append((w, min(width + 0.01, 0.08), -0.005))
        for yaw, gap, bonus in faces + diagonals + along_wall:
            cmd, pb, depth = Opening.for_gap(gap=gap)
            u = np.array([np.cos(yaw), np.sin(yaw)])
            v = np.array([-np.sin(yaw), np.cos(yaw)])
            for alpha in (0.0, 0.45, 0.7) + ((1.05,) if walls else ()):
                for lean in (0.0,) if alpha == 0 else (1.0, -1.0):
                    for off_v in (0.0, 0.003, -0.003, 0.005, -0.005) + (
                        (0.008, -0.008, 0.011, -0.011) if walls else ()
                    ):
                        for off_u in (0.0, 0.002, -0.002):
                            ctr = c[:2] + off_v * v + off_u * u
                            reach = 0.019 * np.sin(alpha)
                            clr = Footprints.finger_clearance(
                                center=ctr - 0.5 * reach * lean * v,
                                yaw=yaw,
                                obstacles=obst,
                                gap=gap,
                                widen=reach,
                            )
                            if clr < 0.002:
                                continue
                            if walls:
                                palm = Footprints.palm(center=ctr, yaw=yaw, lean=lean, alpha=alpha)
                                if Footprints.clearance(shape=palm, obstacles=walls) < 0.003:
                                    continue
                            q, a = Orientations.tilted(yaw=yaw, lean=lean, alpha=alpha)
                            pad = np.array([
                                ctr[0],
                                ctr[1],
                                support + 0.0124 + 0.012 * np.sin(alpha),
                            ])
                            pose = Pose(tuple(pad - a * depth), q)
                            score = (
                                min(clr, 0.01)
                                - 0.3 * (abs(off_v) + abs(off_u))
                                - 0.004 * alpha
                                + bonus
                            )
                            out.append(
                                Candidate(
                                    pose=pose, approach=a, yaw=yaw, score=score, cmd=cmd, pb=pb
                                )
                            )
        out.sort(key=lambda k: -k.score)
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
        ] + self.walls(cube=cube)
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
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """Re-solve hover IK, the arm plan and the straight descent at the base pose the
        robot actually reached (the base stops within millimetres, not exactly)."""
        self.scene.sync()
        q_h = self.scene.ik(pose=hover, seed=S.HOME)
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
    def park_wiper(self) -> str:
        """Carry the held wiper back to its episode-start pose on the counter; release."""
        from pybullet_helpers.geometry import Pose, multiply_poses

        home_pos, home_q = self.session.initial_pose(name=S.WIPER)
        ee = self.scene.ee_now()
        wiper_now = Pose(
            tuple(self.session.position(name=S.WIPER)), self.session.quaternion(name=S.WIPER)
        )
        ee_to_wiper = multiply_poses(ee.invert(), wiper_now)
        goal = multiply_poses(
            Pose((home_pos[0], home_pos[1], home_pos[2] + 0.012), home_q), ee_to_wiper.invert()
        )
        pre = Pose((goal.position[0], goal.position[1], goal.position[2] + 0.08), goal.orientation)
        lift = None
        for rise in (0.08, 0.05, 0.03, 0.015):
            raised = Pose((ee.position[0], ee.position[1], ee.position[2] + rise), ee.orientation)
            lift = self.scene.linear_path(start=self.session.arm(), target=raised, max_jump=0.6)
            if lift is not None:
                break
        if lift is None:
            raise ExecutionError("cannot lift the wiper straight up")
        self.motion.follow(path=lift, grip=1.0)
        rejected: dict[str, int] = {}
        for d in (0.7, 0.65, 0.75, 0.6):
            for dy in (0.0, 0.1, -0.1):
                path = self.scene.plan_base(target=(home_pos[0] + d, home_pos[1] + dy, np.pi))
                if path is None:
                    rejected["base"] = rejected.get("base", 0) + 1
                    continue
                base = path[-1]
                self.scene.sync(base=base)
                q_pre = self.scene.ik(pose=pre, seed=S.HOME)
                down = None if q_pre is None else self.scene.linear_path(start=q_pre, target=goal)
                carry = None
                if down is not None:
                    carry = self.scene.plan_arm(
                        goal=q_pre,
                        bodies=self.scene.bodies(),
                        base=base,
                        held=self.scene.wiper_body,
                        held_tf=ee_to_wiper,
                    )
                if carry is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                self.motion.drive(path=path, grip=1.0)
                redo = self.resolve_at_actual_base(
                    hover=pre,
                    target=goal,
                    bodies=self.scene.bodies(),
                    descent_bodies=set(),
                    held=self.scene.wiper_body,
                    held_tf=ee_to_wiper,
                )
                if redo is not None:
                    carry, down = redo
                self.motion.follow(path=carry, grip=1.0)
                self.motion.follow(path=down, grip=1.0, final_tol=0.01)
                self.motion.set_gripper(command=0.0)
                up = self.scene.linear_path(start=self.session.arm(), target=pre)
                if up:
                    self.motion.follow(path=up, grip=0.0)
                self.motion.go_home(grip=0.0)
                self.motion.hold(ticks=10, grip=0.0)
                return f"stance ({base[0]:.2f},{base[1]:.2f}); rejected {rejected}"
        raise ExecutionError(f"no stance to park the wiper; rejected {rejected}")

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
                    joints=self.scene.fingers(arm=q_pre), bodies=everything
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
        ticks = int(abs(target - start) / speed) + 60
        # closing: aim 1 cm past shut so the drawer seats on its stop (seed 1 stopped 1 mm
        # short of the 1 cm criterion when aimed exactly at 0)
        stroke = (target - 0.01 if target <= 0.0 else target) - start
        self.motion.drive_straight(
            target=(x + stroke, y, th), grip=1.0, arm=hold, speed=speed, max_ticks=ticks
        )
        self.motion.hold(ticks=5, grip=1.0, arm=hold)
        self.release_handle(hold=hold)
        end = self.session.drawer_pos()
        if abs(end - target) > 0.03:
            raise ExecutionError(f"drawer at {end:.3f} after moving it toward {target}")
        return f"drawer {start:.3f} -> {end:.3f} (target {target}); handle pitch {pitch:.2f}"

    def release_handle(self, *, hold: np.ndarray) -> None:
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

    def slam_drawer(
        self, *, strokes: int = 2, open_to: float = 0.25, top_speed: float = 1.0
    ) -> str:
        """Shake cubes off the drawer's front wall: holding the handle, accelerate the
        drawer shut gently (2 m/s^2, under the cubes' ~9.8 m/s^2 friction limit, so they
        ride along), meet the closed stop at speed so they slide on toward the back, then
        reopen slowly so they ride out again. Measured on four packed clusters: 1.5-4 cm
        off the wall per stroke; a fast start instead presses them into the wall."""
        s = SweepDrawerScene
        wall = s.DRAWER_FRONT_INNER_X

        def gaps() -> list[float]:
            dp = self.session.drawer_pos()
            return [
                wall + dp - self.session.position(name=c)[0] - s.CUBE_HALF
                for c in s.CUBES
                if self.session.location(cube=c) == "drawer"
            ]

        before = min(gaps(), default=0.0)
        self.grasp_handle()
        hold = self.session.arm()
        for _ in range(strokes):
            v = 0.0
            for _ in range(60):
                if self.session.drawer_pos() < 0.004:
                    break
                v = min(top_speed, v + 2.0 * 0.1)
                a = np.zeros(11)
                a[0] = -v * 0.1
                a[3:10] = np.clip(hold - self.session.arm(), -0.1, 0.1)
                a[-1] = 1.0
                self.session.step(action=a)
            self.motion.hold(ticks=5, grip=1.0, arm=hold)
            x, y, th = self.session.base()
            dp = self.session.drawer_pos()
            self.motion.drive_straight(
                target=(x + open_to - dp, y, th), grip=1.0, arm=hold, speed=0.012, max_ticks=80
            )
            self.motion.hold(ticks=5, grip=1.0, arm=hold)
        self.release_handle(hold=hold)
        after = min(gaps(), default=0.0)
        return (
            f"{strokes} slam strokes: closest cube {before:.3f} -> {after:.3f} m off the front"
            f" wall; drawer at {self.session.drawer_pos():.3f}"
        )

    # ================================================================ cubes
    def pick(
        self, *, cube: str, partner: str | None = None, max_stances: int = 40
    ) -> tuple[Any, str]:
        """Top-down (possibly tilted) face grasp at 3.2 cm opening -- or, with `partner`,
        both cubes at once; returns EE->cube."""
        from pybullet_helpers.geometry import Pose, multiply_poses

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
            raise ExecutionError(
                f"{cube} ({where}) is boxed in: no face grasp clears its neighbours"
            )
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
            for k in grasps:
                ez = k.pose.position[2]
                floor = DRAWER_HOVER_Z if where == "drawer" else 0.0
                back = max(HOVER, (floor - ez) / max(-k.approach[2], 0.3))
                hover = Pose(
                    tuple(np.array(k.pose.position) - k.approach * back), k.pose.orientation
                )
                q_h = self.scene.ik(pose=hover, seed=S.HOME)
                if q_h is None or self.scene.in_collision(
                    joints=self.scene.fingers(arm=q_h, state=k.pb), bodies=everything
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
                plan = self.scene.plan_arm(goal=q_h, bodies=everything, start=S.HOME, base=base)
                if plan is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                # ---- execute
                self.motion.go_home(grip=0.0)
                if not self.motion.drive(path=path, grip=0.0):
                    raise ExecutionError("base did not converge")
                moved = self.session.position(name=cube) - c
                target = Pose(tuple(np.array(k.pose.position) + moved), k.pose.orientation)
                hover_now = Pose(
                    tuple(np.array(target.position) - k.approach * back), k.pose.orientation
                )
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
                    f" approach {np.round(k.approach, 2).tolist()};"
                    f" gripper {g:.2f}; rejected {rejected}"
                )
                return ee_to_cube, note
        raise ExecutionError(f"no feasible pick for {cube} ({where}); rejected {rejected}")

    def place(
        self,
        *,
        cube: str,
        target_xy: tuple[float, float],
        ee_to_cube: Any,
        target_yaw: float | None = None,
        drop: float = 0.004,
    ) -> str:
        """Carry the held cube to `target_xy` on the countertop and release it just above,
        turned to `target_yaw` if given (the stock Sweep is anchored on cube_0's full pose)."""
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
            held_yaw = float(Rotation.from_quat(held_q).as_euler("zyx")[0])
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
                    self.motion.set_gripper(command=0.0)
                    up = self.scene.linear_path(start=self.session.arm(), target=pre)
                    if up:
                        self.motion.follow(path=up, grip=0.0)
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
                    joints=self.scene.fingers(arm=q_h), bodies=everything
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
