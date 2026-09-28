"""Pre-grasp repositioning: moves that make a cube graspable without grasping it.

The sweep leaves cubes where no grasp fits: pressed to the drawer's front wall, where the
palm cannot descend; against each other, where no finger fits between them; or on the
floor against a lower drawer's face, turned so that a finger would have to go where the
face is. Nothing here lifts a cube. Each move only changes where cubes lie, so that
`Primitives.pick` has a grasp afterwards.
"""

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from shapely.geometry import Polygon

from .footprints import Footprints
from .motion import Angles, ExecutionError, Motion
from .planning_scene import Orientations, PlanningScene
from .primitives import DRAWER_HOVER_Z, HOVER, Primitives
from .session import SweepDrawerSession
from .types import GripperGeometry, NudgePlan, SweepDrawerScene

S = SweepDrawerScene
G = GripperGeometry
# How far above what a cube rests on the fingertip's lowest corner runs.
LOW = 0.004


class Repositioning(BaseModel):
    """Repositioning moves for one session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    scene: PlanningScene
    motion: Motion
    primitives: Primitives

    # The declared base action bound: 0.1 m per 10 Hz step.
    FULL_SPEED: ClassVar[float] = 0.1
    CLOSED: ClassVar[float] = 0.004
    # Upright first: a flat pad face slides a cube, a leaning one tends to tip it.
    TILTS: ClassVar[tuple[float, ...]] = (0.0, 0.45, 0.7, 0.9)
    # Beside a wall the palm has to lean well clear of it.
    WALL_TILTS: ClassVar[tuple[float, ...]] = (0.7, 0.9)
    DISTANCES: ClassVar[tuple[float, ...]] = (0.04, 0.06, 0.09, 0.12)
    ASIDE: ClassVar[tuple[float, ...]] = (0.0, 0.006, -0.006)
    # How far from a wall a cube has to lie for an upright grasp's palm to clear it.
    DRAWER_WALL_CLEAR: ClassVar[float] = 0.06
    # A floor cube nearer the island's drawer faces than this is "against" them: the
    # knuckles of a fingertip pushing along the faces would not clear them.
    ISLAND_NEAR: ClassVar[float] = 0.07

    # ================================================================ drawer wiggle
    def wall_gaps(self) -> dict[str, float]:
        """Each drawer cube's centre distance from the inner face of the front wall."""
        front = S.DRAWER_FRONT_INNER_X + self.session.drawer_pos()
        return {
            c: float(front - self.session.position(name=c)[0])
            for c in S.CUBES
            if self.session.location(cube=c) == "drawer"
        }

    def wiggle_drawer(
        self,
        *,
        clear: float = 0.065,
        max_strokes: int = 12,
        reopen_speed: float = 0.015,
        settle: int = 3,
    ) -> str:
        """Shake the cubes off the drawer's front wall with sharp closing strokes.

        The drawer slides on damping alone, and a cube stays put on its floor only through
        friction. Each stroke closes the drawer at the base's full speed into its joint
        limit: the drawer stops dead, the cubes slide on toward the back, away from the
        front wall. It is then re-opened gently, which moves nothing. The handle stays in
        the hand throughout.

        Two things about the stroke were measured, not assumed. The base must still be
        pushing when the drawer hits its limit: capping how far past shut it is commanded
        to 3 cm cut the slide from 1.7 cm a stroke to 0.3 cm. And the re-opening must be
        slow: the slam shoves the hand along the handle bar, and at 0.03 m a tick the
        loosened grip let go of the drawer on the second stroke, where at 0.015 it held.
        """
        before = self.wall_gaps()
        if not before:
            raise ExecutionError("no cube in the drawer to shake loose")
        opened = self.session.drawer_pos()
        self.primitives.grasp_handle()
        hold, home = self.session.arm(), self.session.base()
        strokes = regrips = 0
        for _ in range(max_strokes):
            gaps = self.wall_gaps()
            if not gaps or min(gaps.values()) >= clear:
                break
            self._stroke(hold=hold, home=home, reopen_speed=reopen_speed, settle=settle)
            strokes += 1
            if WiggleGrip.slipped(opened=opened, now=self.session.drawer_pos()):
                # the hand came back without the drawer: take the handle again where
                # the drawer now is, and pull it back out
                regrips += 1
                self.primitives.release_handle(hold=hold)
                self.primitives.move_drawer(target=opened)
                self.primitives.grasp_handle()
                hold, home = self.session.arm(), self.session.base()
        self.primitives.release_handle(hold=hold)
        after = self.wall_gaps()
        if not strokes:
            return f"already {min(before.values()):.3f} m clear of the front wall"
        moved = {c: round(after[c] - before[c], 3) for c in after if c in before}
        if moved and max(moved.values()) < 0.003:
            raise ExecutionError(f"{strokes} strokes moved no cube off the front wall")
        return (
            f"{strokes} strokes, {regrips} re-grips; nearest cube {min(before.values()):.3f} ->"
            f" {min(after.values(), default=0.0):.3f} m from the front wall; moved {moved}"
        )

    def _stroke(
        self,
        *,
        hold: np.ndarray,
        home: tuple[float, float, float],
        reopen_speed: float,
        settle: int,
    ) -> None:
        for _ in range(40):
            if self.session.drawer_pos() < self.CLOSED:
                break
            self._base_step(delta=(-self.FULL_SPEED, 0.0, 0.0), hold=hold)
        for _ in range(200):
            x, y, th = self.session.base()
            if abs(home[0] - x) < 0.004:
                break
            self._base_step(
                delta=(
                    float(np.clip(home[0] - x, -reopen_speed, reopen_speed)),
                    float(np.clip(home[1] - y, -0.02, 0.02)),
                    float(np.clip(Angles.wrap(angle=home[2] - th), -0.02, 0.02)),
                ),
                hold=hold,
            )
        self.motion.hold(ticks=settle, grip=1.0, arm=hold)

    def _base_step(self, *, delta: tuple[float, float, float], hold: np.ndarray) -> None:
        a = np.zeros(11)
        a[:3] = delta
        a[3:10] = np.clip(hold - self.session.arm(), -0.1, 0.1)
        a[-1] = 1.0
        self.session.step(action=a)

    # ================================================================ fingertip nudge
    def allowed(self, *, where: str, xy: np.ndarray) -> bool:
        """Whether a cube pushed to `xy` still lies where the robot can grasp it."""
        if where == "drawer":
            front = S.DRAWER_FRONT_INNER_X + self.session.drawer_pos()
            return bool(
                S.COUNTER_EDGE_X + 0.055 <= xy[0] <= front - self.DRAWER_WALL_CLEAR
                and abs(xy[1]) <= S.DRAWER_SIDE_INNER_Y - 0.07
            )
        # on the floor a cube may lie right up to the drawer faces: a grasp closing
        # along them, palm leaning away, takes it from there
        return bool(S.DRAWER_FACE_X + S.CUBE_HALF <= xy[0] <= 1.7 and abs(xy[1]) <= 0.95)

    def against_island(self, *, cube: str) -> bool:
        """On the floor, close in front of the island's lower drawer faces."""
        return bool(
            self.session.location(cube=cube) == "floor"
            and self.session.position(name=cube)[0] < S.DRAWER_FACE_X + self.ISLAND_NEAR
        )

    def train(
        self, *, cube: str, direction: np.ndarray, distance: float
    ) -> dict[str, np.ndarray] | None:
        """Every cube the stroke shifts, moved rigidly by it: the pushed cube and any it
        shoves ahead. None if one of them would end where it cannot be grasped."""
        where = self.session.location(cube=cube)
        members = {cube}
        grew = True
        while grew:
            grew = False
            for o in S.CUBES:
                if o in members or self.session.location(cube=o) != where:
                    continue
                shape = Footprints.cube(
                    center=self.session.position(name=o)[:2], yaw=self.session.yaw(name=o)
                )
                for mbr in tuple(members):
                    p = self.session.position(name=mbr)[:2]
                    reach = Footprints.support_distance(
                        yaw=self.session.yaw(name=mbr), direction=direction
                    )
                    path = Footprints.swept(
                        start=p, end=p + direction * distance, radius=reach + 0.002
                    )
                    if path.intersects(shape):
                        members.add(o)
                        grew = True
                        break
        out = {}
        for mbr in members:
            p = self.session.position(name=mbr)
            end = p[:2] + direction * distance
            if not self.allowed(where=where, xy=end):
                return None
            out[mbr] = np.array([end[0], end[1], p[2]])
        return out

    def free_after(self, *, moved: dict[str, np.ndarray], where: str) -> int:
        """Cubes at `where` with a grasp in the rigid 2D model once `moved` have moved."""
        n = 0
        for other in S.CUBES:
            if self.session.location(cube=other) != where:
                continue
            at = moved.get(other)
            xy = self.session.position(name=other)[:2] if at is None else at[:2]
            if not self.allowed(where=where, xy=xy):
                continue
            obstacles = self.primitives.obstacles_2d(cube=other, moved=moved)
            if self.primitives.grasp_candidates(cube=other, at=at, obstacles=obstacles):
                n += 1
        return n

    def tall_walls(self, *, where: str) -> list[Polygon]:
        """Surfaces that rise into the knuckles and palm, 4 cm and more above the tip."""
        if where == "drawer":
            return Footprints.drawer_walls(drawer_pos=self.session.drawer_pos())
        return []

    def low_walls(self, *, where: str) -> list[Polygon]:
        """Surfaces at the height of a cube: a drawer's walls, or on the floor the
        island's bottom board, which stands 2 cm back from the drawer faces above it."""
        if where == "floor":
            return [Footprints.island_board()]
        return self.tall_walls(where=where)

    def _neighbours(self, *, cube: str) -> list[Polygon]:
        z = self.session.position(name=cube)[2]
        return [
            Footprints.cube(center=self.session.position(name=o)[:2], yaw=self.session.yaw(name=o))
            for o in S.CUBES
            if o != cube and abs(self.session.position(name=o)[2] - z) < 0.05
        ]

    def nudge_candidates(self, *, cube: str) -> list[NudgePlan]:
        """Strokes of the closed fingertips, best first.

        A `push` leaves more cubes graspable in the rigid 2D model. A `stir` is a push
        that model expects nothing of: it moves a packed cluster as one block there,
        where real cubes scatter. Either is made from behind the cube with the palm
        leaning over it or, for a floor cube against the island's drawer faces, from
        beside it along the faces with the palm leaning out from them.
        """
        where = self.session.location(cube=cube)
        if where not in ("drawer", "floor"):
            return []
        out = self._pushes(cube=cube, where=where)
        if self.against_island(cube=cube):
            out += self._wall_pushes(cube=cube)
        out.sort(key=lambda k: -k.score)
        return out

    def _pushes(self, *, cube: str, where: str) -> list[NudgePlan]:
        c3 = self.session.position(name=cube)
        yaw = self.session.yaw(name=cube)
        cubes = self._neighbours(cube=cube)
        tall, low = self.tall_walls(where=where), self.low_walls(where=where)
        now = self.free_after(moved={}, where=where)
        blocked = not self.primitives.graspable(cube=cube)
        out: list[NudgePlan] = []
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            d = np.array([np.cos(ang), np.sin(ang)])
            side = np.array([-d[1], d[0]])
            rear = Footprints.support_distance(yaw=yaw, direction=d)
            best = self._best_push(cube=cube, where=where, direction=d, now=now, blocked=blocked)
            if best is None:
                continue
            kind, distance, gain, moved = best
            # along the slide axis a push drags the free-sliding drawer with it
            drag = 0.3 * abs(d[0]) if where == "drawer" else 0.0
            for alpha in self.TILTS:
                for aside in self.ASIDE:
                    aim = c3[:2] + aside * side
                    tip = NudgeGeometry.tip_footprint(
                        centre=aim, rear=rear, direction=d, alpha=alpha
                    )
                    if Footprints.clearance(shape=tip, obstacles=cubes + low) < 0.002:
                        continue
                    body = NudgeGeometry.body_footprint(
                        centre=aim, rear=rear, direction=d, alpha=alpha
                    )
                    if tall and Footprints.clearance(shape=body, obstacles=tall) < 0.003:
                        continue
                    score = gain - 2.0 * distance - 0.2 * alpha - 2.0 * abs(aside) - drag
                    out.append(
                        NudgePlan(
                            cube=cube,
                            kind=kind,
                            direction=d,
                            distance=distance,
                            alpha=alpha,
                            aim=aim,
                            rear=rear,
                            score=float(score - 0.1 * (len(moved) - 1)),
                            moved=moved,
                        )
                    )
                    break
        return out

    def _best_push(
        self, *, cube: str, where: str, direction: np.ndarray, now: int, blocked: bool
    ) -> tuple[Literal["push", "stir"], float, float, dict[str, np.ndarray]] | None:
        """The shortest stroke along `direction` that frees a cube; failing that, for a
        cube that has no grasp, the shortest stroke that stays in bounds, as a stir."""
        stir: tuple[Literal["push", "stir"], float, float, dict[str, np.ndarray]] | None = None
        for distance in self.DISTANCES:
            moved = self.train(cube=cube, direction=direction, distance=distance)
            if moved is None:
                continue
            gain = self.free_after(moved=moved, where=where) - now
            if gain > 0:
                return "push", distance, float(gain), moved
            if stir is None and blocked and distance >= 0.06:
                stir = ("stir", distance, -1.0, moved)
        return stir

    def _wall_pushes(self, *, cube: str) -> list[NudgePlan]:
        """Strokes along the island's drawer faces, for a floor cube too close to them
        for a fingertip to get behind it."""
        c3 = self.session.position(name=cube)
        yaw = self.session.yaw(name=cube)
        cubes = self._neighbours(cube=cube)
        low = self.low_walls(where="floor")
        away = np.array([1.0, 0.0])
        now = self.free_after(moved={}, where="floor")
        blocked = not self.primitives.graspable(cube=cube)
        out: list[NudgePlan] = []
        for d in (np.array([0.0, 1.0]), np.array([0.0, -1.0])):
            rear = Footprints.support_distance(yaw=yaw, direction=d)
            best = self._best_push(cube=cube, where="floor", direction=d, now=now, blocked=blocked)
            if best is None:
                continue
            kind, distance, gain, moved = best
            for alpha in self.WALL_TILTS:
                for shift in (0.0, 0.004, 0.008):
                    aim = c3[:2] + shift * away
                    tip = NudgeGeometry.side_tip_footprint(
                        centre=aim, rear=rear, direction=d, alpha=alpha
                    )
                    if Footprints.clearance(shape=tip, obstacles=cubes + low) < 0.002:
                        continue
                    score = gain - 2.0 * distance - 0.2 * alpha - 5.0 * shift
                    out.append(
                        NudgePlan(
                            cube=cube,
                            kind=kind,
                            direction=d,
                            distance=distance,
                            alpha=alpha,
                            aim=aim,
                            away=away,
                            rear=rear,
                            score=float(score - 0.1 * (len(moved) - 1)),
                            moved=moved,
                        )
                    )
                    break
        return out

    def poses(self, *, plan: NudgePlan, support: float) -> tuple[Any, Any]:
        """(start, stop) end-effector poses of a stroke."""
        if plan.away is None:
            start = NudgeGeometry.start(
                centre=plan.aim,
                support=support,
                rear=plan.rear,
                direction=plan.direction,
                alpha=plan.alpha,
            )
        else:
            start = NudgeGeometry.side_start(
                centre=plan.aim,
                support=support,
                rear=plan.rear,
                direction=plan.direction,
                away=plan.away,
                alpha=plan.alpha,
            )
        return start, NudgeGeometry.stop(
            start=start, direction=plan.direction, distance=plan.distance
        )

    def nudge(self, *, cube: str, max_stances: int = 48, max_plans: int = 6) -> str:
        """Move `cube` with the closed fingertips to where a grasp fits.

        The gripper is closed and tilted so the palm leans away from whatever wall is
        near: over the cube when the fingertip is behind it, out from the wall when the
        fingertip is beside it. Nothing is lifted; the cube slides, or tips and rolls.
        """
        from pybullet_helpers.geometry import Pose
        from scipy.spatial.transform import Rotation

        where = self.session.location(cube=cube)
        c3 = self.session.position(name=cube)
        yaw0 = self.session.yaw(name=cube)
        plans = self.nudge_candidates(cube=cube)
        if not plans:
            raise ExecutionError(f"no nudge frees {cube} ({where})")
        support = self.primitives.support(cube=cube)
        rejected: dict[str, int] = {}
        for plan in plans[:max_plans]:
            start, stop = self.poses(plan=plan, support=support)
            approach = Rotation.from_quat(start.orientation).as_matrix()[:, 2]
            ceiling = DRAWER_HOVER_Z if where == "drawer" else 0.0
            back = max(HOVER, (ceiling - start.position[2]) / max(-approach[2], 0.3))
            hover = Pose(tuple(np.array(start.position) - approach * back), start.orientation)
            movers = tuple(plan.moved)
            for tried, stance in enumerate(self.primitives.stances(target=c3, where=where)):
                if tried >= max_stances:
                    break
                path = self.scene.plan_base(target=stance)
                if path is None:
                    rejected["base"] = rejected.get("base", 0) + 1
                    continue
                base = path[-1]
                self.scene.sync(base=base)
                solved = self._solve_nudge(
                    hover=hover, start=start, stop=stop, movers=movers, rejected=rejected
                )
                if solved is None:
                    continue
                reach = self.scene.plan_arm(
                    goal=solved[0][0], bodies=self.scene.bodies(), start=S.HOME, base=base
                )
                if reach is None:
                    rejected["arm"] = rejected.get("arm", 0) + 1
                    continue
                self.motion.go_home(grip=0.0)
                self.motion.set_gripper(command=1.0)
                if not self.motion.drive(path=path, grip=1.0):
                    raise ExecutionError("base did not converge")
                self.scene.sync()
                redo = self._solve_nudge(
                    hover=hover, start=start, stop=stop, movers=movers, rejected={}
                )
                if redo is not None:
                    again = self.scene.plan_arm(goal=redo[0][0], bodies=self.scene.bodies())
                    if again is not None:
                        solved, reach = redo, again
                down, stroke = solved
                self.motion.follow(path=reach, grip=1.0)
                self.motion.follow(path=down, grip=1.0, final_tol=0.006)
                self.motion.follow(path=stroke, grip=1.0, tol=0.01, final_tol=0.006)
                ee = self.scene.ee_now()
                up = self.scene.linear_path(
                    start=self.session.arm(),
                    target=Pose(tuple(np.array(ee.position) - approach * back), ee.orientation),
                )
                self.motion.follow(path=up if up else list(reversed(down)), grip=1.0)
                self.motion.go_home(grip=1.0)
                self.motion.set_gripper(command=0.0)
                self.motion.hold(ticks=5, grip=0.0)
                end = self.session.position(name=cube)
                along = float((end[:2] - c3[:2]) @ plan.direction)
                turned = np.degrees(abs(Angles.wrap(angle=self.session.yaw(name=cube) - yaw0)))
                heading = np.degrees(np.arctan2(plan.direction[1], plan.direction[0]))
                what = (
                    f"{plan.kind} {cube}: {along:.3f} m toward {heading:.0f} deg"
                    f" (planned {plan.distance}), turned {turned:.0f} deg; tilt {plan.alpha:.2f}"
                )
                if along < 0.5 * plan.distance:
                    raise ExecutionError(
                        f"{what}; fell short; it is now {self.session.location(cube=cube)}"
                    )
                return (
                    f"{what}; shoved along {[o for o in movers if o != cube]}; rejected {rejected}"
                )
        raise ExecutionError(f"no executable nudge for {cube} ({where}); rejected {rejected}")

    def _solve_nudge(
        self,
        *,
        hover: Any,
        start: Any,
        stop: Any,
        movers: tuple[str, ...],
        rejected: dict[str, int],
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """(descent from the hover, stroke) at the scene's current base, or None."""
        everything = self.scene.bodies()
        q_h = self.scene.ik(pose=hover, seed=S.HOME)
        if q_h is None or self.scene.in_collision(
            joints=self.scene.fingers(arm=q_h, state=G.CLOSED_PB), bodies=everything
        ):
            rejected["hover"] = rejected.get("hover", 0) + 1
            return None
        down = self.scene.linear_path(
            start=q_h, target=start, bodies=everything, finger_state=G.CLOSED_PB, margin=0.001
        )
        if down is None:
            rejected["descent"] = rejected.get("descent", 0) + 1
            return None
        stroke = self.scene.linear_path(
            start=down[-1],
            target=stop,
            step=0.005,
            bodies=self.scene.bodies(without_cubes=movers),
            finger_state=G.CLOSED_PB,
        )
        if stroke is None:
            rejected["stroke"] = rejected.get("stroke", 0) + 1
            return None
        return [q_h, *down], stroke


class WiggleGrip:
    """Whether the drawer came back out with the hand after a stroke."""

    # The drawer overshoots its open position by a centimetre or two on the way back;
    # short of it by more than this, the handle is no longer in the hand.
    SHORT: ClassVar[float] = 0.04

    @staticmethod
    def slipped(*, opened: float, now: float) -> bool:
        return now < opened - WiggleGrip.SHORT


class NudgeGeometry:
    """The closed fingertips as a pushing tool.

    Measured on the compiled MuJoCo gripper, closed, in the end-effector frame (x the
    closing axis, z the approach): the two pads together are 1.8 cm along x and 2.2 cm
    along y, and end 4.8 cm down z. Along x the fingers widen to 1.5 cm from the axis by
    the height of a cube's top, and the knuckles flare to 6.5 cm from 4.3 cm above the
    tip; along y nothing passes 2 cm until the palm, 7.3 cm up.

    The tilt is always about the closing axis, so the palm leans along y. Behind a cube
    (`start`) the push runs along y and the palm leans over the cube. Beside a wall
    (`side_start`) the push runs along x, one finger's outer face doing the pushing, and
    the palm leans out from the wall.
    """

    TIP_HALF_ALONG: ClassVar[float] = 0.011
    TIP_HALF_ACROSS: ClassVar[float] = 0.016
    # Reach of a finger's outer face from the closing axis, at the height of a cube's top.
    FINGER_REACH: ClassVar[float] = 0.015
    BODY_HALF_ACROSS: ClassVar[float] = 0.065
    PALM_HALF_ALONG: ClassVar[float] = 0.038
    PALM_HEIGHT: ClassVar[float] = 0.073
    FLOOR_CLEARANCE: ClassVar[float] = LOW
    CONTACT_MARGIN: ClassVar[float] = 0.003

    @staticmethod
    def overhang(*, alpha: float, lift: float = LOW) -> float:
        """How far the leaning finger's leading face reaches past its lowest corner by
        the height of the cube's top."""
        return float((2 * S.CUBE_HALF - lift) * np.tan(alpha))

    @staticmethod
    def reach_behind(*, alpha: float, lift: float = LOW) -> float:
        """Room the fingertip takes behind the cube's rear face at the start of a stroke."""
        return float(
            NudgeGeometry.CONTACT_MARGIN
            + NudgeGeometry.overhang(alpha=alpha, lift=lift)
            + 2 * NudgeGeometry.TIP_HALF_ALONG * np.cos(alpha)
        )

    @staticmethod
    def height(*, support: float, alpha: float, lift: float) -> float:
        """End-effector height that puts the tilted fingertip's lowest corner `lift`
        above `support`."""
        return float(
            support
            + lift
            + NudgeGeometry.TIP_HALF_ALONG * np.sin(alpha)
            + G.TIP_CLOSED * np.cos(alpha)
        )

    @staticmethod
    def start(
        *,
        centre: np.ndarray,
        support: float,
        rear: float,
        direction: np.ndarray,
        alpha: float,
        lift: float = LOW,
    ) -> Any:
        """End-effector pose with the fingertip just behind the cube, about to push it
        along `direction`; `rear` is the cube's extent from its centre against the push."""
        from pybullet_helpers.geometry import Pose

        d = np.asarray(direction, dtype=float)
        side = np.array([-d[1], d[0]])
        quat, _ = Orientations.tilted(
            yaw=float(np.arctan2(d[1], d[0])) - np.pi / 2, lean=1.0, alpha=alpha
        )
        low = (
            float(np.asarray(centre) @ d)
            - rear
            - NudgeGeometry.CONTACT_MARGIN
            - NudgeGeometry.overhang(alpha=alpha, lift=lift)
        )
        along = low - NudgeGeometry.TIP_HALF_ALONG * np.cos(alpha) + G.TIP_CLOSED * np.sin(alpha)
        xy = along * d + float(np.asarray(centre) @ side) * side
        z = NudgeGeometry.height(support=support, alpha=alpha, lift=lift)
        return Pose((float(xy[0]), float(xy[1]), z), quat)

    @staticmethod
    def side_start(
        *,
        centre: np.ndarray,
        support: float,
        rear: float,
        direction: np.ndarray,
        away: np.ndarray,
        alpha: float,
        lift: float = LOW,
        shift: float = 0.0,
    ) -> Any:
        """End-effector pose with the fingertip behind the cube along `direction` and the
        palm leaning toward `away`, a horizontal unit vector across the push; `shift`
        moves the fingertip off the cube's centre line toward `away`."""
        from pybullet_helpers.geometry import Pose

        d = np.asarray(direction, dtype=float)
        w = np.asarray(away, dtype=float)
        lean = 1.0 if float(np.array([-d[1], d[0]]) @ w) > 0 else -1.0
        quat, _ = Orientations.tilted(yaw=float(np.arctan2(d[1], d[0])), lean=lean, alpha=alpha)
        along = (
            float(np.asarray(centre) @ d)
            - rear
            - NudgeGeometry.CONTACT_MARGIN
            - NudgeGeometry.FINGER_REACH
        )
        across = float(np.asarray(centre) @ w) + shift + G.TIP_CLOSED * np.sin(alpha)
        xy = along * d + across * w
        z = NudgeGeometry.height(support=support, alpha=alpha, lift=lift)
        return Pose((float(xy[0]), float(xy[1]), z), quat)

    @staticmethod
    def stop(*, start: Any, direction: np.ndarray, distance: float) -> Any:
        """The stroke's end: the cube is met after the contact margin, then carried."""
        from pybullet_helpers.geometry import Pose

        travel = (distance + NudgeGeometry.CONTACT_MARGIN) * np.asarray(direction, dtype=float)
        p = start.position
        return Pose((float(p[0] + travel[0]), float(p[1] + travel[1]), p[2]), start.orientation)

    @staticmethod
    def tip_footprint(
        *,
        centre: np.ndarray,
        rear: float,
        direction: np.ndarray,
        alpha: float,
        lift: float = LOW,
    ) -> Polygon:
        """The fingers' footprint below the height of a cube's top, at the start pose."""
        return Footprints.rect(
            center=centre,
            yaw=float(np.arctan2(direction[1], direction[0])),
            u0=-rear - NudgeGeometry.reach_behind(alpha=alpha, lift=lift),
            u1=-rear - NudgeGeometry.CONTACT_MARGIN,
            v0=-NudgeGeometry.TIP_HALF_ACROSS,
            v1=NudgeGeometry.TIP_HALF_ACROSS,
        )

    @staticmethod
    def side_tip_footprint(
        *, centre: np.ndarray, rear: float, direction: np.ndarray, alpha: float
    ) -> Polygon:
        """The same for a push along the closing axis: both fingers lie along the push."""
        half = NudgeGeometry.TIP_HALF_ALONG * float(np.cos(alpha))
        return Footprints.rect(
            center=centre,
            yaw=float(np.arctan2(direction[1], direction[0])),
            u0=-rear - NudgeGeometry.CONTACT_MARGIN - 2 * NudgeGeometry.FINGER_REACH,
            u1=-rear - NudgeGeometry.CONTACT_MARGIN,
            v0=-half,
            v1=half,
        )

    @staticmethod
    def body_footprint(
        *,
        centre: np.ndarray,
        rear: float,
        direction: np.ndarray,
        alpha: float,
        lift: float = LOW,
    ) -> Polygon:
        """Knuckles and palm, which ride above the cubes but below a drawer wall's top."""
        tip_centre = (
            -rear
            - NudgeGeometry.CONTACT_MARGIN
            - NudgeGeometry.overhang(alpha=alpha, lift=lift)
            - NudgeGeometry.TIP_HALF_ALONG * np.cos(alpha)
        )
        lean = NudgeGeometry.PALM_HEIGHT * np.sin(alpha)
        return Footprints.rect(
            center=centre,
            yaw=float(np.arctan2(direction[1], direction[0])),
            u0=tip_centre + min(lean - NudgeGeometry.PALM_HALF_ALONG, -0.02),
            u1=tip_centre + lean + 2 * NudgeGeometry.PALM_HALF_ALONG,
            v0=-NudgeGeometry.BODY_HALF_ACROSS,
            v1=NudgeGeometry.BODY_HALF_ACROSS,
        )
