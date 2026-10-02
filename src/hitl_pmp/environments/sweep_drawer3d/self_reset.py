"""The robot's own reset after a SweepIntoDrawer3D attempt.

Order: park the wiper where it started (OpenDrawer and Sweep are anchored on the wiper's
and cube_0's poses, so "somewhere re-pickable" is not enough); pull the drawer out to
0.25 m so the cubes clear the countertop; shake the cubes off its front wall; pick every
drawer cube, nudging apart any that are boxed in; close the drawer; then fetch floor
cubes and strays, dragging clear any that lie against the island; park the base.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .motion import ExecutionError, Motion
from .planning_scene import PlanningScene
from .primitives import DrawerStroke, Primitives, WiperHold
from .repositioning import Repositioning
from .session import CubeHeading, SweepDrawerSession
from .types import CubeLocation, Mechanisms, ResetOutcome, Retrieval, SweepDrawerScene

S = SweepDrawerScene


class SweepDrawerSelfReset(BaseModel):
    """Scripted, planning-backed self-reset for one session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    open_to: float = 0.25
    max_drawer_rounds: int = 30
    max_pushes: int = 10
    max_nudges: int = 12
    max_wiggles: int = 3
    # Mechanisms to run without (see `Mechanisms`): for ablations, never for use.
    without: frozenset[str] = frozenset()

    # Countertop slots inside `blocks_init_region` (x 0.625..0.825, y -0.2..0), 4-4.5 cm apart.
    SLOTS: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.70, -0.06),
        (0.745, -0.10),
        (0.70, -0.14),
        (0.655, -0.10),
        (0.745, -0.035),
        (0.655, -0.035),
        (0.745, -0.165),
        (0.655, -0.165),
    )
    PARK: ClassVar[tuple[float, float, float]] = (1.45, -0.25, float(np.pi))
    # A repositioning move "shifted" a cube if it moved it this far within its container.
    SHIFTED: ClassVar[float] = 0.005
    # How far out the drawer carries a cube on its face for the fingers to clear the
    # countertop's edge on both sides of it.
    FRONT_CLEAR: ClassVar[float] = 0.08

    _scene: Any = PrivateAttr(default=None)
    _motion: Any = PrivateAttr(default=None)
    _prims: Any = PrivateAttr(default=None)
    _rep: Any = PrivateAttr(default=None)
    _actions: int = PrivateAttr(default=0)
    _origin: dict[str, CubeLocation] = PrivateAttr(default_factory=dict)
    _blocked: dict[str, bool] = PrivateAttr(default_factory=dict)
    _refused: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _assists: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _grasp: dict[str, Literal["single", "squeeze", "row"]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        off = Mechanisms.check(names=self.without)
        self._scene = PlanningScene(session=self.session, other_drawers="drawers" not in off)
        self._motion = Motion(session=self.session, scene=self._scene)
        self._prims = Primitives(
            session=self.session,
            scene=self._scene,
            motion=self._motion,
            squeeze="squeeze" not in off,
            finger_board="board" not in off,
        )
        self._rep = Repositioning(
            session=self.session, scene=self._scene, motion=self._motion, primitives=self._prims
        )
        self._actions = 0

    @property
    def primitives(self) -> Primitives:
        return self._prims

    @property
    def repositioning(self) -> Repositioning:
        return self._rep

    # ------------------------------------------------------------------ bookkeeping
    def _do(self, *, name: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Run one primitive as one logged robot action; None if it failed."""
        self.session.begin(name=name, kind=fn.__name__, phase="reset")
        try:
            out = fn(**kwargs)
        except ExecutionError as e:
            self._actions += 1
            self.session.end(success=False, note=str(e))
            return None
        except Exception:
            self._actions += 1
            self.session.end(success=False, note="error: " + traceback.format_exc()[-600:])
            return None
        self._actions += 1
        note = out[1] if isinstance(out, tuple) else str(out)
        self.session.end(success=True, note=note)
        return out

    def _in_container(self, *, cube: str) -> tuple[CubeLocation, np.ndarray]:
        """Where a cube lies relative to what carries it: a drawer cube rides the drawer,
        so sliding the drawer alone has not shifted it."""
        where = self.session.location(cube=cube)
        p = self.session.position(name=cube).copy()
        if where == "drawer":
            p[0] -= self.session.drawer_pos()
        return where, p

    def _reposition(self, *, name: str, fn: Callable[..., Any], strategy: str, **kw: Any) -> Any:
        """One repositioning move, crediting `strategy` to every cube it shifted --
        whether or not the move then reported success: a stroke that fell short of its
        plan still moved what it moved."""
        waiting = [c for c in self._origin if not self.session.in_pile(cube=c)]
        before = {c: self._in_container(cube=c) for c in waiting}
        out = self._do(name=name, fn=fn, **kw)
        for c, (where, p) in before.items():
            now_where, now = self._in_container(cube=c)
            if now_where != where or float(np.linalg.norm(now - p)) > self.SHIFTED:
                self._assists.setdefault(c, []).append(strategy)
        return out

    def _examine(self, *, cubes: list[str]) -> None:
        """Note, the first time each cube is looked at, whether the pick planner finds a
        pick where the sweep left it. The wiggle runs before any pick is tried, so this is
        the only record of whether a cube needed it; the 2D grasp filter alone would not
        do, since it passes grasps the planning scene then rejects."""
        for c in cubes:
            if c not in self._blocked:
                self._blocked[c] = not self._prims.pick_feasible(cube=c)
                if self._blocked[c]:
                    self._refused[c] = self._surroundings(cube=c)

    def _surroundings(self, *, cube: str) -> np.ndarray:
        """What a pick of `cube` depends on: where it lies, where the cubes near it lie,
        and how far out the drawer is."""
        at = self.session.position(name=cube)
        near = [
            self.session.position(name=c)
            for c in S.CUBES
            if c != cube and float(np.linalg.norm(self.session.position(name=c) - at)) < 0.12
        ]
        return np.concatenate([at, *near, [self.session.drawer_pos()]])

    def _stuck(self, *, cube: str) -> bool:
        """The planner found no pick for `cube` and nothing it depends on has moved since,
        so asking again would only cost the search again."""
        return Unmoved.since(before=self._refused.get(cube), now=self._surroundings(cube=cube))

    def _pick_and_place(self, *, cube: str) -> bool:
        """Pick one cube and put it in the pile; False if the pick found nothing."""
        picked = self._do(name=f"pick_{cube}", fn=self._prims.pick, cube=cube)
        if picked is None:
            if cube not in self._grasp:
                self._blocked[cube] = True
            self._refused[cube] = self._surroundings(cube=cube)
            return False
        self._grasp[cube] = self._prims.last_pick.grasp.kind
        target, yaw = self._target(cube=cube)
        placed = self._do(
            name=f"place_{cube}",
            fn=self._prims.place,
            cube=cube,
            target_xy=target,
            target_yaw=yaw,
            ee_to_cube=picked[0],
        )
        if placed is None:
            self._recover_hand()
        return True

    def _target(
        self, *, cube: str, spacing: float = 0.045, margin: float = 0.02, free_spot: bool = False
    ) -> tuple[tuple[float, float], float | None]:
        """cube_0 goes back to its exact episode-start pose (the stock Sweep plans from
        it); the others to the first free spot on a 1.5 cm grid inside the pile region,
        >= 4.5 cm from every cube already on the counter so the fingers fit."""
        if cube == S.CUBES[0] and not free_spot:
            pos, _ = self.session.initial_pose(name=cube)
            quat = self.session.initial_pose(name=cube)[1]
            return (float(pos[0]), float(pos[1])), CubeHeading.of(quaternion=quat)
        taken = [
            self.session.position(name=c)[:2]
            for c in S.CUBES
            if c != cube and self.session.location(cube=c) == "counter"
        ]
        taken.append(self.session.initial_pose(name=S.CUBES[0])[0][:2])
        m = margin
        best = None
        for x in np.arange(S.PILE_X[1] - m, S.PILE_X[0] + m - 1e-9, -0.015):
            for y in np.arange(S.PILE_Y[1] - m, S.PILE_Y[0] + m - 1e-9, -0.015):
                gap = min((float(np.hypot(x - t[0], y - t[1])) for t in taken), default=1.0)
                if gap >= spacing:
                    return (float(x), float(y)), None
                if best is None or gap > best[0]:
                    best = (gap, (float(x), float(y)))
        assert best is not None
        return best[1], None

    def _recover_hand(self) -> None:
        if self.session.gripper() > 0.05:
            self._motion.set_gripper(command=0.0)
        self._motion.go_home(grip=0.0)

    # ------------------------------------------------------------------ phases
    def clear_drawer(self) -> bool:
        """Wiggle first, then pick what a grasp fits; when nothing does, nudge a cube
        clear, and only then fall back on a row grasp or an open-finger push."""
        failures: dict[str, int] = {}
        pushes = nudges = wiggles = 0
        for _ in range(self.max_drawer_rounds):
            todo = [c for c in S.CUBES if self.session.location(cube=c) == "drawer"]
            if not todo:
                return True
            if self.session.drawer_pos() < self.open_to - 0.04:
                # contact with the cubes drags the free-sliding drawer shut; pull it back out
                self._do(name="reopen_drawer", fn=self._prims.move_drawer, target=self.open_to)
            self._examine(cubes=todo)
            nearest = min(self._rep.wall_gaps().values(), default=1.0)
            shake = "wiggle" not in self.without and wiggles < self.max_wiggles
            if shake and nearest < self._rep.DRAWER_WALL_CLEAR:
                wiggles += 1
                self._reposition(
                    name="wiggle_drawer", fn=self._rep.wiggle_drawer, strategy="wiggle"
                )
                continue
            if self._pick_best(todo=todo, failures=failures):
                continue
            nudge = "nudge" not in self.without and nudges < self.max_nudges
            if nudge and self._nudge_best(todo=todo):
                nudges += 1
                continue
            if self._try_pairs(todo=todo, failures=failures):
                continue
            if pushes < self.max_pushes and self._push_best(todo=todo):
                pushes += 1
                continue
            return False
        return not any(self.session.location(cube=c) == "drawer" for c in S.CUBES)

    def _pick_best(self, *, todo: list[str], failures: dict[str, int]) -> bool:
        ranked = sorted(
            todo, key=lambda c: (failures.get(c, 0), -len(self._prims.grasp_candidates(cube=c)))
        )
        for cube in ranked:
            if failures.get(cube, 0) >= 3 or not self._prims.graspable(cube=cube):
                continue
            if self._stuck(cube=cube):
                continue
            if self._pick_and_place(cube=cube):
                return True
            failures[cube] = failures.get(cube, 0) + 1
        return False

    def _nudge_best(self, *, todo: list[str]) -> bool:
        scored = [(self._rep.nudge_candidates(cube=c), c) for c in todo]
        for plans, cube in sorted(scored, key=lambda t: -(t[0][0].score if t[0] else -99.0)):
            if not plans:
                continue
            moved = self._reposition(
                name=f"nudge_{cube}", fn=self._rep.nudge, strategy="nudge", cube=cube
            )
            if moved is not None:
                return True
        return False

    def _push_best(self, *, todo: list[str]) -> bool:
        scored = [(self._prims.push_candidates(cube=c), c) for c in todo]
        for options, cube in sorted(scored, key=lambda t: -(t[0][0][0] if t[0] else -99.0)):
            if not options:
                continue
            pushed = self._reposition(
                name=f"push_{cube}", fn=self._prims.push, strategy="push", cube=cube
            )
            if pushed is not None:
                return True
        return False

    def _try_pairs(self, *, todo: list[str], failures: dict[str, int]) -> bool:
        """Lift two face-to-face neighbours in one grasp when no single cube is free."""
        for a in todo:
            for b in todo:
                if a == b:
                    continue
                key = f"{a}+{b}"
                if failures.get(key, 0) >= 2 or not self._prims.pair_candidates(cube=a, partner=b):
                    continue
                middle = self._prims.row_between(cube=a, partner=b) or ()
                row = "_".join((a, *middle, b))
                picked = self._do(name=f"pick_row_{row}", fn=self._prims.pick, cube=a, partner=b)
                if picked is None:
                    failures[key] = failures.get(key, 0) + 1
                    continue
                for held in (a, *middle, b):
                    self._grasp[held] = "row"
                pa, pb = self.session.position(name=a)[:2], self.session.position(name=b)[:2]
                (sx, sy), _ = self._target(cube=a, spacing=0.07, margin=0.035, free_spot=True)
                target = (float(sx + (pa[0] - pb[0]) / 2), float(sy + (pa[1] - pb[1]) / 2))
                placed = self._do(
                    name=f"place_row_{row}",
                    fn=self._prims.place,
                    cube=a,
                    target_xy=target,
                    ee_to_cube=picked[0],
                    release=0.0,
                )
                if placed is None:
                    failures[key] = failures.get(key, 0) + 1
                    self._recover_hand()
                return True
        return False

    def restore_cube0(self) -> None:
        """The stock Sweep plans from cube_0's pose: put it back where it started if a
        pair placement left it elsewhere."""
        c = S.CUBES[0]
        start = self.session.initial_pose(name=c)[0]
        if self.session.location(cube=c) != "counter":
            return
        if np.hypot(*(self.session.position(name=c)[:2] - start[:2])) < 0.01:
            return
        picked = self._do(name=f"pick_{c}", fn=self._prims.pick, cube=c)
        if picked is None:
            return
        target, yaw = self._target(cube=c)
        if (
            self._do(
                name=f"place_{c}",
                fn=self._prims.place,
                cube=c,
                target_xy=target,
                target_yaw=yaw,
                ee_to_cube=picked[0],
            )
            is None
        ):
            self._recover_hand()

    def clear_front(self) -> None:
        """Take the cubes off the drawer's face and handle while the drawer is out and
        before it is shaken: a slam would throw them."""
        for cube in [c for c in S.CUBES if self.session.rides_drawer_front(cube=c)]:
            self._examine(cubes=[cube])
            if not self._stuck(cube=cube):
                self._pick_and_place(cube=cube)

    def gather_loose(self, *, max_rounds: int = 24) -> bool:
        """Floor cubes, countertop cubes outside the pile region, and strays."""
        failures: dict[str, int] = {}
        nudged: dict[str, int] = {}
        pushes = 0
        shifted = False
        for _ in range(max_rounds):
            todo = [
                c
                for c in S.CUBES
                if self.session.location(cube=c) != "drawer"
                and not self.session.in_pile(cube=c)
                and failures.get(c, 0) < 3
            ]
            if not todo:
                break
            self._examine(cubes=todo)
            # pickable cubes first: taking one away can free the neighbour it boxed in
            todo.sort(
                key=lambda c: (
                    failures.get(c, 0),
                    self._stuck(cube=c),
                    not self._prims.graspable(cube=c),
                )
            )
            cube = todo[0]
            stuck = (
                self._stuck(cube=cube)
                or not self._prims.graspable(cube=cube)
                or failures.get(cube, 0) > 0
            )
            riding = self.session.rides_drawer_front(cube=cube)
            shift = "shift" not in self.without and not shifted
            if stuck and riding and shift and self.session.drawer_pos() < 0.06:
                # on the drawer's face a cube lies a centimetre from the countertop's edge:
                # slide the drawer out and it rides clear of it
                shifted = True
                moved = self._reposition(
                    name="shift_drawer",
                    fn=self._prims.move_drawer,
                    strategy="drawer shift",
                    target=self.FRONT_CLEAR,
                )
                if moved is not None:
                    for c in todo:
                        failures[c] = 0
                    continue
            on_floor = self.session.location(cube=cube) == "floor"
            nudge = "nudge" not in self.without and nudged.get(cube, 0) < 2
            if stuck and on_floor and nudge:
                nudged[cube] = nudged.get(cube, 0) + 1
                moved = self._reposition(
                    name=f"nudge_{cube}", fn=self._rep.nudge, strategy="nudge", cube=cube
                )
                if moved is not None:
                    # it lies somewhere new: what failed where it was says nothing now
                    failures[cube] = 0
                    continue
            if not self._prims.graspable(cube=cube):
                pushed = None
                if pushes < self.max_pushes:
                    pushes += 1
                    pushed = self._reposition(
                        name=f"push_{cube}", fn=self._prims.push, strategy="push", cube=cube
                    )
                if pushed is None:
                    failures[cube] = failures.get(cube, 0) + 1
                continue
            if self._stuck(cube=cube) or not self._pick_and_place(cube=cube):
                failures[cube] = failures.get(cube, 0) + 1
        return all(self.session.in_pile(cube=c) for c in S.CUBES)

    def close_drawer(self, *, attempts: int = 2) -> bool:
        """Shut the drawer and check that it is shut."""
        for _ in range(attempts):
            if self.session.drawer_pos() < DrawerStroke.CLOSED_TOLERANCE:
                return True
            self._do(name="close_drawer", fn=self._prims.move_drawer, target=0.0)
        return self.session.drawer_pos() < DrawerStroke.CLOSED_TOLERANCE

    def park(self) -> str:
        self._motion.go_home(grip=0.0)
        self._motion.drive_to(target=self.PARK, grip=0.0)
        return f"parked at ({self.PARK[0]}, {self.PARK[1]})"

    def run(self) -> ResetOutcome:
        t0, tick0 = time.perf_counter(), self.session.ticks
        self._actions = 0
        self._origin = {
            c: self.session.location(cube=c) for c in S.CUBES if not self.session.in_pile(cube=c)
        }
        self._prims.choose_wiper_parking_pose()
        wiper = self.session.position(name=S.WIPER)
        if wiper[2] < S.COUNTER_TOP - 0.02 and not WiperHold.in_hand(
            gripper=np.asarray(self._scene.ee_now().position), wiper=wiper
        ):
            self._do(name="recover_wiper", fn=self._prims.recover_wiper)
        if self.session.position(name=S.WIPER)[2] > 0.47 or self.session.gripper() > 0.2:
            self._do(name="park_wiper", fn=self._prims.park_wiper)
        if any(self.session.location(cube=c) == "drawer" for c in S.CUBES):
            if self.session.drawer_pos() < self.open_to - 0.02:
                self._do(name="open_drawer_wide", fn=self._prims.move_drawer, target=self.open_to)
            self.clear_front()
            self.clear_drawer()
        self.close_drawer()
        self.gather_loose()
        self.restore_cube0()
        # a pick beside the drawer can drag it back open: check once everything is put away
        self.close_drawer()
        self._do(name="park_base", fn=self.park)
        return self.outcome(wall_s=time.perf_counter() - t0, ticks=self.session.ticks - tick0)

    def outcome(self, *, wall_s: float, ticks: int) -> ResetOutcome:
        pos, q = self.session.position(name=S.WIPER), self.session.quaternion(name=S.WIPER)
        home_pos, home_q = self.session.wiper_parking_pose()
        from scipy.spatial.transform import Rotation

        dyaw = (
            Rotation.from_quat(q).as_euler("zyx")[0] - Rotation.from_quat(home_q).as_euler("zyx")[0]
        )
        retrievals = {
            c: Retrieval(
                origin=where,
                blocked=self._blocked.get(c, False),
                assists=tuple(self._assists.get(c, ())),
                grasp=Retrieval.grasp_of(cube=c, picked=self._grasp),
            )
            for c, where in self._origin.items()
            if self.session.in_pile(cube=c)
        }
        return ResetOutcome(
            locations=self.session.locations(),
            in_pile={c: self.session.in_pile(cube=c) for c in S.CUBES},
            drawer_pos=round(self.session.drawer_pos(), 4),
            wiper_xy_error=round(float(np.hypot(*(pos[:2] - home_pos[:2]))), 4),
            wiper_yaw_error_deg=round(
                float(np.degrees(abs((dyaw + np.pi) % (2 * np.pi) - np.pi))), 2
            ),
            wiper_on_counter=bool(S.COUNTER_TOP - 0.01 < pos[2] < S.COUNTER_TOP + 0.02),
            robot_actions=self._actions,
            ticks=ticks,
            wall_s=round(wall_s, 1),
            retrievals=retrievals,
        )


class Unmoved:
    """Whether what a pick depends on is where it was when the planner refused it."""

    # Resting cubes jitter by tenths of a millimetre from tick to tick.
    TOLERANCE: ClassVar[float] = 0.002

    @staticmethod
    def since(*, before: np.ndarray | None, now: np.ndarray) -> bool:
        if before is None or before.shape != now.shape:
            return False
        return bool(np.max(np.abs(before - now)) < Unmoved.TOLERANCE)
