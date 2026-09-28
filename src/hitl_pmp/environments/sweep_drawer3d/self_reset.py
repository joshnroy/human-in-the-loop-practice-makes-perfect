"""The robot's own reset after a SweepIntoDrawer3D attempt.

Order: park the wiper where it started (OpenDrawer and Sweep are anchored on the wiper's
and cube_0's poses, so "somewhere re-pickable" is not enough); pull the drawer out to
0.25 m so the cubes clear the countertop; pick every drawer cube, pushing apart any that
are boxed in; close the drawer; then fetch floor cubes and strays; park the base.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .motion import ExecutionError, Motion
from .planning_scene import PlanningScene
from .primitives import Primitives
from .session import SweepDrawerSession
from .types import ResetOutcome, SweepDrawerScene

S = SweepDrawerScene


class SweepDrawerSelfReset(BaseModel):
    """Scripted, planning-backed self-reset for one session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    open_to: float = 0.25
    max_drawer_rounds: int = 20
    max_pushes: int = 10
    max_slams: int = 6

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

    _scene: Any = PrivateAttr(default=None)
    _motion: Any = PrivateAttr(default=None)
    _prims: Any = PrivateAttr(default=None)
    _actions: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        self._scene = PlanningScene(session=self.session)
        self._motion = Motion(session=self.session, scene=self._scene)
        self._prims = Primitives(session=self.session, scene=self._scene, motion=self._motion)
        self._actions = 0

    @property
    def primitives(self) -> Primitives:
        return self._prims

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

    def _target(
        self, *, cube: str, spacing: float = 0.045, margin: float = 0.02, free_spot: bool = False
    ) -> tuple[tuple[float, float], float | None]:
        """cube_0 goes back to its exact episode-start pose (the stock Sweep plans from
        it); the others to the first free spot on a 1.5 cm grid inside the pile region,
        >= 4.5 cm from every cube already on the counter so the fingers fit."""
        if cube == S.CUBES[0] and not free_spot:
            pos, _ = self.session.initial_pose(name=cube)
            quat = self.session.initial_pose(name=cube)[1]
            from scipy.spatial.transform import Rotation

            return (float(pos[0]), float(pos[1])), float(
                Rotation.from_quat(quat).as_euler("zyx")[0]
            )
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
    def _wall_bound(self, *, todo: list[str]) -> bool:
        """Any cube close enough to the front wall that the palm cannot come down by it."""
        wall = S.DRAWER_FRONT_INNER_X + self.session.drawer_pos()
        return any(wall - self.session.position(name=c)[0] - S.CUBE_HALF < 0.045 for c in todo)

    def clear_drawer(self) -> bool:
        failures: dict[str, int] = {}
        pushes = 0
        slams = 0
        for _ in range(self.max_drawer_rounds):
            todo = [c for c in S.CUBES if self.session.location(cube=c) == "drawer"]
            if not todo:
                return True
            if self.session.drawer_pos() < self.open_to - 0.04:
                # contact with the cubes drags the free-sliding drawer shut; pull it back out
                self._do(name="reopen_drawer", fn=self._prims.move_drawer, target=self.open_to)
            ranked = sorted(
                todo, key=lambda c: (failures.get(c, 0), -len(self._prims.grasp_candidates(cube=c)))
            )
            progressed = False
            for cube in ranked:
                if failures.get(cube, 0) >= 3 or not self._prims.graspable(cube=cube):
                    continue
                picked = self._do(name=f"pick_{cube}", fn=self._prims.pick, cube=cube)
                if picked is None:
                    failures[cube] = failures.get(cube, 0) + 1
                    continue
                if (
                    self._do(
                        name=f"place_{cube}",
                        fn=self._prims.place,
                        cube=cube,
                        target_xy=self._target(cube=cube)[0],
                        target_yaw=self._target(cube=cube)[1],
                        ee_to_cube=picked[0],
                    )
                    is None
                ):
                    failures[cube] = failures.get(cube, 0) + 1
                    self._recover_hand()
                progressed = True
                break
            if not progressed and slams < self.max_slams and self._wall_bound(todo=todo):
                # cubes pressed to the front wall: slide them off it before anything else
                if self._do(name="slam_drawer", fn=self._prims.slam_drawer, strokes=2) is not None:
                    slams += 2
                    progressed = True
                else:
                    slams = self.max_slams
            if not progressed:
                progressed = self._try_pairs(todo=todo, failures=failures)
            if not progressed and pushes < self.max_pushes:
                scored = [(self._prims.push_candidates(cube=c), c) for c in todo]
                for options, cube in sorted(scored, key=lambda t: -(t[0][0][0] if t[0] else -99.0)):
                    if not options:
                        continue
                    if self._do(name=f"push_{cube}", fn=self._prims.push, cube=cube) is not None:
                        pushes += 1
                        progressed = True
                        break
            if not progressed and slams < self.max_slams:
                # nothing fits: a stroke also reshuffles a packed cluster, so try one more
                if self._do(name="slam_drawer", fn=self._prims.slam_drawer, strokes=1) is not None:
                    slams += 1
                    failures.clear()
                    progressed = True
                else:
                    slams = self.max_slams
            if not progressed:
                return False
        return not any(self.session.location(cube=c) == "drawer" for c in S.CUBES)

    def _try_pairs(self, *, todo: list[str], failures: dict[str, int]) -> bool:
        """Lift two face-to-face neighbours in one grasp when no single cube is free."""
        for a in todo:
            for b in todo:
                if a == b:
                    continue
                key = f"{a}+{b}"
                if failures.get(key, 0) >= 2 or not self._prims.pair_candidates(cube=a, partner=b):
                    continue
                row = "_".join((a, *(self._prims.row_between(cube=a, partner=b) or ()), b))
                picked = self._do(name=f"pick_row_{row}", fn=self._prims.pick, cube=a, partner=b)
                if picked is None:
                    failures[key] = failures.get(key, 0) + 1
                    continue
                pa, pb = self.session.position(name=a)[:2], self.session.position(name=b)[:2]
                (sx, sy), _ = self._target(cube=a, spacing=0.07, margin=0.035, free_spot=True)
                target = (float(sx + (pa[0] - pb[0]) / 2), float(sy + (pa[1] - pb[1]) / 2))
                placed = self._do(
                    name=f"place_row_{row}",
                    fn=self._prims.place,
                    cube=a,
                    target_xy=target,
                    ee_to_cube=picked[0],
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

    def gather_loose(self, *, max_rounds: int = 12) -> bool:
        """Floor cubes, countertop cubes outside the pile region, and strays."""
        failures: dict[str, int] = {}
        pushes = 0
        for _ in range(max_rounds):
            todo = [
                c
                for c in S.CUBES
                if self.session.location(cube=c) != "drawer"
                and not self.session.in_pile(cube=c)
                and failures.get(c, 0) < 2
            ]
            if not todo:
                return True
            # graspable cubes first; a boxed-in one gets pushed apart instead of picked
            todo.sort(key=lambda c: not self._prims.graspable(cube=c))
            cube = todo[0]
            if not self._prims.graspable(cube=cube) and pushes < self.max_pushes:
                pushes += 1
                if self._do(name=f"push_{cube}", fn=self._prims.push, cube=cube) is None:
                    failures[cube] = failures.get(cube, 0) + 1
                continue
            picked = self._do(name=f"pick_{cube}", fn=self._prims.pick, cube=cube)
            if picked is None:
                failures[cube] = failures.get(cube, 0) + 1
                continue
            if (
                self._do(
                    name=f"place_{cube}",
                    fn=self._prims.place,
                    cube=cube,
                    target_xy=self._target(cube=cube)[0],
                    target_yaw=self._target(cube=cube)[1],
                    ee_to_cube=picked[0],
                )
                is None
            ):
                failures[cube] = failures.get(cube, 0) + 1
                self._recover_hand()
        return False

    def park(self) -> str:
        self._motion.go_home(grip=0.0)
        self._motion.drive_to(target=self.PARK, grip=0.0)
        return f"parked at ({self.PARK[0]}, {self.PARK[1]})"

    def _on_face_ledge(self) -> list[str]:
        out = []
        for c in S.CUBES:
            x, y, z = self.session.position(name=c)
            front = S.DRAWER_FACE_X + self.session.drawer_pos()
            if (
                0.43 < z < 0.47
                and front - 0.03 < x < front + 0.005
                and abs(y) < S.DRAWER_HALF_WIDTH
            ):
                out.append(c)
        return out

    def run(self) -> ResetOutcome:
        t0, tick0 = time.perf_counter(), self.session.ticks
        self._actions = 0
        if self.session.position(name=S.WIPER)[2] > 0.47 or self.session.gripper() > 0.2:
            self._do(name="park_wiper", fn=self._prims.park_wiper)
        if any(self.session.location(cube=c) == "drawer" for c in S.CUBES):
            if self.session.drawer_pos() < self.open_to - 0.02:
                self._do(name="open_drawer_wide", fn=self._prims.move_drawer, target=self.open_to)
            self.clear_drawer()
        if self.session.drawer_pos() > 0.01:
            self._do(name="close_drawer", fn=self._prims.move_drawer, target=0.0)
        if self._on_face_ledge() and self.session.drawer_pos() < 0.05:
            # a cube wedged on the closed drawer's face top, against the countertop edge:
            # open the drawer and it rides out on the face, clear of the counter
            self._do(name="open_drawer_ledge", fn=self._prims.move_drawer, target=0.12)
        self.gather_loose()
        if self.session.drawer_pos() > 0.01:
            self._do(name="close_drawer", fn=self._prims.move_drawer, target=0.0)
        self.restore_cube0()
        self._do(name="park_base", fn=self.park)
        return self.outcome(wall_s=time.perf_counter() - t0, ticks=self.session.ticks - tick0)

    def outcome(self, *, wall_s: float, ticks: int) -> ResetOutcome:
        pos, q = self.session.position(name=S.WIPER), self.session.quaternion(name=S.WIPER)
        home_pos, home_q = self.session.initial_pose(name=S.WIPER)
        from scipy.spatial.transform import Rotation

        dyaw = (
            Rotation.from_quat(q).as_euler("zyx")[0] - Rotation.from_quat(home_q).as_euler("zyx")[0]
        )
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
        )
