"""Closed-loop execution of planned arm paths, base paths and gripper commands.

Every command stays inside the declared action space (|delta| <= 0.1 per step). Arm
commands are scaled uniformly, never clipped per joint: per-joint clipping bends the
executed motion off the planned (collision-checked) segment, which is how an early
version of this reset pushed the open drawer shut with its own forearm.
"""

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .planning_scene import ArmMath, PlanningScene
from .session import SweepDrawerSession
from .types import SweepDrawerScene


class ExecutionError(Exception):
    """A primitive could not be carried out; the message says why."""


class Angles:
    @staticmethod
    def wrap(*, angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


class Motion(BaseModel):
    """Executes motions for one session through its planning scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    scene: PlanningScene

    def _grip_hold(self) -> float:
        return 1.0 if self.session.gripper() > 0.2 else 0.0

    def hold(self, *, ticks: int, grip: float | None = None, arm: np.ndarray | None = None) -> None:
        q = self.session.arm() if arm is None else arm
        g = self._grip_hold() if grip is None else grip
        for _ in range(ticks):
            a = np.zeros(11)
            a[3:10] = np.clip(ArmMath.wrap(delta=q - self.session.arm()), -0.1, 0.1)
            a[-1] = g
            self.session.step(action=a)

    def set_gripper(
        self, *, command: float, max_ticks: int = 30, arm: np.ndarray | None = None
    ) -> float:
        """Command the gripper and wait until the fingers themselves stop moving (the
        state's `pos_gripper` echoes the command at once, so it cannot be waited on).
        Returns the finger closure (driver angle, 0 open .. ~0.8 closed)."""
        q = self.session.arm() if arm is None else arm
        last = None
        still = 0
        for i in range(max_ticks):
            a = np.zeros(11)
            a[3:10] = np.clip(ArmMath.wrap(delta=q - self.session.arm()), -0.1, 0.1)
            a[-1] = command
            self.session.step(action=a)
            f = self.session.finger_closure()
            still = still + 1 if last is not None and abs(f - last) < 0.002 else 0
            if still >= 2 and i >= 4:
                break
            last = f
        return self.session.finger_closure()

    def follow(
        self,
        *,
        path: Sequence[np.ndarray] | None,
        grip: float | None = None,
        tol: float = 0.03,
        final_tol: float = 0.01,
        max_ticks: int = 600,
        tick_guard: Callable[[], None] | None = None,
        stop_condition: Callable[[], bool] | None = None,
    ) -> bool:
        """Track waypoints; True at the final waypoint or an explicit physical goal.

        The optional condition is checked after the guard on each physical tick.
        None preserves final-joint-target termination.
        """
        if path is None:
            return False
        g = self._grip_hold() if grip is None else grip
        dense: list[np.ndarray] = []
        prev = self.session.arm()
        for w in path:
            w = prev + ArmMath.wrap(delta=np.asarray(w[:7], dtype=float) - prev)
            n = max(1, int(np.ceil(np.max(np.abs(w - prev)) / 0.05)))
            dense.extend(prev + (w - prev) * k / n for k in range(1, n + 1))
            prev = w
        if not dense:
            return True
        idx = 0
        for _ in range(max_ticks):
            q = self.session.arm()
            while idx < len(dense) - 1 and np.max(np.abs(ArmMath.wrap(delta=dense[idx] - q))) < tol:
                idx += 1
            err = ArmMath.wrap(delta=dense[idx] - q)
            if idx == len(dense) - 1 and np.max(np.abs(err)) < final_tol:
                return True
            cmd = err.copy()
            peak = float(np.max(np.abs(cmd)))
            if peak > 0.1:
                cmd *= 0.1 / peak
            a = np.zeros(11)
            a[3:10] = cmd
            a[-1] = g
            self.session.step(action=a)
            if tick_guard is not None:
                tick_guard()
            if stop_condition is not None and stop_condition():
                return True
        return False

    def drive(
        self,
        *,
        path: Sequence[tuple[float, float, float]],
        grip: float | None = None,
        max_ticks: int = 500,
        tol: float = 0.004,
        arm: np.ndarray | None = None,
        max_translation_step: float | None = None,
        max_yaw_step: float | None = None,
        translation_step_change: float | None = None,
        yaw_step_change: float | None = None,
        tick_guard: Callable[[], None] | None = None,
    ) -> bool:
        """Track a base path while holding the arm at its requested configuration.

        Optional translation/yaw caps are meters/radians commanded per control tick.
        Change caps bound consecutive commands, starting from zero; translation uses
        the Euclidean norm. None preserves the existing tracking commands. The guard
        runs immediately after each executed tick and may raise to stop execution.
        """
        from prpl_utils.utils import get_signed_angle_distance

        for limit in (max_translation_step, max_yaw_step, translation_step_change, yaw_step_change):
            if limit is not None and (not np.isfinite(limit) or limit <= 0):
                raise ValueError("Optional drive limits must be finite and positive")
        previous_translation = np.zeros(2)
        previous_yaw = 0.0
        g = self._grip_hold() if grip is None else grip
        q_hold = self.session.arm() if arm is None else arm.copy()
        remaining = list(path)
        for _ in range(max_ticks):
            x, y, th = self.session.base()
            th = Angles.wrap(angle=th)
            while len(remaining) > 1:
                nx, ny, nth = remaining[0]
                close = abs(get_signed_angle_distance(Angles.wrap(angle=nth), th)) < 0.06
                if np.hypot(nx - x, ny - y) < 0.04 and close:
                    remaining.pop(0)
                else:
                    break
            nx, ny, nth = remaining[0]
            dth = get_signed_angle_distance(Angles.wrap(angle=nth), th)
            if len(remaining) == 1 and np.hypot(nx - x, ny - y) < tol and abs(dth) < 0.01:
                return True
            a = np.zeros(11)
            a[0], a[1], a[2] = nx - x, ny - y, dth
            translation_norm = float(np.linalg.norm(a[:2]))
            if max_translation_step is not None and translation_norm > max_translation_step:
                a[:2] *= max_translation_step / translation_norm
            if max_yaw_step is not None:
                a[2] = np.clip(a[2], -max_yaw_step, max_yaw_step)
            if translation_step_change is not None:
                change = a[:2] - previous_translation
                change_norm = float(np.linalg.norm(change))
                if change_norm > translation_step_change:
                    change *= translation_step_change / change_norm
                a[:2] = previous_translation + change
            if yaw_step_change is not None:
                a[2] = previous_yaw + np.clip(
                    a[2] - previous_yaw, -yaw_step_change, yaw_step_change
                )
            a[3:10] = np.clip(ArmMath.wrap(delta=q_hold - self.session.arm()), -0.1, 0.1)
            a[-1] = g
            self.session.step(action=a)
            previous_translation = a[:2].copy()
            previous_yaw = float(a[2])
            if tick_guard is not None:
                tick_guard()
        return False

    def drive_straight(
        self,
        *,
        target: tuple[float, float, float],
        grip: float,
        arm: np.ndarray,
        speed: float = 0.012,
        max_ticks: int = 300,
    ) -> bool:
        """Straight-line base motion at bounded speed with the arm held at `arm` (used to
        pull or push the drawer with the handle in the hand)."""
        from prpl_utils.utils import get_signed_angle_distance

        for _ in range(max_ticks):
            x, y, th = self.session.base()
            d = np.array([target[0] - x, target[1] - y])
            dth = get_signed_angle_distance(Angles.wrap(angle=target[2]), Angles.wrap(angle=th))
            if np.linalg.norm(d) < 0.005 and abs(dth) < 0.02:
                return True
            n = float(np.linalg.norm(d))
            if n > speed:
                d *= speed / n
            a = np.zeros(11)
            a[0], a[1] = d
            a[2] = np.clip(dth, -0.05, 0.05)
            a[3:10] = np.clip(ArmMath.wrap(delta=arm - self.session.arm()), -0.1, 0.1)
            a[-1] = grip
            self.session.step(action=a)
        return False

    def go_home(
        self, *, grip: float | None = None, held: int | None = None, held_tf: object = None
    ) -> bool:
        """Plan (collision-aware) back to the retract configuration; if the planner finds
        nothing from here, move there directly."""
        home = np.asarray(SweepDrawerScene.HOME)
        bodies = self.scene.bodies(without=() if held is None else (held,))
        plan = self.scene.plan_arm(goal=home, bodies=bodies, held=held, held_tf=held_tf)
        return self.follow(path=plan if plan is not None else [home], grip=grip, final_tol=0.03)

    def drive_to(self, *, target: tuple[float, float, float], grip: float | None = None) -> None:
        path = self.scene.plan_base(target=target)
        if path is None:
            raise ExecutionError(
                f"no base path to ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
            )
        if not self.drive(path=path, grip=grip):
            raise ExecutionError("base did not converge")
