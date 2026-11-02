"""Instance-local fast arm feedback for a held table wiper."""

from collections.abc import Callable
from typing import Any

import numpy as np


class SweepServo:
    """Refresh arm feedback each physics tick without advancing or changing state.

    Enter only while the wiper is held. Its head offset is captured relative to
    the pinch site's robot body on entry, then follows that body rigidly rather
    than the wiper freejoint. The caller owns contact detection and edge safety.

    ``normal_force_n`` starts at zero and clamps finite requests to the configured
    nonnegative limit. Change it between ShelfSkills steps. ``servo_updates``
    counts successful callback updates; ``clipped_torque_count`` counts individual
    joint commands clipped after adding preload, not clipping inside the original
    controller. Both counters accumulate across uses of this context manager.
    """

    def __init__(
        self,
        *,
        shelf: Any,
        wiper: Any,
        max_normal_force_n: float = 15.0,
        control_update: Callable[[], None] | None = None,
    ) -> None:
        maximum = float(max_normal_force_n)
        if not np.isfinite(maximum) or maximum < 0:
            raise ValueError("max_normal_force_n must be finite and nonnegative")
        self._shelf = shelf
        self._wiper = wiper
        self._robot = shelf.core._robot_env
        self._max_normal_force_n = maximum
        self._normal_force_n = 0.0
        self._control_update = control_update
        self.servo_updates = 0
        self.clipped_torque_count = 0
        self._original_update: Callable[..., Any] | None = None
        self._compute_arm_torques: Callable[..., Any] | None = None
        self._latched_tick: int | None = None
        self._target: np.ndarray | None = None
        self._head_body = 0
        self._head_offset = np.zeros(3)
        self._jacobian = np.zeros((3, shelf.model.nv))

    @property
    def max_normal_force_n(self) -> float:
        """Maximum allowed downward preload, in newtons."""
        return self._max_normal_force_n

    normal_force_n = property(
        fget=lambda self: self._normal_force_n,
        fset=lambda self, value: self._set_normal_force_n(value=value),
        doc="Bounded downward preload requested by the outer controller, in newtons.",
    )

    def _set_normal_force_n(self, *, value: float) -> None:
        force = float(value)
        if not np.isfinite(force):
            raise ValueError("normal_force_n must be finite")
        self._normal_force_n = float(np.clip(force, 0, self._max_normal_force_n))

    def __enter__(self) -> "SweepServo":
        if self._original_update is not None:
            raise RuntimeError("SweepServo is already installed")
        shelf = self._shelf
        self._head_body = int(shelf.model.site_bodyid[shelf.site])
        rotation = shelf.data.xmat[self._head_body].reshape(3, 3)
        self._head_offset = rotation.T @ (
            shelf.data.geom_xpos[self._wiper.head_geom] - shelf.data.xpos[self._head_body]
        )
        self._latched_tick = None
        self._target = None
        self._compute_arm_torques = self._robot._compute_arm_torques
        self._original_update = self._robot._update_ctrl
        self._robot._update_ctrl = lambda action: self._update_ctrl(action=action)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._original_update is not None:
            self._robot._update_ctrl = self._original_update
            self._original_update = None
            self._compute_arm_torques = None

    def _update_ctrl(self, *, action: np.ndarray) -> None:
        shelf = self._shelf
        if self._control_update is not None:
            self._control_update()
        if self._target is None or self._latched_tick != shelf.ticks:
            current = shelf.data.qpos[shelf.qa].copy()
            error = (shelf.arm_reference - current + np.pi) % (2 * np.pi) - np.pi
            self._target = current + np.clip(error, -0.1, 0.1)
            self._latched_tick = shelf.ticks
        assert self._compute_arm_torques is not None
        assert self._original_update is not None
        torques = np.array(self._compute_arm_torques(target_positions=self._target), copy=True)
        if self._normal_force_n > 0:
            point = (
                shelf.data.xpos[self._head_body]
                + shelf.data.xmat[self._head_body].reshape(3, 3) @ self._head_offset
            )
            shelf.mujoco.mj_jac(
                shelf.model, shelf.data, self._jacobian, None, point, self._head_body
            )
            torques += self._jacobian[:, shelf.va].T @ np.array([0, 0, -self._normal_force_n])
        limits = self._robot.ARM_TORQUE_LIMITS
        bounded = np.clip(torques, -limits, limits)
        forwarded = np.array(action, copy=True)
        forwarded[3:10] = bounded
        self._original_update(forwarded)
        self.servo_updates += 1
        self.clipped_torque_count += int(np.count_nonzero(torques != bounded))
