"""Fixed-tilt horizontal translation across the native world-+y pour direction."""

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from .native_controller import NativePouringController, PourParameters


class LateralPourParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tilt_angle: float = Field(default=103, ge=0, le=150, allow_inf_nan=False)
    wiggle_seconds: float = Field(default=12, ge=0, le=20, allow_inf_nan=False)


class LateralPouringController(NativePouringController):
    AMPLITUDE_M: ClassVar[float] = 0.015
    FREQUENCY_HZ: ClassVar[float] = 1.0
    RAMP_SECONDS: ClassVar[float] = 0.5

    active_lateral: LateralPourParameters | None = None
    lateral_trace: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def command_position(cls, *, center: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
        ramp = max(
            0.0, min(1.0, elapsed / cls.RAMP_SECONDS, (duration - elapsed) / cls.RAMP_SECONDS)
        )
        envelope = np.sin(np.pi * ramp / 2) ** 2
        offset = cls.AMPLITUDE_M * envelope * np.sin(2 * np.pi * cls.FREQUENCY_HZ * elapsed)
        # The native pour points along world +y. World x is perpendicular to
        # that direction and remains horizontal even when the pitcher is tilted.
        return center + np.array([offset, 0.0, 0.0])

    def pause(self, *, phase: str, seconds: float, grip: float | None = None) -> None:
        p = self.active_lateral
        if phase != "hold pour" or p is None:
            super().pause(phase=phase, seconds=seconds, grip=grip)
            return
        steps = round(p.wiggle_seconds * self.frequency)
        if not steps:
            return
        super().pause(phase="stabilize fixed tilt", seconds=1)
        duration = steps / self.frequency
        tray, _ = self.body_pose(name="tray")
        center = tray + [0, 0, 0.45]
        orientation = (
            Rotation.from_rotvec(np.array([-1.0, 0, 0]) * np.deg2rad(p.tilt_angle))
            * Rotation.from_euler("z", np.pi / 2)
            * self.rotation_offset
        )
        self.phase = "horizontal sideways wiggle"
        for index in range(1, steps + 1):
            elapsed = index / self.frequency
            position = self.command_position(center=center, elapsed=elapsed, duration=duration)
            q = self.inverse_kinematics(position=position, rotation=orientation)
            self.motor_step(target_q=q, grip=1)
            actual_hand, _ = self.hand_pose()
            pitcher, _ = self.body_pose(name="pitcher")
            self.lateral_trace.append({
                "time": float(self.data.time),
                "elapsed": elapsed,
                "commanded_position": position.tolist(),
                "commanded_tilt_degrees": p.tilt_angle,
                "actual_hand_position": actual_hand.tolist(),
                "pitcher_position": pitcher.tolist(),
                "locations": self.scene.cube_locations(),
            })

    def run_lateral_pour(self, *, parameters: LateralPourParameters) -> dict[str, Any]:
        if self.held != "pitcher":
            raise ValueError("Pour requires a retained handle grasp")
        before = self.scene.physical_snapshot()
        self.active_lateral = parameters
        self.lateral_trace = []
        error = None
        try:
            # The placeholder dwell is replaced by pause() above. The longer
            # sideways stage is governed solely by the two-parameter model.
            super().pour(
                parameters=PourParameters(
                    angle=parameters.tilt_angle,
                    offset_x=0,
                    offset_y=0,
                    tilt_seconds=10,
                    hold_seconds=0.2,
                )
            )
            for _ in range(8):
                if self.scene.cubes_settled():
                    break
                super().pause(phase="wait for cubes to settle", seconds=1)
        except ValueError as exc:
            error = str(exc)
        finally:
            self.active_lateral = None
        after = self.scene.physical_snapshot()
        event = {
            "skill": "Pour",
            "controller": "horizontal_lateral_wiggle_v1",
            "parameters": parameters.model_dump(),
            "success": error is None and after["goal"],
            "error": error,
            "before": before,
            "after": after,
            "seconds": after["time"] - before["time"],
            "wiggle_trace": list(self.lateral_trace),
            "fixed_settings": {
                "aim_offset_m": [0, 0],
                "amplitude_m": self.AMPLITUDE_M,
                "frequency_hz": self.FREQUENCY_HZ,
                "pour_direction_world": [0, 1, 0],
                "wiggle_axis_world": [1, 0, 0],
                "orientation": "held constant throughout the sideways stage",
                "outward_seconds": 10,
                "return_seconds": 10,
            },
        }
        self.events.append(event)
        return event
