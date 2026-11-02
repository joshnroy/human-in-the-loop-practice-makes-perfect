"""Parameterized physical pouring with mouth-crossing feedback.

This prototype uses simulator object poses as ideal visual observations. Feedback
does not choose the tilt, amplitude, or time budget, and cannot retract a piece
that has escaped. All robot motion goes through native motors and handle contact.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from .shrimp_native import ShrimpPourController


class FeedbackPourParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tilt_degrees: float = Field(default=102, ge=60, le=130, allow_inf_nan=False)
    amplitude_m: float = Field(default=0.01, ge=0, le=0.025, allow_inf_nan=False)
    duration_seconds: float = Field(default=6, ge=0, le=20, allow_inf_nan=False)


class FeedbackPourController(ShrimpPourController):
    """Center the wrist over the tray, tilt, feed sideways, and cut off flow."""

    FEED_HZ: ClassVar[float] = 1.0
    START_RAMP_SECONDS: ClassVar[float] = 0.5
    TILT_SPEED_DEG_S: ClassVar[float] = 10.0
    RETURN_SPEED_DEG_S: ClassVar[float] = 45.0

    active_feedback: FeedbackPourParameters | None = None
    feedback_trace: list[dict[str, Any]] = Field(default_factory=list)
    outbound: set[str] = Field(default_factory=set)
    cutoff_reason: str | None = None

    @classmethod
    def offset(cls, *, elapsed: float, amplitude_m: float) -> float:
        # The start ramp is independent of the duration budget. A short budget
        # truncates this same waveform rather than shrinking its amplitude.
        ramp = min(1.0, max(0.0, elapsed / cls.START_RAMP_SECONDS))
        return float(
            amplitude_m * np.sin(np.pi * ramp / 2) ** 2 * np.sin(2 * np.pi * cls.FEED_HZ * elapsed)
        )

    def observe_mouth(self) -> int:
        origin, rotation = self.body_pose(name="pitcher")
        height = self.scene.get_object("pitcher").height
        rows = []
        for name, location in zip(self.scene.cube_names, self.scene.cube_locations(), strict=True):
            center, _ = self.body_pose(name=name)
            local = rotation.inv().apply(center - origin)
            # Actual open mouth plane, not the conservative cavity-core score.
            # Crossing uses body center; partial protrusion alone is not counted.
            if local[2] >= height or location == "tray":
                self.outbound.add(name)
            rows.append({"name": name, "local_center": local.tolist(), "location": location})
        self.feedback_trace.append({
            "time": float(self.data.time),
            "phase": self.phase,
            "outbound": sorted(self.outbound),
            "pieces": rows,
        })
        return len(self.outbound)

    def tracked_step(self, *, center: np.ndarray, angle: float, offset: float = 0) -> None:
        _, actual = self.body_pose(name="pitcher")
        _, hand = self.hand_pose()
        measured_grasp = actual.inv() * hand
        desired = Rotation.from_rotvec(
            np.array([-1.0, 0, 0]) * np.deg2rad(angle)
        ) * Rotation.from_euler("z", np.pi / 2)
        q = self.inverse_kinematics(
            position=center + [offset, 0, 0], rotation=desired * measured_grasp
        )
        self.motor_step(target_q=q, grip=1)

    def run_feedback_pour(
        self, *, parameters: FeedbackPourParameters, requested_count: int | None = None
    ) -> dict[str, Any]:
        if self.held != "pitcher":
            raise ValueError("Pour requires a retained handle grasp")
        requested = self.scene.requested_count if requested_count is None else requested_count
        if type(requested) is not int or not 1 <= requested <= len(self.scene.cube_names):
            raise ValueError("Invalid requested skill count")
        before = self.scene.physical_snapshot()
        self.active_feedback = parameters
        self.feedback_trace = []
        self.outbound = set()
        self.cutoff_reason = None
        error = None
        try:
            self.feedback_pour(parameters=parameters, requested_count=requested)
            for _ in range(10):
                if self.scene.cubes_settled():
                    break
                self.pause(phase="wait for shrimp to settle", seconds=1)
        except ValueError as exc:
            error = str(exc)
        finally:
            self.active_feedback = None
        after = self.scene.physical_snapshot()
        locations = self.scene.cube_locations()
        success = (
            error is None
            and locations.count("tray") == requested
            and locations.count("pitcher") == len(self.scene.cube_names) - requested
            and self.scene.cubes_settled()
        )
        event = {
            "skill": "Pour",
            "controller": "mouth_feedback_v1",
            "parameters": parameters.model_dump(),
            "success": success,
            "requested_count": requested,
            "error": error,
            "before": before,
            "after": after,
            "seconds": after["time"] - before["time"],
            "cutoff_reason": self.cutoff_reason,
            "feedback_trace": self.feedback_trace,
            "fixed_settings": {
                "frequency_hz": self.FEED_HZ,
                "aim_offset_m": [0, 0],
                "return_speed_deg_s": self.RETURN_SPEED_DEG_S,
                "observation": "ideal simulator body centers crossing mouth plane",
            },
        }
        self.events.append(event)
        return event

    def feedback_pour(
        self, *, parameters: FeedbackPourParameters, requested_count: int | None = None
    ) -> None:
        tray, _ = self.body_pose(name="tray")
        center = tray + [0, 0, 0.45]
        self.move_base(
            target=np.array([center[0] - 0.55, center[1], 0]),
            phase="position base for pour",
            seconds=5,
        )
        self.posture_goal = 1.0
        upright = Rotation.from_euler("z", np.pi / 2)
        self.move(
            phase="transport pour pivot above tray",
            position=center,
            rotation=upright * self.rotation_offset,
            seconds=5,
            grip=1,
        )
        requested = self.scene.requested_count if requested_count is None else requested_count
        angle = 0.0
        tilt_step = self.TILT_SPEED_DEG_S / self.frequency
        while angle < parameters.tilt_degrees:
            if self.observe_mouth() >= requested:
                self.cutoff_reason = "requested count crossed mouth during tilt"
                break
            angle = min(parameters.tilt_degrees, angle + tilt_step)
            self.phase = "feedback tilt pitcher"
            self.tracked_step(center=center, angle=angle)
        elapsed = 0.0
        offset = 0.0
        while self.cutoff_reason is None:
            if self.observe_mouth() >= requested:
                self.cutoff_reason = "requested count crossed mouth during feeding"
                break
            if elapsed + 0.5 / self.frequency >= parameters.duration_seconds:
                self.cutoff_reason = "feeding duration budget exhausted"
                break
            elapsed += 1 / self.frequency
            offset = self.offset(elapsed=elapsed, amplitude_m=parameters.amplitude_m)
            self.phase = "feedback horizontal sideways wiggle"
            self.tracked_step(center=center, angle=angle, offset=offset)
        # Same bounded physical cutoff for every parameter choice. Continue
        # observing but do not assume no further pieces can leave during return.
        start_offset = offset
        return_elapsed = 0.0
        while angle > 0:
            angle = max(0, angle - self.RETURN_SPEED_DEG_S / self.frequency)
            return_elapsed += 1 / self.frequency
            offset = start_offset * max(0, 1 - return_elapsed / 0.3)
            self.phase = "feedback return upright"
            self.tracked_step(center=center, angle=angle, offset=offset)
            self.observe_mouth()
        self.pause(phase="settle shrimp in tray", seconds=2)
