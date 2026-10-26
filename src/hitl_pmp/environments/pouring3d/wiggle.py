"""Two-parameter, centered tilt-then-wiggle prototype using native motors."""

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from hitl_pmp.core.problem.environment.types import Action, State

from .environment import PouringEnvironment
from .native_controller import NativePouringController, PourParameters


class TiltWiggleParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Reserve four degrees of headroom below the native 150-degree command cap.
    tilt_angle: float = Field(default=108, ge=0, le=146, allow_inf_nan=False)
    wiggle_seconds: float = Field(default=2, ge=0, le=4, allow_inf_nan=False)


class TiltWiggleController(NativePouringController):
    WIGGLE_DEGREES: ClassVar[float] = 4.0
    WIGGLE_HZ: ClassVar[float] = 1.0
    TILT_SECONDS: ClassVar[float] = 10.0
    MAX_SETTLE_SECONDS: ClassVar[int] = 10

    active_wiggle: TiltWiggleParameters | None = None
    wiggle_trace: list[dict[str, float]] = Field(default_factory=list)

    @classmethod
    def fixed_settings(cls) -> dict[str, Any]:
        return {
            "aim_offset_m": [0, 0],
            "wiggle_amplitude_degrees": cls.WIGGLE_DEGREES,
            "wiggle_hz": cls.WIGGLE_HZ,
            "axis": "fore-aft, same axis as native pour",
            "outward_seconds": cls.TILT_SECONDS,
            "return_seconds": cls.TILT_SECONDS,
            "settle_seconds_min": 2,
            "settle_seconds_max": cls.MAX_SETTLE_SECONDS,
            "envelope": "sin(pi*t/T)^2; start and finish at the base tilt",
        }

    def pause(self, *, phase: str, seconds: float, grip: float | None = None) -> None:
        parameters = self.active_wiggle
        if phase != "hold pour" or parameters is None:
            super().pause(phase=phase, seconds=seconds, grip=grip)
            return
        # Replace only the dwell at maximum tilt. Pickup, transport, tilt, and
        # return use the existing native controller without any pose assignment.
        count = round(parameters.wiggle_seconds * self.frequency)
        if count == 0:
            return
        duration = count / self.frequency
        tray, _ = self.body_pose(name="tray")
        target = tray + [0, 0, 0.45]
        upright = Rotation.from_euler("z", np.pi / 2)
        self.phase = "wiggle pitcher"
        for index in range(1, count + 1):
            elapsed = index / self.frequency
            envelope = np.sin(np.pi * elapsed / duration) ** 2
            offset = float(
                self.WIGGLE_DEGREES * envelope * np.sin(2 * np.pi * self.WIGGLE_HZ * elapsed)
            )
            angle = parameters.tilt_angle + offset
            orientation = Rotation.from_rotvec(np.array([-1.0, 0, 0]) * np.deg2rad(angle)) * upright
            q = self.inverse_kinematics(
                position=target, rotation=orientation * self.rotation_offset
            )
            self.motor_step(target_q=q, grip=1)
            _, rotation = self.body_pose(name="pitcher")
            actual_tilt = float(np.rad2deg(np.arccos(np.clip(rotation.apply([0, 0, 1])[2], -1, 1))))
            self.wiggle_trace.append({
                "time": float(self.data.time),
                "offset_degrees": offset,
                "commanded_tilt_degrees": angle,
                "actual_tilt_degrees": actual_tilt,
            })

    def run_tilt_wiggle(self, *, parameters: TiltWiggleParameters) -> dict[str, Any]:
        if self.held != "pitcher":
            raise ValueError("Pour requires a retained pitcher grasp")
        before = self.scene.physical_snapshot()
        error = None
        self.wiggle_trace = []
        self.active_wiggle = parameters
        try:
            super().pour(
                parameters=PourParameters(
                    offset_x=0,
                    offset_y=0,
                    angle=parameters.tilt_angle,
                    tilt_seconds=self.TILT_SECONDS,
                    hold_seconds=max(0.2, parameters.wiggle_seconds),
                )
            )
            # Settling is a completion condition, independent of the requested
            # count. Never stop the wiggle or return early when a count is seen.
            for _ in range(self.MAX_SETTLE_SECONDS - 2):
                if self.scene.cubes_settled():
                    break
                self.pause(phase="wait for cubes to settle", seconds=1)
        except ValueError as exc:
            error = str(exc)
        finally:
            self.active_wiggle = None
        after = self.scene.physical_snapshot()
        event = {
            "skill": "Pour",
            "controller": "tilt_then_wiggle_v1",
            "success": error is None and after["goal"],
            "error": error,
            "parameters": parameters.model_dump(),
            "fixed_settings": self.fixed_settings(),
            "seconds": after["time"] - before["time"],
            "before": before,
            "after": after,
            "wiggle_trace": list(self.wiggle_trace),
        }
        self.events.append(event)
        print(
            f"TiltThenWiggle: success={event['success']}; locations={after['locations']}; "
            f"error={error}",
            flush=True,
        )
        return event


class TiltWigglePouringEnvironment(PouringEnvironment):
    """Macro action: [skill ID, cube ID, tilt angle, wiggle seconds]."""

    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 0, 0], dtype=np.float32),
        high=np.array([7, 4, 146, 4], dtype=np.float32),
    )
    controller_class: ClassVar[type[TiltWiggleController]] = TiltWiggleController

    def hard_reset(self) -> None:
        super().hard_reset()
        self.sim.controller = self.controller_class.create(env=self.sim.env)
        self.current_state = self.observe()

    def take_action(self, *, action: Action) -> State:
        values = np.asarray(action, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite four-component macro action")
        if np.any(values < self.action_space.low) or np.any(values > self.action_space.high):
            raise ValueError("Action outside declared bounds")
        if values[0] != int(values[0]) or values[1] != int(values[1]):
            raise ValueError("Skill and cube IDs must be integers")
        skill = self.skill_names[int(values[0])]
        if skill == "HumanPourBack":
            raise ValueError("Human reset uses HumanOracle.execute_movables_reset")
        if skill == "Pour":
            parameters = TiltWiggleParameters(tilt_angle=values[2], wiggle_seconds=values[3])
            controller = self.sim.controller
            assert isinstance(controller, TiltWiggleController)
            self.last_info = controller.run_tilt_wiggle(parameters=parameters)
            self.last_info["cost_seconds"] = self.last_info["seconds"]
            self.last_info["human_interventions"] = 0
            self.sim.elapsed_seconds += self.last_info["seconds"]
            self.sim.events.append(self.last_info)
        else:
            self.last_info = self.sim.execute(skill=skill, cube=int(values[1]))
        self.current_state = self.observe()
        return self.current_state

    def noop_action(self) -> Action:
        return np.array([7, 0, 108, 2], dtype=float)

    def get_valid_actions(self) -> list[Action]:
        return [
            np.array([action[0], action[1], 108, 2], dtype=float)
            for action in super().get_valid_actions()
        ]
