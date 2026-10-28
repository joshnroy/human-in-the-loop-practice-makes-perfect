"""Persistent native pouring; only explicit human resets may respawn payload."""

from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
from gymnasium.spaces import Box
from pydantic import Field, PrivateAttr, model_validator

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, State

from .backend import PourBackend
from .skills import RESET_NAME, SKILL_NAMES
from .types import FrozenPourState, PourObjects


class FrozenPourEnvironment(Environment):
    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
        high=np.array([6, 4, 5, 130, 0.025, 60], dtype=np.float64),
        dtype=np.float64,
    )
    payload_count: Literal[3, 5] = 3
    requested_count: int = Field(default=2, ge=1, le=5, strict=True)
    surface_mu: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    canonical_seed: int = 0
    scene_bg: bool = True
    skill_step_limit: int = Field(default=2500, gt=0)
    last_info: dict[str, Any] = Field(default_factory=dict)
    _controller: Any = PrivateAttr(default=None)
    _seed: int = PrivateAttr(default=0)
    _steps: int = PrivateAttr(default=0)
    _skill_start: int = PrivateAttr(default=0)
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)
    _motion_sink: Callable[..., None] | None = PrivateAttr(default=None)
    _skill_sink: Callable[..., None] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_request(self) -> "FrozenPourEnvironment":
        if self.requested_count > self.payload_count:
            raise ValueError("Requested count exceeds payload")
        return self

    def objects(self) -> PourObjects:
        return PourObjects(count=self.payload_count)

    def controller(self) -> Any:
        if self._controller is None:
            self.hard_reset()
        return self._controller

    def get_current_state(self) -> State:
        self.controller()
        assert self.current_state is not None
        return self.current_state

    def observe(self) -> FrozenPourState:
        return PourBackend.observe(
            controller=self.controller(), objects=self.objects(), seed=self._seed, steps=self._steps
        )

    def hard_reset(self) -> None:
        self.reset_to_seed(seed=self.canonical_seed, requested=self.requested_count)

    def reset_to_seed(self, *, seed: int, requested: int) -> FrozenPourState:
        if not 1 <= requested <= self.payload_count:
            raise ValueError("Invalid task count")
        self.close()
        self._seed, self._steps, self.requested_count = seed, 0, requested
        self._controller = PourBackend.create(
            count=self.payload_count,
            requested=requested,
            mu=self.surface_mu,
            scene_bg=self.scene_bg,
            seed=seed,
        )
        self._controller.before_step = self.before_tick
        self._controller.after_step = self.after_tick
        self._controller.callback = self._motion_sink
        self.last_info = {}
        self.current_state = self.observe()
        return self.current_state

    def set_state(self, *, state: State) -> None:
        if (
            not isinstance(state, FrozenPourState)
            or state.get(obj=PourObjects.scene, feature_name="steps_taken") != 0
        ):
            raise ValueError("Only harness episode-initial states can be restored")
        self.reset_to_seed(
            seed=int(state.get(obj=PourObjects.scene, feature_name="seed")),
            requested=int(state.get(obj=PourObjects.scene, feature_name="requested_count")),
        )

    def before_tick(self) -> None:
        if self._before is not None:
            self._before()
        if self._controller.steps - self._skill_start >= self.skill_step_limit:
            raise ValueError("Per-skill control-step limit exceeded")

    def after_tick(self) -> None:
        if self._after is not None:
            self._after()

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before, self._after = before, after

    def set_motion_sink(self, *, sink: Callable[..., None] | None) -> None:
        self._motion_sink = sink
        if self._controller is not None:
            self._controller.callback = sink

    def set_skill_sink(self, *, sink: Callable[..., None] | None) -> None:
        self._skill_sink = sink

    def take_action(self, *, action: Action) -> State:
        c = self.controller()
        start = float(c.data.time)
        self._skill_start = c.steps
        self.last_info = {"action": np.asarray(action).tolist(), "error": None, "success": False}
        try:
            values = np.asarray(action, dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("Expected [skill, shrimp, request, tilt, amplitude, duration]")
            if any(values[i] != int(values[i]) for i in range(3)):
                raise ValueError("Discrete action fields must be integers")
            skill_id, index, requested = map(int, values[:3])
            if not 0 <= skill_id < len(SKILL_NAMES) or not 0 <= index < self.payload_count:
                raise ValueError("Invalid skill or shrimp index")
            if not 1 <= requested <= self.payload_count:
                raise ValueError("Invalid skill count")
            skill = SKILL_NAMES[skill_id]
            self.last_info["skill"] = skill
            if self._skill_sink is not None:
                self._skill_sink(
                    name=skill,
                    objects=[],
                    params=values[3:].tolist() if skill == "Pour" else [],
                    controller={"requested_count": requested, "shrimp_index": index},
                )
            if skill == "Pour":
                from hitl_pmp.environments.pouring3d.frozen_native import (
                    ExtendedFeedbackPourParameters,
                )

                if c.scene.cube_locations() != ["pitcher"] * self.payload_count:
                    raise ValueError("A fresh Pour attempt starts with all payload in pitcher")
                parameters = ExtendedFeedbackPourParameters(
                    tilt_degrees=values[3], amplitude_m=values[4], duration_seconds=values[5]
                )
                result = c.run_feedback_pour(parameters=parameters, requested_count=requested)
                self.last_info.update({k: v for k, v in result.items() if k != "feedback_trace"})
            elif skill == "PickPitcher":
                c.pick_pitcher()
                self.last_info["success"] = c.held == "pitcher"
            elif skill == "PlacePitcherAtHome":
                if c.held != "pitcher":
                    raise ValueError("Placement requires holding pitcher")
                c.place_pitcher()
                self.last_info["success"] = c.held is None and c.vessel_home(name="pitcher")
            elif skill.startswith("PickShrimp"):
                c.pick_shrimp(index=index, location="tray" if skill.endswith("Tray") else "floor")
                self.last_info["success"] = c.held == c.scene.cube_names[index]
            elif skill == "DropShrimpIntoPitcher":
                c.drop_shrimp(index=index)
                self.last_info["success"] = True
            else:
                c.pause(phase="wait", seconds=1)
                self.last_info["success"] = True
        except ValueError as exc:
            self.last_info.update(error=str(exc), success=False)
            if self.last_info.get("skill", "").startswith("PickShrimp") and c.held is None:
                try:
                    c.clear_failed_pick()
                except ValueError as retreat_error:
                    self.last_info["recovery_error"] = str(retreat_error)
        self._steps += 1
        self.last_info.update(
            seconds=float(c.data.time) - start,
            controller_steps=c.steps - self._skill_start,
            locations=c.scene.cube_locations(),
            held=c.held,
        )
        self.current_state = self.observe()
        return self.current_state

    def adopt_interrupted_skill(self, *, action: Action, reason: str) -> State:
        self._steps += 1
        self.last_info.update(
            action=np.asarray(action).tolist(),
            error=f"interrupted: {reason}",
            success=False,
            controller_steps=self.controller().steps - self._skill_start,
        )
        self.current_state = self.observe()
        return self.current_state

    def get_valid_actions(self) -> list[Action]:
        return []

    def noop_action(self) -> Action:
        return np.array([6, 0, self.requested_count, 0, 0, 0], dtype=float)

    def reset_movables(self, *, destination: str | None = None) -> bool:
        if destination not in {None, RESET_NAME}:
            raise ValueError(f"Unknown human reset {destination}")
        c = self.controller()
        if (
            c.held is not None
            or not c.vessel_home(name="pitcher")
            or not c.vessel_home(name="tray")
            or c.scene.cube_locations() == ["pitcher"] * self.payload_count
        ):
            return False
        c.scene.respawn_payload_in_pitcher()
        self._steps += 1
        self.last_info = {
            "skill": RESET_NAME,
            "success": c.ready(),
            "error": None,
            "human_interventions": 1,
            "seconds": 0,
            "controller_steps": 0,
        }
        self.current_state = self.observe()
        return bool(self.last_info["success"])

    def cost_observation(self) -> Any:
        return {
            "locations": self.controller().scene.cube_locations(),
            "held": self.controller().held,
            "elapsed_seconds": float(self.controller().data.time),
        }

    def close(self) -> None:
        if self._controller is not None:
            self._controller.env.close()
            self._controller = None
