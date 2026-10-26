"""Macro-action adapter around the native KinDER pouring task.

The diagnostic observation includes physical state and controller history.
Human actions are explicit and costed; practice-planner integration is separate.
"""

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import ConfigDict, Field

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type

from .objects import PouringObjects as O
from .types import PouringState


class PouringEnvironment(Environment):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 0, -0.4, -0.4, 2, 0.2], dtype=np.float32),
        high=np.array([7, 4, 150, 0.4, 0.4, 15, 8], dtype=np.float32),
    )
    skill_names: ClassVar[tuple[str, ...]] = (
        "PickPitcher",
        "Pour",
        "PlacePitcherAtHome",
        "PickCubeFromTray",
        "DropCubeIntoPitcher",
        "HumanPourBack",
        "PickCubeFromFloor",
        "Wait",
    )
    requested_count: int = Field(default=5, ge=1, le=5, strict=True)
    seed: int = 0
    scene_bg: bool = True
    sim: Any = None
    world: Object | None = None
    last_info: dict[str, Any] = Field(default_factory=dict)

    def hard_reset(self) -> None:
        from .simulation import PouringSimulation

        if self.sim is not None:
            self.sim.close()
        self.sim = PouringSimulation.create(
            seed=self.seed, scene_bg=self.scene_bg, requested_count=self.requested_count
        )
        arrays = self.state_arrays()
        features = tuple(
            f"{name}_{i}" for name, values in arrays.items() for i in range(len(values))
        )
        self.world = Object(name="world", type=Type(name="pouring_world", feature_names=features))
        self.current_state = self.observe()
        self.last_info = {}

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "requested_count": np.array([self.sim.controller.scene.requested_count]),
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
            "actuator_state": self.sim.data.act.copy(),
            "control": self.sim.data.ctrl.copy(),
            "controller_arm": self.sim.controller.last_q.copy(),
            "controller_base": self.sim.controller.base_target.copy(),
            "grasp_position": (
                self.sim.controller.hand_offset.copy()
                if self.sim.controller.hand_offset is not None
                else np.zeros(3)
            ),
            "grasp_rotation": (
                self.sim.controller.rotation_offset.as_quat()
                if self.sim.controller.rotation_offset is not None
                else np.array([0, 0, 0, 1])
            ),
            "controller_history": np.array([
                self.sim.controller.posture,
                self.sim.controller.posture_goal,
                self.sim.controller.lost_contacts,
            ]),
            "handling": np.array([
                0
                if self.sim.held is None
                else 1
                if self.sim.held == "pitcher"
                else 2 + int(self.sim.held[-1]),
                float(self.sim.gripper_closed),
            ]),
        }

    def observe(self) -> State:
        assert self.world is not None
        data = {self.world: np.concatenate(list(self.state_arrays().values()))}
        for obj in (*O.cubes, O.pitcher, O.tray):
            body = self.sim.model.body(obj.name).id
            data[obj] = np.r_[self.sim.data.xpos[body], self.sim.data.xquat[body]]
        data[O.robot] = self.sim.data.qpos[self.sim.controller.base_q].copy()
        return PouringState(data=data, abstract_atoms=self.sim.abstract_atoms())

    def take_action(self, *, action: Action) -> State:
        values = np.asarray(action, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite seven-component macro action")
        if np.any(values < self.action_space.low) or np.any(values > self.action_space.high):
            raise ValueError("Action outside declared bounds")
        if values[0] != int(values[0]) or values[1] != int(values[1]):
            raise ValueError("Skill and cube IDs must be integers")
        skill = self.skill_names[int(values[0])]
        if skill == "HumanPourBack":
            raise ValueError(
                "Human reset uses HumanOracle.execute_movables_reset, not a robot action"
            )
        self.last_info = self.sim.execute(
            skill=skill,
            cube=int(values[1]),
            parameters=dict(
                zip(
                    ["angle", "offset_x", "offset_y", "tilt_seconds", "hold_seconds"],
                    values[2:],
                    strict=True,
                )
            )
            if skill == "Pour"
            else None,
        )
        self.current_state = self.observe()
        return self.current_state

    def get_valid_actions(self) -> list[Action]:
        candidates = [self.noop_action()]
        for skill_id, skill in enumerate(self.skill_names[:-1]):
            if skill == "HumanPourBack":
                continue
            for cube in range(5) if skill_id in {3, 4, 6} else [0]:
                try:
                    self.sim.check_preconditions(skill=skill, cube=cube, parameters={})
                except ValueError:
                    continue
                candidates.append(np.array([skill_id, cube, 145, 0, 0, 10, 4], dtype=float))
        return candidates

    def noop_action(self) -> Action:
        return np.array([7, 0, 145, 0, 0, 10, 4], dtype=float)

    def set_state(self, *, state: State) -> None:
        del state
        raise NotImplementedError("Arbitrary state restoration is not a physical pouring reset")

    def reset_movables(self, *, destination: str | None = None) -> bool:
        """Human-only cube respawn, through the shared HumanOracle interface.

        As in SweepOffShelf3D, preserve the robot, vessels, simulation time, and
        accumulated robot costs. The harness/provider prices this intervention.
        """
        if destination not in {None, "reset_cubes"}:
            raise ValueError(f"Unknown human reset destination {destination!r}")
        self.last_info = self.sim.reset_payload()
        self.current_state = self.observe()
        return True

    def cost_observation(self) -> Any:
        return {
            "locations": self.sim.locations(),
            "held": self.sim.held,
            "elapsed_seconds": self.sim.elapsed_seconds,
        }
