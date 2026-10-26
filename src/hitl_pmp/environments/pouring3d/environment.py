"""Project Environment adapter for the pouring prototype's macro actions.

The observation retains full qpos/qvel, mocap poses and equality activation.
This is a diagnostic state vector, not a claim of a validated Markov abstraction.
Human action 5 is explicit and costed; it must be routed through HumanOracle
before integration with the project's practice planners.
"""

from typing import Any, ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import ConfigDict, Field

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type


class PouringEnvironment(Environment):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 0, -0.4, -0.4, 0.5, 0.2], dtype=np.float32),
        high=np.array([7, 4, 150, 0.4, 0.4, 4, 4], dtype=np.float32),
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
    seed: int = 0
    sim: Any = None
    world: Object | None = None
    last_info: dict[str, Any] = Field(default_factory=dict)

    def hard_reset(self) -> None:
        from .simulation import PouringSimulation

        if self.sim is not None:
            self.sim.close()
        self.sim = PouringSimulation.create(seed=self.seed)
        arrays = self.state_arrays()
        features = tuple(
            f"{name}_{i}" for name, values in arrays.items() for i in range(len(values))
        )
        self.world = Object(name="world", type=Type(name="pouring_world", feature_names=features))
        self.current_state = self.observe()
        self.last_info = {}

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
            "mocap_pos": self.sim.data.mocap_pos.ravel().copy(),
            "mocap_quat": self.sim.data.mocap_quat.ravel().copy(),
            "equality": self.sim.data.eq_active.astype(float).copy(),
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
        return State(data={self.world: np.concatenate(list(self.state_arrays().values()))})

    def take_action(self, *, action: Action) -> State:
        values = np.asarray(action, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite seven-component macro action")
        if np.any(values < self.action_space.low) or np.any(values > self.action_space.high):
            raise ValueError("Action outside declared bounds")
        if values[0] != int(values[0]) or values[1] != int(values[1]):
            raise ValueError("Skill and cube IDs must be integers")
        skill = self.skill_names[int(values[0])]
        if skill == "Wait":
            self.sim.elapsed_seconds += 1.0
            self.last_info = {
                "skill": "Wait",
                "success": True,
                "cost_seconds": 1.0,
                "human_interventions": 0,
            }
        else:
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
            for cube in range(5) if skill_id in {3, 4, 6} else [0]:
                try:
                    self.sim.check_preconditions(skill=skill, cube=cube, parameters={})
                except ValueError:
                    continue
                candidates.append(np.array([skill_id, cube, 130, 0, -0.055, 2, 2], dtype=float))
        return candidates

    def noop_action(self) -> Action:
        return np.array([7, 0, 130, 0, -0.055, 2, 2], dtype=float)

    def set_state(self, *, state: State) -> None:
        del state
        raise NotImplementedError("Arbitrary state restoration is not a physical pouring reset")

    def cost_observation(self) -> Any:
        return {
            "locations": self.sim.locations(),
            "held": self.sim.held,
            "elapsed_seconds": self.sim.elapsed_seconds,
        }
