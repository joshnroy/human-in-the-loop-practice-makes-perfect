"""Four-component skill interface with the approved sideways Pour motion."""

from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.core.problem.environment.types import Action, State

from .environment import PouringEnvironment
from .lateral_wiggle import LateralPouringController, LateralPourParameters


class LateralPouringEnvironment(PouringEnvironment):
    """Action: [skill ID, cube ID, held tilt angle, sideways wiggle seconds]."""

    action_space: ClassVar[Box] = Box(
        low=np.array([0, 0, 0, 0], dtype=np.float32),
        high=np.array([7, 4, 150, 20], dtype=np.float32),
    )

    def hard_reset(self) -> None:
        super().hard_reset()
        self.sim.controller = LateralPouringController.create(env=self.sim.env)
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
            parameters = LateralPourParameters(tilt_angle=values[2], wiggle_seconds=values[3])
            controller = self.sim.controller
            assert isinstance(controller, LateralPouringController)
            self.last_info = controller.run_lateral_pour(parameters=parameters)
            self.last_info["cost_seconds"] = self.last_info["seconds"]
            self.last_info["human_interventions"] = 0
            self.sim.elapsed_seconds += self.last_info["seconds"]
            self.sim.events.append(self.last_info)
        else:
            self.last_info = self.sim.execute(skill=skill, cube=int(values[1]))
        self.current_state = self.observe()
        return self.current_state

    def noop_action(self) -> Action:
        return np.array([7, 0, 103, 12], dtype=float)

    def get_valid_actions(self) -> list[Action]:
        return [
            np.array([action[0], action[1], 103, 12], dtype=float)
            for action in super().get_valid_actions()
        ]
