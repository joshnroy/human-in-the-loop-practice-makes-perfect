from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Goal

Policy = Callable[[State], Action]


class Rollout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: list[State]
    actions: list[Action]

    @model_validator(mode="after")
    def _check_lengths(self) -> Rollout:
        if len(self.actions) != len(self.states) - 1:
            raise ValueError(
                f"Rollout has {len(self.states)} states but {len(self.actions)} "
                "actions; expected len(actions) == len(states) - 1."
            )
        return self


class SetupCommand(BaseModel):
    """Either the robot executes this goal itself (execute_setup_command) or it's
    handed to the human (execute_human_command) -- target says which."""

    target: SetupCommandTarget
    goal: Goal


class SetupCommandTarget(Enum):
    ROBOT = "robot"
    HUMAN = "human"


Skill = Any
