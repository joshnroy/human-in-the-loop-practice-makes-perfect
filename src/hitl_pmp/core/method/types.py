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


# TODO(follow-up PR): Skill is a placeholder. Per predicators/structs.py's NSRT /
# _GroundNSRT / ParameterizedOption / _Option (a 3-level lifted -> objects-bound ->
# objects+params-bound hierarchy), this should probably become:
#   class Skill(BaseModel): name: str; types: tuple[Type, ...]; param_dim: int
#   class GroundSkill(BaseModel): skill: Skill; objects: tuple[Object, ...]
# with continuous params sampled inside execute_skill (not part of GroundSkill) --
# improve_skill_parameters needs to update the *sampler*, not one already-consumed
# param value, matching _GroundNSRT.sample_option(). Preconditions/effects
# (STRIPSOperator's symbolic half) deliberately deferred too: needs a Variable/
# LiftedAtom layer (predicators: Object/Variable both subclass _TypedEntity,
# distinguished by whether the name starts with "?") that nothing here consumes yet.
Skill = Any
