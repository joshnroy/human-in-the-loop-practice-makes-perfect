"""Shared execution accounting, independent of skill and trial segmentation.

Functions are evaluated on the pre-execution context. A prepared charge is recorded
only when execution is dispatched; receipt replay must not record it again.
"""

import importlib
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChargeFunction(BaseModel):
    """Constant by default, or a trusted host function ``module:name``.

    The callable accepts ``context=`` (actor, skill, observation, robot_steps,
    human_invocations). It must be deterministic and return a finite nonnegative
    number. It is host configuration, never agent-writable code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    value: float = Field(default=1, ge=0, allow_inf_nan=False)
    function: str | None = None

    def evaluate(self, *, context: dict[str, Any]) -> float:
        result = float(self.value)
        if self.function is not None:
            module, separator, name = self.function.partition(":")
            if not separator or not name or name.startswith("_"):
                raise ValueError("Cost function must be module:public_name")
            callback = getattr(importlib.import_module(module), name)
            result = float(callback(context=context))
        if not math.isfinite(result) or result < 0:
            raise ValueError("Cost/duration functions must return finite nonnegative numbers")
        return result


class HumanCharge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cost: ChargeFunction = Field(default_factory=ChargeFunction)
    duration: ChargeFunction = Field(default_factory=ChargeFunction)


class PracticeCosts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    robot_step: ChargeFunction = Field(default_factory=ChargeFunction)
    human_skills: dict[str, HumanCharge] = Field(
        default_factory=lambda: {
            "reset_cube_far": HumanCharge(),
            "reset_cube_and_bin_near": HumanCharge(),
        }
    )
    human_weight: float = Field(default=1, ge=0, allow_inf_nan=False)
    objective_lambda: float = Field(default=3e-6, ge=0, allow_inf_nan=False)

    @staticmethod
    def load(*, path: str | Path | None) -> "PracticeCosts":
        return (
            PracticeCosts()
            if path is None
            else PracticeCosts.model_validate_json(Path(path).read_text())
        )


class ExecutionCharge(BaseModel):
    model_config = ConfigDict(frozen=True)
    actor: Literal["robot", "human"]
    skill: str
    cost: float = Field(ge=0, allow_inf_nan=False)
    steps: int = Field(gt=0)


class PracticeAccounting(BaseModel):
    costs: PracticeCosts = Field(default_factory=PracticeCosts)
    robot_steps: int = 0
    human_steps: int = 0
    robot_cost: float = 0
    human_cost: float = 0
    human_invocations: dict[str, int] = Field(default_factory=dict)

    @property
    def practice_steps(self) -> int:
        return self.robot_steps + self.human_steps

    @property
    def total_cost(self) -> float:
        return self.robot_cost + self.costs.human_weight * self.human_cost

    def context(self, *, actor: str, skill: str, observation: Any = None) -> dict[str, Any]:
        return dict(
            actor=actor,
            skill=skill,
            observation=observation,
            robot_steps=self.robot_steps,
            human_invocations=dict(self.human_invocations),
        )

    def robot_charge(self, *, observation: Any = None) -> ExecutionCharge:
        context = self.context(actor="robot", skill="robot_step", observation=observation)
        return ExecutionCharge(
            actor="robot",
            skill="robot_step",
            steps=1,
            cost=self.costs.robot_step.evaluate(context=context),
        )

    def human_charge(self, *, skill: str, observation: Any = None) -> ExecutionCharge:
        if skill not in self.costs.human_skills:
            raise ValueError(f"Unknown human skill: {skill}")
        functions = self.costs.human_skills[skill]
        context = self.context(actor="human", skill=skill, observation=observation)
        duration = functions.duration.evaluate(context=context)
        if duration < 1 or not duration.is_integer():
            raise ValueError("Human duration must be a positive integer of counted steps")
        return ExecutionCharge(
            actor="human",
            skill=skill,
            steps=int(duration),
            cost=functions.cost.evaluate(context=context),
        )

    def record(self, *, charge: ExecutionCharge) -> None:
        if charge.actor == "robot":
            self.robot_steps += charge.steps
            self.robot_cost += charge.cost
        else:
            self.human_steps += charge.steps
            self.human_cost += charge.cost
            self.human_invocations[charge.skill] = self.human_invocations.get(charge.skill, 0) + 1

    def summary(self) -> dict[str, Any]:
        return dict(
            robot_cost=self.robot_cost,
            human_cost=self.human_cost,
            physical_cost=self.total_cost,
            human_steps=self.human_steps,
            human_invocations_by_skill=dict(self.human_invocations),
        )
