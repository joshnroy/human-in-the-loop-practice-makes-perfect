"""A small reference reset selector for checking the environment's tradeoff.

This is not EES or the belief-space practice planner. It learns only the human's
fixed Bernoulli stack competence, then compares one human attempt followed by
robot recovery against direct robot reset. Calibration is explicitly scheduled
by the demo harness, rather than disguised as cost-optimal exploration.
"""

import math

from pydantic import BaseModel, Field


class TableSettingResetSelector(BaseModel):
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)

    @property
    def competence_estimate(self) -> float:
        return (1 + self.successes) / (2 + self.successes + self.failures)

    def observe(self, *, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def choose(
        self, *, robot_cost: float, human_attempt_cost: float, spill_recovery_cost: float
    ) -> str:
        if any(
            not math.isfinite(cost) or cost < 0
            for cost in (robot_cost, human_attempt_cost, spill_recovery_cost)
        ):
            raise ValueError("Reset costs must be finite and nonnegative")
        human_expected = human_attempt_cost + (1 - self.competence_estimate) * spill_recovery_cost
        return "human" if human_expected < robot_cost else "robot"
