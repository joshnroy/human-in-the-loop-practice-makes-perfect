"""Deployment value for a distribution of fixed, single-attempt skill plans.

This is a configurable surrogate, not a task planner: each equally weighted plan
succeeds when all its skills succeed. Recovery and transfer are not forecast.

With `skill_steps` the domain also models deployment cost (practice_domain.py): a plan
runs its skills in order and ends at its first failure, each attempt costing its
skill's success or failure control steps. The plans being independent and distinct,
every expectation below is a product of mean competences, so it is exact.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from .practice_domain import PracticeDomain, SampleSkills
from .types.skill_belief import SkillBelief

# Mean controller steps per robot skill execution, (on success, on failure), over the
# practice robot events of the 2026-11-06 pouring3d_kitchen Della sweep (hrsweep
# 16997464, 35/36 runs). Failure means include 0-tick refusals. Executions
# (success/failure): DropShrimpIntoPitcher 770/164, PickPitcher 3150/10,
# PickShrimpFromFloor 2/15, PickShrimpFromSurface 27/1, PickShrimpFromTray 911/76,
# PlacePitcherAtHome 2831/1884, Pour1 330/735, Pour2 286/1060, Pour3 783/915.
POURING_KITCHEN_SKILL_STEPS: dict[str, tuple[float, float]] = {
    "DropShrimpIntoPitcher": (449.0, 94.0),
    "PickPitcher": (512.0, 276.0),
    "PickShrimpFromFloor": (294.0, 122.0),
    "PickShrimpFromSurface": (607.0, 266.0),
    "PickShrimpFromTray": (600.0, 341.0),
    "PlacePitcherAtHome": (324.0, 79.0),
    "Pour1": (969.0, 276.0),
    "Pour2": (1061.0, 276.0),
    "Pour3": (895.0, 101.0),
}

Competence = TypeVar("Competence", float, NDArray[np.float64])


class SkillPlansPracticeDomain(PracticeDomain):
    deployment_plans: tuple[tuple[str, ...], ...]
    # (success, failure) control steps per planned skill; empty models no cost.
    skill_steps: dict[str, tuple[float, float]] = {}

    @model_validator(mode="after")
    def validate_plans(self) -> "SkillPlansPracticeDomain":
        known = dict(self.belief_sources)
        if not self.deployment_plans or any(not plan for plan in self.deployment_plans):
            raise ValueError("Deployment plans must be nonempty")
        for plan in self.deployment_plans:
            if len(set(plan)) != len(plan):
                raise ValueError("Each plan must use independent, distinct skill beliefs")
            if any(name not in known for name in plan):
                raise ValueError("Deployment plans must reference declared skill beliefs")
        if self.skill_steps and not set(self.deployment_beliefs()) <= set(self.skill_steps):
            raise ValueError("skill_steps must price every skill a deployment plan runs")
        return self

    def deployment_beliefs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for plan in self.deployment_plans for name in plan))

    def _step_weight(self) -> float:
        return self.deployment_cost_lambda * self.robot_step_price if self.skill_steps else 0.0

    def _plan_steps(self, *, competence: Callable[[str], Competence]) -> Competence:
        """E[control steps per task], averaged over the equally weighted plans."""
        total: Any = 0.0
        for plan in self.deployment_plans:
            reached: Any = 1.0
            for name in plan:
                success, failure = self.skill_steps[name]
                p = competence(name)
                total = total + reached * (p * success + (1.0 - p) * failure)
                reached = reached * p
        return total / len(self.deployment_plans)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        draws = {name: sample_skills(beliefs[name], count) for name in self.deployment_beliefs()}
        return [{name: rows[i] for name, rows in draws.items()} for i in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        solved = float(
            np.mean([
                np.prod([theta[name].competence for name in plan]) for plan in self.deployment_plans
            ])
        )
        weight = self._step_weight()
        if not weight:
            return solved
        return solved - weight * float(
            self._plan_steps(competence=lambda name: float(theta[name].competence))
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        draws = {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in self.deployment_beliefs()
        }
        solved = np.mean(
            [np.prod([draws[name] for name in plan], axis=0) for plan in self.deployment_plans],
            axis=0,
        )
        weight = self._step_weight()
        if not weight:
            return solved
        return solved - weight * self._plan_steps(competence=lambda name: draws[name])

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        solved = float(
            np.mean([
                np.prod([beliefs[name].mean_competence() for name in plan])
                for plan in self.deployment_plans
            ])
        )
        weight = self._step_weight()
        if not weight:
            return solved
        return solved - weight * self._mean_steps(beliefs=beliefs)

    def _mean_steps(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return float(
            self._plan_steps(competence=lambda name: float(beliefs[name].mean_competence()))
        )

    def expected_deployment_cost(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        """Expected control steps per task, times the run's robot-step price."""
        if not self.skill_steps:
            return 0.0
        return self._mean_steps(beliefs=beliefs) * self.robot_step_price
