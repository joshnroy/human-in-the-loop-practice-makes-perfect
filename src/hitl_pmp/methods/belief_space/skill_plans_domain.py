"""Deployment value for a distribution of fixed, single-attempt skill plans.

This is a configurable surrogate, not a task planner: each equally weighted plan
succeeds when all its skills succeed. Recovery and transfer are not forecast.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from .practice_domain import PracticeDomain, SampleSkills
from .types.skill_belief import SkillBelief


class SkillPlansPracticeDomain(PracticeDomain):
    deployment_plans: tuple[tuple[str, ...], ...]

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
        return self

    def deployment_beliefs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for plan in self.deployment_plans for name in plan))

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        draws = {name: sample_skills(beliefs[name], count) for name in self.deployment_beliefs()}
        return [{name: rows[i] for name, rows in draws.items()} for i in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(
            np.mean([
                np.prod([theta[name].competence for name in plan]) for plan in self.deployment_plans
            ])
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        draws = {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in self.deployment_beliefs()
        }
        return np.mean(
            [np.prod([draws[name] for name in plan], axis=0) for plan in self.deployment_plans],
            axis=0,
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return float(
            np.mean([
                np.prod([beliefs[name].mean_competence() for name in plan])
                for plan in self.deployment_plans
            ])
        )
