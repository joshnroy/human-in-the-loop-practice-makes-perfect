"""Pouring practice value and count-conditioned competence contracts.

The deployment surrogate is one physical Pick-Pour-Place attempt, averaged over
explicit test counts. It does not assume successful recovery from a wrong split,
or transfer of competence between counts. The parameter sampler remains shared.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from hitl_pmp.core.method.types import GroundSkill

from .practice_domain import PracticeDomain, SampleSkills
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief


class PouringPracticeDomain(PracticeDomain):
    deployment_horizon: int = 3
    sampler_skill: str = "Pour"
    evaluation_counts: tuple[int, ...] = (1, 2, 3)
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = ()
    initial_competence_alpha: float | None = 1.0
    initial_competence_beta: float | None = 3.0
    default_search_depth: int = 4
    deterministic_skills: frozenset[str] = frozenset({"Wait"})
    human_skill_names: frozenset[str] = frozenset({"reset_shrimp"})
    failure_effect_skills: frozenset[str] = frozenset({
        "PickPitcher",
        "Pour",
        "PlacePitcherAtHome",
        "PickShrimpFromTray",
        "PickShrimpFromFloor",
        "DropShrimpIntoPitcher",
        "reset_shrimp",
    })
    payload_count: int = Field(default=3, ge=1)

    @classmethod
    def create(
        cls, *, payload_count: int, evaluation_counts: tuple[int, ...]
    ) -> "PouringPracticeDomain":
        if (
            not evaluation_counts
            or len(set(evaluation_counts)) != len(evaluation_counts)
            or any(type(k) is not int or not 1 <= k <= payload_count for k in evaluation_counts)
        ):
            raise ValueError("Evaluation counts must be unique and inside the payload range")
        support = (
            "PickPitcher",
            "PlacePitcherAtHome",
            "PickShrimpFromTray",
            "PickShrimpFromFloor",
            "DropShrimpIntoPitcher",
            "reset_shrimp",
        )
        return cls(
            payload_count=payload_count,
            evaluation_counts=evaluation_counts,
            belief_sources=(
                *((name, PracticeExampleSource.OUTCOME) for name in support),
                *(
                    (f"Pour/{k}", PracticeExampleSource.SAMPLER)
                    for k in range(1, payload_count + 1)
                ),
            ),
        )

    def ground_belief_name(self, *, ground_skill: GroundSkill) -> str | None:
        if ground_skill.skill.name != self.sampler_skill:
            return super().ground_belief_name(ground_skill=ground_skill)
        bindings = dict(
            zip((v.name for v in ground_skill.skill.parameters), ground_skill.objects, strict=True)
        )
        request = bindings["k"].name.removeprefix("request_")
        name = f"Pour/{request}"
        if name not in dict(self.belief_sources):
            raise ValueError(f"Unsupported count binding {request}")
        return name

    def deployment_beliefs(self) -> tuple[str, ...]:
        return ("PickPitcher", "PlacePitcherAtHome", *(f"Pour/{k}" for k in self.evaluation_counts))

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        draws = {name: sample_skills(beliefs[name], count) for name in self.deployment_beliefs()}
        return [{name: rows[i] for name, rows in draws.items()} for i in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(
            theta["PickPitcher"].competence
            * theta["PlacePitcherAtHome"].competence
            * np.mean([theta[f"Pour/{k}"].competence for k in self.evaluation_counts])
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        draws = {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in self.deployment_beliefs()
        }
        return (
            draws["PickPitcher"]
            * draws["PlacePitcherAtHome"]
            * np.mean([draws[f"Pour/{k}"] for k in self.evaluation_counts], axis=0)
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        # Independent beliefs make this expectation exact; no Monte Carlo noise.
        return float(
            beliefs["PickPitcher"].mean_competence()
            * beliefs["PlacePitcherAtHome"].mean_competence()
            * np.mean([beliefs[f"Pour/{k}"].mean_competence() for k in self.evaluation_counts])
        )
