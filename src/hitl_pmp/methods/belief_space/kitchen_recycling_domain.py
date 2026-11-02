"""Single-object kitchen adapter for the existing belief-space practice solvers.

Deployment uses the existing pick/open/retry/toss surrogate at the far bin, with
no human and no near/far competence sharing. Real evaluation remains motor-step
bounded simulation; this surrogate is a planning value, not a measured success rate.
The surrogate assumes failed picks close an empty gripper and failed tosses end
the attempt. Practice failure effects instead use the existing empirical model:
real retained/released outcomes are learned per exact context and sampler mode;
unseen failures keep the framework's optimistic identity approximation.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hitl_pmp.core.method.skill_provider import SkillProvider

from .practice_domain import PracticeDomain, SampleSkills
from .tossing3d_deployment_model import DeploymentPolicyExpectation, evaluate_deployment_policies
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief


class KitchenRecyclingPracticeDomain(PracticeDomain):
    deployment_horizon: int = 6
    deployment_skills: tuple[str, str, str]

    @staticmethod
    def for_provider(*, provider: SkillProvider) -> "KitchenRecyclingPracticeDomain":
        items = [obj.name for obj in provider.objects() if obj.type.name == "item"]
        if len(items) != 1:
            raise ValueError("Kitchen belief-space practice requires exactly one object")
        item = items[0]
        robots = provider.skills()
        humans = tuple(reset.skill for reset in provider.movables_reset_skills())
        skills = tuple(dict.fromkeys((*robots, *humans)))
        return KitchenRecyclingPracticeDomain(
            sampler_skill=f"Toss_{item}_far",
            deployment_skills=(f"Pick_{item}_far", f"Toss_{item}_far", "OpenGripperEmpty_far"),
            belief_sources=tuple(
                (
                    skill.name,
                    PracticeExampleSource.SAMPLER
                    if skill.param_dim > 0
                    else PracticeExampleSource.OUTCOME,
                )
                for skill in skills
            ),
            human_skill_names=frozenset(skill.name for skill in humans),
            failure_effect_skills=frozenset(skill.name for skill in skills),
        )

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in self.deployment_skills}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        pick, toss, opened = (np.array([theta[name].competence]) for name in self.deployment_skills)
        return float(
            evaluate_deployment_policies(
                pick_competences=pick,
                toss_competences=toss,
                open_competences=opened,
                horizon=self.deployment_horizon,
            )[0]
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        pick, toss, opened = (
            beliefs[name].sample(rng=rng, count=count)[:, 0] for name in self.deployment_skills
        )
        return evaluate_deployment_policies(
            pick_competences=pick,
            toss_competences=toss,
            open_competences=opened,
            horizon=self.deployment_horizon,
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        pick, toss, opened = (beliefs[name] for name in self.deployment_skills)
        return DeploymentPolicyExpectation.evaluate(
            pick=pick, toss=toss, opened=opened, horizon=self.deployment_horizon
        )
