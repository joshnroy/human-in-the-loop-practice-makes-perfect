"""Deployment value when the goal can be reached by several routes of unequal reward.

Each route is one skill, attempted once: the deployed policy takes the route whose
reward times competence is largest, so the value of a set of competences is
`max_i(reward_i * competence_i)`. It is a reward, not a probability, and is at most the
largest route reward.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hitl_pmp.core.method.skill_provider import SkillProvider

from .practice_domain import PracticeDomain, SampleSkills
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief
from .types.weighted_hypothesis_belief import WeightedHypothesisBelief

ROUTES_EXPECTATION_SAMPLES = 4096
ROUTES_EXPECTATION_SEED = 0
OPTIMISTIC_QUANTILE = 0.9


class RewardedRoutesPracticeDomain(PracticeDomain):
    """Every route and human skill carries an inferred competence, from Beta(1, 3) as in
    the shelf and rack domains. The expectation is a fixed-seed Monte Carlo, so two
    stopping values are compared under common random numbers."""

    deployment_horizon: int = 1
    default_search_depth: int = 3
    initial_competence_alpha: float | None = 1.0
    initial_competence_beta: float | None = 3.0
    # Route skill name -> the reward of the goal state it reaches.
    route_rewards: tuple[tuple[str, float], ...]

    @staticmethod
    def from_provider(*, provider: SkillProvider) -> "RewardedRoutesPracticeDomain":
        """A route is a robot skill that adds a predicate the provider prices
        (`SkillProvider.goal_rewards`). Every other robot skill is taken to be a known
        mechanism that always works (Two Routes' `ResetA`); the human's skills are
        inferred like the routes."""
        rewards = provider.goal_rewards()
        if not rewards:
            raise ValueError(f"{type(provider).__name__} prices no goal: nothing to weigh")
        routes = tuple(
            (
                skill.name,
                max(
                    rewards[atom.predicate.name]
                    for atom in skill.add_effects
                    if atom.predicate.name in rewards
                ),
            )
            for skill in provider.skills()
            if any(atom.predicate.name in rewards for atom in skill.add_effects)
        )
        route_names = {name for name, _ in routes}
        humans = tuple(dict.fromkeys(g.skill.name for g in provider.human_cube_bin_reset_skills()))
        return RewardedRoutesPracticeDomain(
            route_rewards=routes,
            belief_sources=tuple(
                (name, PracticeExampleSource.OUTCOME)
                for name in (*(name for name, _ in routes), *humans)
            ),
            deterministic_skills=frozenset(
                skill.name for skill in provider.skills() if skill.name not in route_names
            ),
            human_skill_names=frozenset(humans),
            failure_effect_skills=frozenset((*route_names, *humans)),
        )

    def route_competence_prior(
        self, *, name: str, alpha: float, beta: float
    ) -> tuple[float, float]:
        """The Beta prior this route's belief starts from: its own override where it has
        one, else the shared (alpha, beta). Deployment must start from the same one."""
        overrides = {n: (a, b) for n, a, b in self.competence_prior_overrides}
        overrides.update({n: (m * k, (1 - m) * k) for n, m, k, _, _ in self.skill_priors})
        return overrides.get(name, (alpha, beta))

    def optimistic_policy_value(
        self, *, beliefs: Mapping[str, SkillBelief], tries_left: float
    ) -> float:
        """An upper bound on the deployment value reachable with `tries_left` more
        practice tries: every try given to whichever single route ends worth the most,
        at the most optimistic competence and learning rate the belief holds."""
        best = 0.0
        for name, reward in self.route_rewards:
            belief = beliefs[name]
            if isinstance(belief, WeightedHypothesisBelief):
                reaches = sorted(
                    (
                        min(
                            1.0,
                            item.hypothesis.competence + item.hypothesis.learning_rate * tries_left,
                        ),
                        item.probability,
                    )
                    for item in belief.hypotheses
                )
                # Exact for a known curve; for a spread-out prior its 90th percentile, since
                # the extreme tail would make every route look perfect.
                reach, mass = reaches[-1][0], 0.0
                for value, probability in reaches:
                    mass += probability
                    if mass >= OPTIMISTIC_QUANTILE:
                        reach = value
                        break
            else:
                rows = belief.sample(rng=np.random.default_rng(ROUTES_EXPECTATION_SEED), count=512)
                reach = float(np.max(np.minimum(1.0, rows[:, 0] + rows[:, 1] * tries_left)))
            best = max(best, reward * reach)
        return best

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        draws = {name: sample_skills(beliefs[name], count) for name, _ in self.route_rewards}
        return [{name: rows[i] for name, rows in draws.items()} for i in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(max(reward * theta[name].competence for name, reward in self.route_rewards))

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        return np.max(
            [
                reward * beliefs[name].sample(rng=rng, count=count)[:, 0]
                for name, reward in self.route_rewards
            ],
            axis=0,
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return float(
            self.sample_policy_values(
                beliefs=beliefs,
                rng=np.random.default_rng(ROUTES_EXPECTATION_SEED),
                count=ROUTES_EXPECTATION_SAMPLES,
            ).mean()
        )
