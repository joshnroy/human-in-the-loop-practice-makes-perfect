"""Deployment value when the goal can be reached by several routes of unequal reward.

Each route is one skill, attempted once: the deployed policy takes the route whose
reward times competence is largest, so the value of a set of competences is
`max_i(reward_i * competence_i)`. It is a reward, not a probability, and is at most the
largest route reward.
"""

from collections import OrderedDict
from collections.abc import Collection, Mapping
from typing import Any, ClassVar

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
        self,
        *,
        beliefs: Mapping[str, SkillBelief],
        tries_left: float,
        routes: Collection[str] | None = None,
    ) -> float:
        """An upper bound on the deployment value reachable with `tries_left` more
        practice tries: every try given to whichever single route ends worth the most,
        at the most optimistic competence and learning rate the belief holds. With
        `routes`, only those routes are counted."""
        best = 0.0
        for name, reward in self.route_rewards:
            if routes is not None and name not in routes:
                continue
            belief = beliefs[name]
            if isinstance(belief, WeightedHypothesisBelief):
                reaches = sorted(
                    (min(1.0, competence + rate * tries_left), probability)
                    for competence, rate, probability in zip(
                        belief.competences, belief.learning_rates, belief.probabilities, strict=True
                    )
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
        key = self.value_memo_key(beliefs=beliefs)
        memo = RewardedRoutesPracticeDomain._value_memo
        if key is not None and (cached := memo.get(key)) is not None:
            memo.move_to_end(key)
            return cached
        value = float(
            self.sample_policy_values(
                beliefs=beliefs,
                rng=np.random.default_rng(ROUTES_EXPECTATION_SEED),
                count=ROUTES_EXPECTATION_SAMPLES,
            ).mean()
        )
        if key is not None:
            memo[key] = value
            while len(memo) > RewardedRoutesPracticeDomain.VALUE_MEMO_SIZE:
                memo.popitem(last=False)
        return value

    # A search values the same beliefs again and again -- J and the open-list bound value
    # each node, and a known curve reaches one belief by many paths -- and each value is
    # a fresh fixed-seed draw of thousands of samples. Bounded: a prior grid's key holds
    # its competences and weights, about 7 kB per route.
    VALUE_MEMO_SIZE: ClassVar[int] = 4096
    _value_memo: ClassVar[OrderedDict[tuple[object, ...], float]] = OrderedDict()

    def value_memo_key(self, *, beliefs: Mapping[str, SkillBelief]) -> tuple[object, ...] | None:
        """What the fixed-seed value depends on, exactly, or None to compute it afresh.

        A weighted belief's draw reads only its weights (which rows the uniforms pick)
        and competences (the rows' first column); the bytes tell -0.0 from 0.0, which
        compare equal but need not give the same float."""
        if type(self) is not RewardedRoutesPracticeDomain:
            return None
        parts: list[object] = [self.route_rewards]
        for name, _ in self.route_rewards:
            belief = beliefs[name]
            if type(belief) is not WeightedHypothesisBelief:
                return None
            parts.append(
                np.array(belief.probabilities).tobytes() + np.array(belief.competences).tobytes()
            )
        return tuple(parts)

    @staticmethod
    def clear_value_memo() -> None:
        RewardedRoutesPracticeDomain._value_memo.clear()

    @staticmethod
    def value_memo_size() -> int:
        return len(RewardedRoutesPracticeDomain._value_memo)
