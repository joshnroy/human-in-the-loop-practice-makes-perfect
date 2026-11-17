"""Deployment value when the goal can be reached by several routes of unequal reward.

Each route is one skill, attempted once: the deployed policy takes the route whose
reward times competence is largest, so the value of a set of competences is
`max_i(reward_i * competence_i)`. It is a reward, not a probability, and is at most the
largest route reward.
"""

import hashlib
from collections import OrderedDict
from collections.abc import Collection, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import LiftedAtom

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
    # Route skill name -> the learned skills that must succeed before it, in order
    # (Two Routes' chained Route B: RouteB, then RouteB2). A route is worth its reward
    # times every one of its skills' competence. Empty for a one-skill route.
    route_prerequisites: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def route_chain(self, *, name: str) -> tuple[str, ...]:
        """A route's skill first, then the skills before it: the order values are drawn."""
        return (name, *dict(self.route_prerequisites).get(name, ()))

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
        prerequisites = RewardedRoutesPracticeDomain.chains(
            provider=provider, routes=tuple(name for name, _ in routes)
        )
        chained = tuple(dict.fromkeys(n for _, before in prerequisites for n in before))
        route_names = {name for name, _ in routes} | set(chained)
        humans = tuple(dict.fromkeys(g.skill.name for g in provider.human_cube_bin_reset_skills()))
        return RewardedRoutesPracticeDomain(
            route_rewards=routes,
            route_prerequisites=prerequisites,
            # A chain is the one structure where a try's odds decide what else gets
            # practiced (sweep5 only after gather succeeds), so only there must the
            # forecast see each try's learning at once, as the environment does.
            competence_advances_every_try=bool(prerequisites),
            belief_sources=tuple(
                (name, PracticeExampleSource.OUTCOME)
                for name in (*(name for name, _ in routes), *chained, *humans)
            ),
            deterministic_skills=frozenset(
                skill.name for skill in provider.skills() if skill.name not in route_names
            ),
            human_skill_names=frozenset(humans),
            failure_effect_skills=frozenset((*route_names, *humans)),
            failure_kept_predicates=tuple(sorted(provider.failure_kept_predicates().items())),
        )

    @staticmethod
    def chains(
        *, provider: SkillProvider, routes: tuple[str, ...]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Each route's prerequisite skills, read off preconditions and effects: walking
        back from the route's skill, a precondition the human's reset does not restore
        must be added by an earlier skill, which must succeed first. The start is what
        a reset restores: the human's, or any robot skill that undoes the goal every
        route reaches."""
        skills = {skill.name: skill for skill in provider.skills()}

        def names(*, atoms: Collection[LiftedAtom]) -> set[str]:
            return {atom.predicate.name for atom in atoms}

        goal = set.intersection(*(names(atoms=skills[route].add_effects) for route in routes))
        start = {
            atom.predicate.name
            for ground in provider.human_cube_bin_reset_skills()
            for atom in ground.add_effects
        }
        for skill in skills.values():
            if names(atoms=skill.delete_effects) & goal:
                start |= names(atoms=skill.add_effects)
        found = []
        for route in routes:
            before: list[str] = []
            current = skills[route]
            while needed := names(atoms=current.preconditions) - start:
                current = next(
                    skill for skill in skills.values() if needed & names(atoms=skill.add_effects)
                )
                if current.name in before or current.name == route:
                    raise ValueError(f"{route}'s prerequisites form a cycle at {current.name}")
                before.insert(0, current.name)
            if before:
                found.append((route, tuple(before)))
        return tuple(found)

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
            value = reward
            for skill in self.route_chain(name=name):
                value *= self.optimistic_reach(belief=beliefs[skill], tries_left=tries_left)
            best = max(best, value)
        return best

    @staticmethod
    def optimistic_reach(*, belief: SkillBelief, tries_left: float) -> float:
        """The competence `tries_left` more tries reach at the belief's optimistic end."""
        if isinstance(belief, WeightedHypothesisBelief):
            reaches = sorted(
                (min(1.0, competence + rate * tries_left), probability)
                for competence, rate, probability in zip(
                    belief.competences.tolist(),
                    belief.learning_rates.tolist(),
                    belief.probabilities.tolist(),
                    strict=True,
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
            return reach
        rows = belief.sample(rng=np.random.default_rng(ROUTES_EXPECTATION_SEED), count=512)
        return float(np.max(np.minimum(1.0, rows[:, 0] + rows[:, 1] * tries_left)))

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        draws = {
            skill: sample_skills(beliefs[skill], count)
            for name, _ in self.route_rewards
            for skill in self.route_chain(name=name)
        }
        return [{name: rows[i] for name, rows in draws.items()} for i in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(
            max(
                reward * np.prod([theta[skill].competence for skill in self.route_chain(name=name)])
                for name, reward in self.route_rewards
            )
        )

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
    # a fresh fixed-seed draw of thousands of samples. Bounded, and each key a 32-byte
    # digest: the bytes it digests run to tens of kB per prior grid, and a memo of
    # them held ~240 MiB.
    VALUE_MEMO_SIZE: ClassVar[int] = 4096
    _value_memo: ClassVar[OrderedDict[bytes, float]] = OrderedDict()

    def value_memo_key(self, *, beliefs: Mapping[str, SkillBelief]) -> bytes | None:
        """What the fixed-seed value depends on, exactly, or None to compute it afresh.

        A weighted belief's draw reads only its weights (which rows the uniforms pick)
        and competences (the rows' first column); the bytes tell -0.0 from 0.0, which
        compare equal but need not give the same float. The key is the SHA-256 of those
        bytes, each route's length-prefixed, after the routes and their rewards."""
        if type(self) is not RewardedRoutesPracticeDomain:
            return None
        digest = hashlib.sha256(repr(self.route_rewards).encode())
        for name, _ in self.route_rewards:
            for skill in self.route_chain(name=name):
                belief = beliefs[skill]
                if type(belief) is not WeightedHypothesisBelief:
                    return None
                encoded = (
                    np.asarray(belief.probabilities).tobytes()
                    + np.asarray(belief.competences).tobytes()
                )
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return digest.digest()

    @staticmethod
    def clear_value_memo() -> None:
        RewardedRoutesPracticeDomain._value_memo.clear()

    @staticmethod
    def value_memo_size() -> int:
        return len(RewardedRoutesPracticeDomain._value_memo)
