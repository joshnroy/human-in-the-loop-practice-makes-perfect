"""Exact integer DP allocation, jointly optimizing the deployment task policy.

Every allocation is evaluated by solving the full success/failure task MDP, not
by holding today's deployment plans fixed. The cost-weighted budget is our
adaptation; the paper counts practice episodes equally. Explicit enumeration
caps raise rather than silently substitute a heuristic.
"""

import math
from collections.abc import Hashable, Mapping
from typing import TypeVar

from .dp_mdp import DpMdp

SkillKey = TypeVar("SkillKey", bound=Hashable)

# Equal predicted success is decided by spend first; this absorbs float noise in both.
_TOLERANCE = 1e-12


class DpAllocator:
    """Static-method container: an allocation is a pure function of its inputs."""

    @staticmethod
    def predicted_success(
        *,
        mdp: DpMdp[SkillKey],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        allocation: Mapping[SkillKey, int],
        groups: Mapping[SkillKey, SkillKey] | None = None,
    ) -> float:
        """Optimal task value after training, including policies not optimal today."""
        predicted = DpAllocator.predicted_competences(
            competences=competences,
            learning_rates=learning_rates,
            allocation=allocation,
            groups=groups,
        )
        value, _policy = mdp.solve(competences=predicted)
        return value

    @staticmethod
    def predicted_competences(
        *,
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        allocation: Mapping[SkillKey, int],
        groups: Mapping[SkillKey, SkillKey] | None = None,
    ) -> dict[SkillKey, float]:
        """Expand logical training counts into concrete deployment probabilities."""
        identities = groups if groups is not None else {}
        return {
            skill: min(
                1.0,
                max(
                    0.0,
                    competence
                    + learning_rates[skill] * allocation.get(identities.get(skill, skill), 0),
                ),
            )
            for skill, competence in competences.items()
        }

    @staticmethod
    def allocate(
        *,
        mdp: DpMdp[SkillKey],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        costs: Mapping[SkillKey, float],
        budget: float,
        max_skills: int = 8,
        max_allocations: int = 200_000,
        groups: Mapping[SkillKey, SkillKey] | None = None,
    ) -> dict[SkillKey, int]:
        """The integer attempt counts, one per key of `costs`, that maximise
        `predicted_success` subject to sum(b_u * c_u) <= `budget`; among equally good
        allocations, the cheapest. `groups` maps concrete skills to allocation keys;
        each key in `costs` is charged once and returned once. An unpriced group
        keeps its competence. Omitted mappings use the concrete skill itself.

        Raises `ValueError` when more than `max_skills` skills would be enumerated, or
        when the enumeration could exceed `max_allocations` allocations."""
        for skill, cost in costs.items():
            if not cost > 0.0 or not math.isfinite(cost):
                raise ValueError(f"cost of {skill!r} must be finite and positive, got {cost}")
        steps = {transition.skill for transition in mdp.transitions}
        missing = [step for step in steps if step not in competences or step not in learning_rates]
        if missing:
            raise ValueError(f"no competence or learning rate for MDP skills {missing!r}")
        if not math.isfinite(budget):
            raise ValueError("DP budget must be finite")
        for skill in steps:
            if not 0 <= competences[skill] <= 1 or not math.isfinite(learning_rates[skill]):
                raise ValueError(f"invalid competence or learning rate for {skill!r}")
        allocation = dict.fromkeys(costs, 0)
        if budget <= 0.0 or not mdp.initial_probabilities:
            return allocation
        limits = {
            skill: limit
            for skill, limit in DpAllocator._allocation_limits(
                mdp=mdp,
                competences=competences,
                learning_rates=learning_rates,
                costs=costs,
                budget=budget,
                groups=groups,
            ).items()
            if limit > 0
        }
        if len(limits) > max_skills:
            raise ValueError(
                f"DP allocation over {len(limits)} improvable skills exceeds the cap of "
                f"{max_skills} skills; raise max_skills or shrink the problem"
            )
        bound = math.prod(limit + 1 for limit in limits.values())
        if bound > max_allocations:
            raise ValueError(
                f"DP allocation could enumerate {bound} allocations, above the cap of "
                f"{max_allocations}; raise max_allocations or shrink the problem"
            )

        active = list(limits)
        best_value = -math.inf
        best_spent = math.inf
        best: dict[SkillKey, int] = {}
        counts: dict[SkillKey, int] = {}

        def visit(*, index: int, spent: float) -> None:
            nonlocal best_value, best_spent, best
            if index == len(active):
                value = DpAllocator.predicted_success(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    allocation=counts,
                    groups=groups,
                )
                if value > best_value + _TOLERANCE or (
                    value >= best_value - _TOLERANCE and spent < best_spent - _TOLERANCE
                ):
                    best_value, best_spent, best = value, spent, dict(counts)
                return
            skill = active[index]
            for count in range(limits[skill] + 1):
                cost = spent + count * costs[skill]
                if cost > budget + 1e-9:
                    break
                counts[skill] = count
                visit(index=index + 1, spent=cost)
            counts.pop(skill, None)

        visit(index=0, spent=0.0)
        allocation.update(best)
        return allocation

    @staticmethod
    def _allocation_limits(
        *,
        mdp: DpMdp[SkillKey],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        costs: Mapping[SkillKey, float],
        budget: float,
        groups: Mapping[SkillKey, SkillKey] | None = None,
    ) -> dict[SkillKey, int]:
        """A shared count remains useful until its last improving member saturates."""
        identities = groups if groups is not None else {}
        for group in costs:
            if identities.get(group, group) != group:
                raise ValueError("DP costs must price allocation groups, not duplicate members")
        limits = dict.fromkeys(costs, 0)
        for skill in mdp.relevant_skills():
            group = identities.get(skill, skill)
            if group not in costs:
                continue
            limits[group] = max(
                limits[group],
                DpAllocator._useful_attempts(
                    competence=competences[skill],
                    learning_rate=learning_rates[skill],
                    affordable=max(0, math.floor(budget / costs[group] + 1e-9)),
                ),
            )
        return limits

    @staticmethod
    def _useful_attempts(*, competence: float, learning_rate: float, affordable: int) -> int:
        """The most attempts worth enumerating: none for a skill that is not improving or
        already at 1, else up to the count that first reaches 1, since every attempt past
        it costs budget and adds nothing."""
        if learning_rate <= 0.0 or competence >= 1.0:
            return 0
        return max(0, min(affordable, math.ceil((1.0 - competence) / learning_rate)))
