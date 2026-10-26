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
    ) -> float:
        """Optimal task value after training, including policies not optimal today."""
        predicted = {
            skill: min(1.0, max(0.0, competence + learning_rates[skill] * allocation.get(skill, 0)))
            for skill, competence in competences.items()
        }
        value, _policy = mdp.solve(competences=predicted)
        return value

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
    ) -> dict[SkillKey, int]:
        """The integer attempt counts, one per key of `costs`, that maximise
        `predicted_success` subject to sum(b_u * c_u) <= `budget`; among equally good
        allocations, the cheapest. A transition skill outside `costs` keeps its competence.

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
        relevant = mdp.relevant_skills()

        limits: dict[SkillKey, int] = {}
        for skill, cost in costs.items():
            limit = DpAllocator._useful_attempts(
                competence=competences.get(skill, 1.0),
                learning_rate=learning_rates.get(skill, 0.0),
                affordable=math.floor(budget / cost + 1e-9),
            )
            if skill in relevant and limit > 0:
                limits[skill] = limit
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
    def _useful_attempts(*, competence: float, learning_rate: float, affordable: int) -> int:
        """The most attempts worth enumerating: none for a skill that is not improving or
        already at 1, else up to the count that first reaches 1, since every attempt past
        it costs budget and adds nothing."""
        if learning_rate <= 0.0 or competence >= 1.0:
            return 0
        return max(0, min(affordable, math.ceil((1.0 - competence) / learning_rate)))
