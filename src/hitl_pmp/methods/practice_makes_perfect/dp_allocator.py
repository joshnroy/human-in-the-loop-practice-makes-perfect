"""The budget allocation step of the Deliberate Practice baseline (Vats & Konidaris,
*Deliberate Practice: Learning Robot Skills under a Budget*), adapted to this project.

The paper splits a practice budget B across skills before practice, maximising
predicted task success under the linear competence model min(1, p_u + Delta_u * b_u),
and solves that globally as a bilinear program in Gurobi. Here the same optimum is found
by exact enumeration over integer allocations, which is tractable because only skills on
a seen task's plan with room to improve are enumerated, each only up to the attempt
count that saturates it. Above a cap the allocator raises: it never falls back to a
heuristic, because a silently approximate optimum would no longer be this baseline.

Two departures from the paper, both ours: the budget constraint is sum_u b_u * c_u <= B
with a per-attempt cost c_u, where the paper counts every practice episode alike
(sum_u b_u <= B); and the objective is the mean, over the seen tasks' plans, of each
plan's product of predicted competences, where the paper values its abstract MDP."""

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

SkillKey = TypeVar("SkillKey", bound=Hashable)

# Equal predicted success is decided by spend first; this absorbs float noise in both.
_TOLERANCE = 1e-12


class DpAllocator:
    """Static-method container: an allocation is a pure function of its inputs."""

    @staticmethod
    def predicted_success(
        *,
        plans: Sequence[Sequence[SkillKey]],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        allocation: Mapping[SkillKey, int],
    ) -> float:
        """The mean over `plans` of the product of each step's predicted competence after
        `allocation[step]` more attempts (0 for a step it does not name)."""
        if not plans:
            return 0.0
        total = 0.0
        for plan in plans:
            product = 1.0
            for step in plan:
                predicted = competences[step] + learning_rates[step] * allocation.get(step, 0)
                product *= min(1.0, max(0.0, predicted))
            total += product
        return total / len(plans)

    @staticmethod
    def allocate(
        *,
        plans: Sequence[Sequence[SkillKey]],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        costs: Mapping[SkillKey, float],
        budget: float,
        max_skills: int = 8,
        max_allocations: int = 200_000,
    ) -> dict[SkillKey, int]:
        """The integer attempt counts, one per key of `costs`, that maximise
        `predicted_success` subject to sum(b_u * c_u) <= `budget`; among equally good
        allocations, the cheapest. A plan step outside `costs` keeps its competence.

        Raises `ValueError` when more than `max_skills` skills would be enumerated, or
        when the enumeration could exceed `max_allocations` allocations."""
        for skill, cost in costs.items():
            if not cost > 0.0 or not math.isfinite(cost):
                raise ValueError(f"cost of {skill!r} must be finite and positive, got {cost}")
        steps = {step for plan in plans for step in plan}
        missing = [step for step in steps if step not in competences or step not in learning_rates]
        if missing:
            raise ValueError(f"no competence or learning rate for plan steps {missing!r}")
        allocation = dict.fromkeys(costs, 0)
        if budget <= 0.0 or not plans:
            return allocation

        limits: dict[SkillKey, int] = {}
        for skill, cost in costs.items():
            limit = DpAllocator._useful_attempts(
                competence=competences.get(skill, 1.0),
                learning_rate=learning_rates.get(skill, 0.0),
                affordable=math.floor(budget / cost + 1e-9),
            )
            if skill in steps and limit > 0:
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
                    plans=plans,
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
