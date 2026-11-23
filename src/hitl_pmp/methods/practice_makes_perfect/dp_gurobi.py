"""Joint integer training allocation and task-policy optimization using Gurobi.

The bilinear terms couple predicted skill success with task-policy occupancy.
An acyclic deterministic policy suffices for nonnegative terminal rewards and
success-or-absorbing-failure dynamics. Policy ranks exclude circulation, giving
valid unit occupancy bounds even for the undiscounted success objective.

Task value is nondecreasing in every success probability. When an affine forecast
stays nonnegative, p <= raw and p <= 1 preserve the maximum task value. The
probabilities themselves need not equal the clipped forecasts: increasing them cannot
reduce task value or change spend. Forecasts crossing zero still need the lower
clip, including declining siblings of an improving shared group. This avoids
numerically fragile MIN equalities near upper saturation without changing the
optimization objective or the independent certification checks.

Dual feasibility must be tightened alongside primal feasibility and MIP gaps:
small learning gains can fall below the default reduced-cost tolerance, yielding
a suboptimal allocation and an understated solver bound that agree with each other.

Gurobi can also report OPTIMAL on a point that violates the bilinear flow equalities by
far more than FeasibilityTol in the original model (presolve and scaling act on a
transformed one). Such a point can claim unattainable reward, so the independent value
check rejects it; that rejection is correct and is never loosened. Instead a rejected
solve is repeated once with NumericFocus raised, and certified again. A solve that
passes on the first attempt is never repeated, so retrying changes only solves that
would otherwise have raised.
"""

import importlib.util
import math
import time
from collections.abc import Hashable, Mapping
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from .dp_allocator import DpAllocator
from .dp_mdp import DpMdp

SkillKey = TypeVar("SkillKey", bound=Hashable)


class DpAllocationResult(BaseModel, Generic[SkillKey]):
    allocation: dict[SkillKey, int]
    solver: str
    status: str
    objective: float
    objective_bound: float
    relative_gap: float
    absolute_gap: float = 0.0
    tie_break_retries: int = 0
    runtime_seconds: float
    # Gurobi's MaxVio for the returned solution, in the original model; None off Gurobi.
    max_violation: float | None = None
    # The rejected first attempt's diagnostics when the certified result is a retry.
    numeric_retry: dict[str, object] | None = None


class DpOptimizationError(RuntimeError):
    def __init__(self, *, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"DP requires certified optimization; no fallback: {diagnostics}")


class DpGurobiAllocator:
    FEASIBILITY_TOLERANCE: ClassVar[float] = 1e-9
    # The value check's slack: the solver's own tolerances are all 1e-9.
    CERTIFICATION_TOLERANCE: ClassVar[float] = 1e-7
    RETRY_NUMERIC_FOCUS: ClassVar[int] = 3

    @staticmethod
    def allocate(
        *,
        mdp: DpMdp[SkillKey],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        costs: Mapping[SkillKey, float],
        budget: float,
        time_limit: float = 60.0,
        groups: Mapping[SkillKey, SkillKey] | None = None,
    ) -> DpAllocationResult[SkillKey]:
        if importlib.util.find_spec("gurobipy") is None:
            raise RuntimeError("DP's Gurobi backend needs the optional 'dp' dependency")

        if not math.isfinite(budget) or not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("DP budget must be finite and solver time limit finite and positive")
        for skill, cost in costs.items():
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError(f"invalid DP cost for {skill!r}")
        identities = groups if groups is not None else {}
        skills = list(
            dict.fromkeys([
                *(edge.skill for edge in mdp.transitions),
                *(costs if groups is None else competences),
            ])
        )
        for skill in skills:
            if skill not in competences or skill not in learning_rates:
                raise ValueError(f"no competence or learning rate for {skill!r}")
            if not 0 <= competences[skill] <= 1 or not math.isfinite(learning_rates[skill]):
                raise ValueError(f"invalid competence or learning rate for {skill!r}")
        started = time.monotonic()
        for group in costs:
            if identities.get(group, group) != group:
                raise ValueError("DP costs must price allocation groups, not duplicate members")
        limits = {group: max(0, math.floor(budget / cost + 1e-9)) for group, cost in costs.items()}
        first = DpGurobiAllocator._solve(
            mdp=mdp,
            competences=competences,
            learning_rates=learning_rates,
            costs=costs,
            budget=budget,
            time_limit=time_limit,
            groups=groups,
            skills=skills,
            limits=limits,
            numeric_focus=None,
        )
        if isinstance(first, DpAllocationResult):
            first.runtime_seconds = time.monotonic() - started
            return first
        retry = DpGurobiAllocator._solve(
            mdp=mdp,
            competences=competences,
            learning_rates=learning_rates,
            costs=costs,
            budget=budget,
            time_limit=time_limit,
            groups=groups,
            skills=skills,
            limits=limits,
            numeric_focus=DpGurobiAllocator.RETRY_NUMERIC_FOCUS,
        )
        if isinstance(retry, DpAllocationResult):
            retry.numeric_retry = first
            retry.runtime_seconds = time.monotonic() - started
            return retry
        raise DpOptimizationError(diagnostics={**retry, "first_attempt": first})

    @staticmethod
    def _solve(
        *,
        mdp: DpMdp[SkillKey],
        competences: Mapping[SkillKey, float],
        learning_rates: Mapping[SkillKey, float],
        costs: Mapping[SkillKey, float],
        budget: float,
        time_limit: float,
        groups: Mapping[SkillKey, SkillKey] | None,
        skills: list[SkillKey],
        limits: dict[SkillKey, int],
        numeric_focus: int | None,
    ) -> DpAllocationResult[SkillKey] | dict[str, object]:
        """One model build and solve: a certified result, or the diagnostics of a solve
        whose allocation failed the independent value check."""
        import gurobipy as gp

        identities = groups if groups is not None else {}
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", 0)
            environment.start()
            with gp.Model("deliberate-practice", env=environment) as model:
                model.Params.NonConvex = 2
                model.Params.Threads = 1
                model.Params.TimeLimit = time_limit
                model.Params.MIPGap = 1e-9
                model.Params.MIPGapAbs = 1e-9
                model.Params.FeasibilityTol = DpGurobiAllocator.FEASIBILITY_TOLERANCE
                model.Params.IntFeasTol = 1e-9
                model.Params.OptimalityTol = 1e-9
                if numeric_focus is not None:
                    model.Params.NumericFocus = numeric_focus
                counts = {
                    group: model.addVar(
                        lb=0, ub=limits[group], vtype=gp.GRB.INTEGER, name=f"b_{index}"
                    )
                    for index, group in enumerate(costs)
                }
                probabilities = {}
                maximum_competences = {}
                for index, skill in enumerate(skills):
                    competence = competences[skill]
                    rate = learning_rates[skill]
                    group = identities.get(skill, skill)
                    limit = limits.get(group, 0)
                    maximum_competences[skill] = min(1.0, competence + max(0.0, rate) * limit)
                    probability = model.addVar(lb=0, ub=1, name=f"p_{index}")
                    if limit:
                        lower = min(competence, competence + rate * limit)
                        upper = max(competence, competence + rate * limit)
                        raw = model.addVar(lb=lower, ub=upper)
                        model.addConstr(raw == competence + rate * counts[group])
                        if lower < 0:
                            capped = model.addVar(lb=lower, ub=min(1.0, upper))
                            model.addGenConstrMin(capped, [raw], constant=1.0)
                            model.addGenConstrMax(probability, [capped], constant=0.0)
                        else:
                            model.addConstr(probability <= raw)
                    else:
                        model.addConstr(probability == competence)
                    probabilities[skill] = probability
                spend = gp.quicksum(cost * counts[skill] for skill, cost in costs.items())
                model.addConstr(spend <= max(0, budget), name="practice_budget")
                states = set(mdp.initial_probabilities) | set(mdp.terminal_rewards)
                for edge in mdp.transitions:
                    states.update([edge.state, edge.successor])
                ranks = {
                    state: model.addVar(lb=0, ub=len(states), name=f"rank_{state}")
                    for state in states
                    if state not in mdp.terminal_rewards
                }
                outgoing: dict[int, list[Any]] = {state: [] for state in ranks}
                incoming: dict[int, list[Any]] = {state: [] for state in states}
                selections: dict[int, list[Any]] = {state: [] for state in ranks}
                for index, edge in enumerate(mdp.transitions):
                    if edge.state in mdp.terminal_rewards:
                        continue
                    selected = model.addVar(vtype=gp.GRB.BINARY, name=f"policy_{index}")
                    occupancy = model.addVar(lb=0, ub=1, name=f"mu_{index}")
                    model.addConstr(occupancy <= selected)
                    model.addConstr(
                        ranks[edge.state]
                        >= ranks.get(edge.successor, 0) + 1 - (len(states) + 1) * (1 - selected)
                    )
                    outgoing[edge.state].append(occupancy)
                    selections[edge.state].append(selected)
                    incoming[edge.successor].append(
                        mdp.discount * probabilities[edge.skill] * occupancy
                    )
                for state in ranks:
                    discard = model.addVar(lb=0, ub=1, name=f"discard_{state}")
                    model.addConstr(gp.quicksum(selections[state]) <= 1)
                    model.addQConstr(
                        gp.quicksum(outgoing[state]) + discard
                        == mdp.initial_probabilities.get(state, 0) + gp.quicksum(incoming[state])
                    )
                reward = model.addVar(lb=0, ub=max(mdp.terminal_rewards.values(), default=0))
                model.addQConstr(
                    reward
                    == gp.quicksum(
                        value
                        * (mdp.initial_probabilities.get(state, 0) + gp.quicksum(incoming[state]))
                        for state, value in mdp.terminal_rewards.items()
                    )
                )
                model.setObjective(reward, gp.GRB.MAXIMIZE)
                model.optimize()
                DpGurobiAllocator._require_optimal(model=model, phase="task_value")
                allocation = {skill: int(round(counts[skill].X)) for skill in costs}
                value = DpAllocator.predicted_success(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    allocation=allocation,
                    groups=groups,
                )
                theoretical_bound, _ = mdp.solve(competences=maximum_competences)
                bound = min(model.ObjBound, theoretical_bound)
                spent = sum(costs[skill] * count for skill, count in allocation.items())
                slack = DpGurobiAllocator.CERTIFICATION_TOLERANCE
                if spent > max(0, budget) + slack or bound - value > slack:
                    return {
                        "phase": "primary_independent_verification",
                        "objective": value,
                        "solver_objective": model.ObjVal,
                        "objective_bound": bound,
                        "spent": spent,
                        "budget": budget,
                        "max_violation": model.MaxVio,
                        "numeric_focus": numeric_focus,
                    }
                return DpAllocationResult(
                    allocation=allocation,
                    solver="gurobi",
                    status="OPTIMAL",
                    objective=value,
                    objective_bound=bound,
                    relative_gap=max(0, bound - value) / max(abs(value), 1e-12),
                    absolute_gap=max(0, bound - value),
                    runtime_seconds=0.0,
                    max_violation=model.MaxVio,
                )

    @staticmethod
    def _require_optimal(*, model: Any, phase: str) -> None:
        import gurobipy as gp

        if model.Status != gp.GRB.OPTIMAL:
            raise DpOptimizationError(
                diagnostics={
                    "phase": phase,
                    "status": model.Status,
                    "objective": model.ObjVal if model.SolCount else None,
                    "objective_bound": model.ObjBound,
                    "relative_gap": model.MIPGap if model.SolCount else None,
                    "runtime_seconds": model.Runtime,
                }
            )
