"""Joint integer training allocation and task-policy optimization using Gurobi.

The bilinear terms couple predicted skill success with task-policy occupancy.
An acyclic deterministic policy suffices for nonnegative terminal rewards and
success-or-absorbing-failure dynamics. Policy ranks exclude circulation, giving
valid unit occupancy bounds even for the undiscounted success objective.

Task value is nondecreasing in every success probability. When an affine forecast
stays nonnegative, p <= raw and p <= 1 preserve the maximum task value and the
least-cost allocations meeting a task-value floor. The probabilities themselves
need not equal the clipped forecasts: increasing them to those forecasts cannot
reduce task value or change spend. Forecasts crossing zero still need the lower
clip, including declining siblings of an improving shared group. This avoids
numerically fragile MIN equalities near upper saturation without changing either
optimization objective or the independent certification checks.
"""

import math
import time
from collections.abc import Hashable, Mapping
from typing import Any, Generic, TypeVar

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


class DpOptimizationError(RuntimeError):
    def __init__(self, *, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"DP requires certified optimization; no fallback: {diagnostics}")


class DpGurobiAllocator:
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
        try:
            import gurobipy as gp
        except ImportError as error:
            raise RuntimeError("DP's Gurobi backend needs the optional 'dp' dependency") from error

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
        limits = DpAllocator._allocation_limits(
            mdp=mdp,
            competences=competences,
            learning_rates=learning_rates,
            costs=costs,
            budget=budget,
            groups=groups,
        )
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", 0)
            environment.start()
            with gp.Model("deliberate-practice", env=environment) as model:
                model.Params.NonConvex = 2
                model.Params.Threads = 1
                model.Params.TimeLimit = time_limit
                model.Params.MIPGap = 1e-9
                model.Params.MIPGapAbs = 1e-9
                model.Params.FeasibilityTol = 1e-9
                model.Params.IntFeasTol = 1e-9
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
                primary_allocation = {skill: int(round(counts[skill].X)) for skill in costs}
                value = DpAllocator.predicted_success(
                    mdp=mdp,
                    competences=competences,
                    learning_rates=learning_rates,
                    allocation=primary_allocation,
                    groups=groups,
                )
                theoretical_bound, _ = mdp.solve(competences=maximum_competences)
                bound = min(model.ObjBound, theoretical_bound)
                primary_spent = sum(
                    costs[skill] * count for skill, count in primary_allocation.items()
                )
                if primary_spent > max(0, budget) + 1e-7 or bound - value > 1e-7:
                    raise DpOptimizationError(
                        diagnostics={
                            "phase": "primary_independent_verification",
                            "objective": value,
                            "solver_objective": model.ObjVal,
                            "objective_bound": bound,
                            "spent": primary_spent,
                            "budget": budget,
                        }
                    )
                minimum_task_value = value - 1e-8
                model.addConstr(reward >= minimum_task_value, name="preserve_optimal_task_value")
                model.setObjective(spend, gp.GRB.MINIMIZE)
                rejected_allocations: set[tuple[int, ...]] = set()
                while True:
                    remaining = time_limit - (time.monotonic() - started)
                    if remaining <= 0:
                        raise DpOptimizationError(
                            diagnostics={
                                "phase": "least_cost_tie_break",
                                "reason": "time_limit_before_certification",
                                "tie_break_retries": len(rejected_allocations),
                            }
                        )
                    model.Params.TimeLimit = remaining
                    model.optimize()
                    DpGurobiAllocator._require_optimal(model=model, phase="least_cost_tie_break")
                    allocation = {skill: int(round(counts[skill].X)) for skill in costs}
                    actual = DpAllocator.predicted_success(
                        mdp=mdp,
                        competences=competences,
                        learning_rates=learning_rates,
                        allocation=allocation,
                        groups=groups,
                    )
                    spent = sum(costs[skill] * count for skill, count in allocation.items())
                    over_budget = spent > max(0, budget) + 1e-7
                    preserves_task_value = actual >= minimum_task_value
                    if not over_budget and preserves_task_value and bound - actual <= 1e-7:
                        if time.monotonic() - started >= time_limit:
                            raise DpOptimizationError(
                                diagnostics={
                                    "phase": "least_cost_tie_break",
                                    "reason": "time_limit_before_certification",
                                    "tie_break_retries": len(rejected_allocations),
                                }
                            )
                        break
                    diagnostics = {
                        "phase": "independent_verification",
                        "objective": actual,
                        "solver_objective": value,
                        "objective_bound": bound,
                        "spent": spent,
                        "budget": budget,
                        "minimum_task_value": minimum_task_value,
                        "tie_break_retries": len(rejected_allocations),
                    }
                    signature = tuple(allocation.values())
                    if (
                        not over_budget and preserves_task_value
                    ) or signature in rejected_allocations:
                        raise DpOptimizationError(diagnostics=diagnostics)
                    rejected_allocations.add(signature)
                    DpGurobiAllocator._exclude_allocation(
                        model=model,
                        counts=counts,
                        allocation=allocation,
                        index=len(rejected_allocations),
                    )
                return DpAllocationResult(
                    allocation=allocation,
                    solver="gurobi",
                    status="OPTIMAL",
                    objective=actual,
                    objective_bound=bound,
                    relative_gap=max(0, bound - actual) / max(abs(actual), 1e-12),
                    absolute_gap=max(0, bound - actual),
                    tie_break_retries=len(rejected_allocations),
                    runtime_seconds=time.monotonic() - started,
                )

    @staticmethod
    def _exclude_allocation(
        *,
        model: Any,
        counts: Mapping[SkillKey, Any],
        allocation: Mapping[SkillKey, int],
        index: int,
    ) -> None:
        import gurobipy as gp

        alternatives = []
        for position, (skill, count) in enumerate(allocation.items()):
            variable = counts[skill]
            if count > variable.LB:
                lower = model.addVar(vtype=gp.GRB.BINARY, name=f"reject_{index}_{position}_lower")
                model.addGenConstrIndicator(lower, True, variable <= count - 1)
                alternatives.append(lower)
            if count < variable.UB:
                higher = model.addVar(vtype=gp.GRB.BINARY, name=f"reject_{index}_{position}_higher")
                model.addGenConstrIndicator(higher, True, variable >= count + 1)
                alternatives.append(higher)
        model.addConstr(gp.quicksum(alternatives) >= 1, name=f"reject_allocation_{index}")

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
