"""Bounded best-first search through a sampled belief-space determinization.

The traversal follows the open/closed/cost structure of Algorithm 3 in Curtis
et al. (2025), while leaving the heuristic domain-independent and injectable:
https://proceedings.mlr.press/v305/curtis25a.html
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Generic, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from .planner import BeliefSpacePlanner
from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


class DeterminizedSearchNode(BaseModel, Generic[EnvironmentStateT, BeliefStateT]):
    """Generic node information exposed to an injected A* heuristic."""

    model_config = ConfigDict(frozen=True)

    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    stop_value: float
    path_cost: float


class DeterminizedHeuristic(Protocol[EnvironmentStateT, BeliefStateT]):
    """Additional estimated cost-to-go used to order the A* frontier."""

    def __call__(
        self, *, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]
    ) -> float: ...


def zero_heuristic(*, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]) -> float:
    """Neutral default: order the frontier only by accumulated path cost."""
    del node
    return 0.0


class DeterminizedPathCost(Protocol[EnvironmentStateT, BeliefStateT, ActionT]):
    """Incremental cost used to relax paths through the determinized graph."""

    def __call__(
        self,
        *,
        parent: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT],
        child: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT],
        action: ActionT,
        outcome_probability: float,
        sampled_cost: float,
    ) -> float: ...


def objective_delta_path_cost(
    *,
    parent: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT],
    child: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT],
    action: ActionT,
    outcome_probability: float,
    sampled_cost: float,
) -> float:
    """Convert improvement in the supplied objective ``G`` to A* path cost.

    The differences telescope, so minimizing accumulated cost selects the
    discovered node with the largest stop/deployment value. Domains may inject
    another additive path-cost model without changing the planner.
    """
    del action, outcome_probability, sampled_cost
    return parent.stop_value - child.stop_value


def solve_belief_space_determinized(
    *,
    environment_state: EnvironmentStateT,
    summed_cost: float,
    belief_state: BeliefStateT,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    horizon: int,
    max_expansions: int,
    max_seconds: float | None = None,
    seed: int,
    heuristic: DeterminizedHeuristic[EnvironmentStateT, BeliefStateT] = zero_heuristic,
    path_cost: DeterminizedPathCost[
        EnvironmentStateT, BeliefStateT, ActionT
    ] = objective_delta_path_cost,
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, ActionT | StopAction]:
    """Search one sampled deterministic successor per applicable action.

    Each action samples one successor from the model's complete outcome
    distribution. Nodes are ordered by ``-stop_value + heuristic(node)``: the
    negative sign converts this maximization problem to A*'s lowest-cost-first
    convention. The supplied seed controls chance-outcome determinization; any
    additional model sampling remains controlled by the experiment's master-seeded model.
    """
    assert max_expansions >= 0, "max_expansions must be non-negative"
    assert horizon >= 0, "horizon must be non-negative"
    assert num_samples >= 1, "num_samples must be positive"
    assert max_seconds is None or (math.isfinite(max_seconds) and max_seconds >= 0.0), (
        "max_seconds must be finite and non-negative"
    )
    assert math.isfinite(summed_cost) and summed_cost >= 0, (
        "summed_cost must be finite and non-negative"
    )
    rng = np.random.default_rng(seed)
    started_at = time.perf_counter()

    def stop_value(*, state: BeliefStateT, cost: float) -> float:
        policy_values = model.sample_policy_values_from_belief(
            belief_state=state, num_samples=num_samples
        )
        assert len(policy_values) == num_samples
        values = np.fromiter(
            (model.G(policy_value=float(value), summed_cost=cost) for value in policy_values),
            dtype=np.float64,
            count=num_samples,
        )
        assert all(not math.isnan(value) and value != math.inf for value in values), (
            "stop values must be finite or negative infinity"
        )
        return float(np.mean(values))

    root_value = stop_value(state=belief_state, cost=summed_cost)
    best_value = root_value
    best_path_cost = 0.0
    best_action: ActionT | StopAction = STOP_ACTION
    root_values: dict[ActionT, float] = {}
    frontier: list[
        tuple[
            float,
            int,
            float,
            EnvironmentStateT,
            BeliefStateT,
            float,
            int,
            object,
            float,
            ActionT | StopAction,
        ]
    ] = []
    root_key = model.search_cache_key(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=horizon,
    )
    values_by_key = {root_key: root_value}
    costs_by_key = {root_key: 0.0}
    closed_costs: dict[object, float] = {}
    provenance_by_key: dict[object, list[ActionT]] = {root_key: []}
    children_by_key: dict[object, list[object]] = {root_key: []}
    expanded_nodes = 0
    generated_nodes = 0
    merged_nodes = 0
    action_evaluations = 0
    chance_outcomes = 0
    sequence = 0
    max_frontier_size = 0
    max_depth_reached = 0
    reopened_nodes = 0
    termination_reason = "frontier_exhausted"

    def add_provenance(*, key: object, actions: list[ActionT]) -> None:
        """Credit a shared descendant to every root action that can reach it."""
        pending = [(key, action) for action in actions]
        while pending:
            current_key, action = pending.pop()
            provenance = provenance_by_key[current_key]
            if action in provenance:
                continue
            provenance.append(action)
            root_values[action] = max(
                root_values.get(action, -math.inf), values_by_key[current_key]
            )
            pending.extend((child_key, action) for child_key in children_by_key[current_key])

    def priority_for(
        *,
        state: EnvironmentStateT,
        belief: BeliefStateT,
        cost: float,
        depth: int,
        value: float,
        path_cost_so_far: float,
    ) -> float:
        estimate = heuristic(
            node=DeterminizedSearchNode(
                environment_state=state,
                belief_state=belief,
                summed_cost=cost,
                depth=depth,
                stop_value=value,
                path_cost=path_cost_so_far,
            )
        )
        assert math.isfinite(estimate), "heuristic must return a finite value"
        return path_cost_so_far + estimate

    if trace is not None:
        trace.record(event="stop_value", node=0, value=root_value)

    if root_value != -math.inf:
        root_priority = priority_for(
            state=environment_state,
            belief=belief_state,
            cost=summed_cost,
            depth=0,
            value=root_value,
            path_cost_so_far=0.0,
        )
        heapq.heappush(
            frontier,
            (
                root_priority,
                sequence,
                0.0,
                environment_state,
                belief_state,
                summed_cost,
                0,
                root_key,
                root_value,
                STOP_ACTION,
            ),
        )
        max_frontier_size = 1

    # A hard-budget G can reject the root with -inf. Since execution costs are
    # non-negative, deeper nodes cannot restore feasibility.
    while frontier:
        if expanded_nodes >= max_expansions:
            termination_reason = "expansion_budget"
            break
        if max_seconds is not None and time.perf_counter() - started_at >= max_seconds:
            termination_reason = "time_budget"
            break
        (
            _priority,
            _sequence,
            current_path_cost,
            current_environment,
            current_belief,
            current_cost,
            current_depth,
            current_key,
            current_value,
            current_first_action,
        ) = heapq.heappop(frontier)
        if current_path_cost != costs_by_key.get(current_key):
            continue
        closed_cost = closed_costs.get(current_key)
        if closed_cost is not None and closed_cost <= current_path_cost:
            continue
        closed_costs[current_key] = current_path_cost
        expanded_nodes += 1
        max_depth_reached = max(max_depth_reached, current_depth)

        if current_path_cost < best_path_cost:
            best_path_cost = current_path_cost
            best_value = current_value
            best_action = current_first_action

        if current_depth >= horizon:
            continue

        for action in model.get_valid_actions(environment_state=current_environment):
            action_evaluations += 1
            outcomes = model.transition_outcomes(
                environment_state=current_environment,
                practice_action=action,
                belief_state=current_belief,
            )
            assert outcomes, f"action {action!r} has no chance outcomes"
            chance_outcomes += len(outcomes)
            probabilities = np.fromiter((outcome[2] for outcome in outcomes), dtype=np.float64)
            assert np.all(np.isfinite(probabilities)) and np.all(probabilities > 0.0), (
                "chance probabilities must be finite and positive"
            )
            total_probability = float(np.sum(probabilities))
            assert math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12), (
                f"chance probabilities sum to {total_probability}, not 1"
            )
            outcome_index = int(rng.choice(len(outcomes), p=probabilities / total_probability))
            next_environment, sampled_cost, outcome_probability = outcomes[outcome_index]
            assert math.isfinite(sampled_cost) and sampled_cost >= 0, (
                "sampled_cost must be finite and non-negative"
            )
            next_cost = current_cost + sampled_cost
            next_depth = current_depth + 1
            next_belief = model.update_belief_state(
                belief_state=current_belief,
                environment_state=current_environment,
                potential_next_environment_state=next_environment,
                practice_action=action,
            )
            generated_nodes += 1
            first_action = action if current_depth == 0 else current_first_action
            first_actions = [action] if current_depth == 0 else provenance_by_key[current_key]
            key = model.search_cache_key(
                environment_state=next_environment,
                summed_cost=next_cost,
                belief_state=next_belief,
                horizon=horizon - next_depth,
            )
            cached_value = values_by_key.get(key)
            if cached_value is not None:
                merged_nodes += 1
                value = cached_value
            else:
                value = stop_value(state=next_belief, cost=next_cost)
                values_by_key[key] = value
                provenance_by_key[key] = []
                children_by_key[key] = []
            if key not in children_by_key[current_key]:
                children_by_key[current_key].append(key)
            add_provenance(key=key, actions=first_actions)
            if value == -math.inf:
                continue
            parent_node = DeterminizedSearchNode(
                environment_state=current_environment,
                belief_state=current_belief,
                summed_cost=current_cost,
                depth=current_depth,
                stop_value=current_value,
                path_cost=current_path_cost,
            )
            provisional_child = DeterminizedSearchNode(
                environment_state=next_environment,
                belief_state=next_belief,
                summed_cost=next_cost,
                depth=next_depth,
                stop_value=value,
                path_cost=current_path_cost,
            )
            incremental_cost = path_cost(
                parent=parent_node,
                child=provisional_child,
                action=action,
                outcome_probability=outcome_probability,
                sampled_cost=sampled_cost,
            )
            assert math.isfinite(incremental_cost), "path cost must be finite"
            next_path_cost = current_path_cost + incremental_cost
            if next_path_cost < best_path_cost:
                best_path_cost = next_path_cost
                best_value = value
                best_action = first_action
            previous_cost = costs_by_key.get(key, math.inf)
            if next_path_cost >= previous_cost:
                continue
            if key in closed_costs:
                reopened_nodes += 1
            costs_by_key[key] = next_path_cost
            sequence += 1
            heapq.heappush(
                frontier,
                (
                    priority_for(
                        state=next_environment,
                        belief=next_belief,
                        cost=next_cost,
                        depth=next_depth,
                        value=value,
                        path_cost_so_far=next_path_cost,
                    ),
                    sequence,
                    next_path_cost,
                    next_environment,
                    next_belief,
                    next_cost,
                    next_depth,
                    key,
                    value,
                    first_action,
                ),
            )
            max_frontier_size = max(max_frontier_size, len(frontier))

    if trace is not None:
        for action, value in root_values.items():
            trace.record(
                event="action_value",
                node=0,
                action=action.model_dump(mode="json", fallback=str),
                value=value,
            )
        trace.record(
            event="choice",
            node=0,
            action="STOP"
            if best_action == STOP_ACTION
            else best_action.model_dump(mode="json", fallback=str),
            value=best_value,
            reason="best_sampled_path_stop_wins_ties",
        )
        trace.record(
            event="search_summary",
            node=0,
            solver="determinized",
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            unique_nodes=len(values_by_key),
            merged_nodes=merged_nodes,
            reopened_nodes=reopened_nodes,
            action_evaluations=action_evaluations,
            chance_outcomes=chance_outcomes,
            frontier_nodes=len(frontier),
            max_frontier_size=max_frontier_size,
            max_depth_reached=max_depth_reached,
            horizon=horizon,
            max_expansions=max_expansions,
            max_seconds=max_seconds,
            elapsed_seconds=time.perf_counter() - started_at,
            termination_reason=termination_reason,
        )
    return best_value, best_action


class DeterminizedAStarPlanner(
    BeliefSpacePlanner[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]
):
    """Algorithm-3-style determinized search with an injected generic heuristic."""

    name = "determinized_astar"

    def __init__(
        self,
        *,
        max_expansions: int,
        seed: int,
        max_seconds: float | None = None,
        heuristic: DeterminizedHeuristic[EnvironmentStateT, BeliefStateT] = zero_heuristic,
        path_cost: DeterminizedPathCost[
            EnvironmentStateT, BeliefStateT, ActionT
        ] = objective_delta_path_cost,
    ) -> None:
        self.max_expansions = max_expansions
        self.seed = seed
        self.max_seconds = max_seconds
        self.heuristic = heuristic
        self.path_cost = path_cost

    def solve(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        num_samples: int = NUM_SAMPLES,
        trace: SearchTrace | None = None,
    ) -> tuple[float, ActionT | StopAction]:
        return solve_belief_space_determinized(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
            model=model,
            max_expansions=self.max_expansions,
            max_seconds=self.max_seconds,
            seed=self.seed,
            heuristic=self.heuristic,
            path_cost=self.path_cost,
            num_samples=num_samples,
            trace=trace,
        )
