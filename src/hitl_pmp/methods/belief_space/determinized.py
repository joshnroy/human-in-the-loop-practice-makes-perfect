"""Bounded best-first search through a sampled belief-space determinization."""

from __future__ import annotations

import heapq
import math

import numpy as np

from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


def solve_belief_space_determinized(
    *,
    environment_state: EnvironmentStateT,
    summed_cost: float,
    belief_state: BeliefStateT,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    max_expansions: int,
    seed: int,
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, ActionT | StopAction]:
    """Search one sampled deterministic successor per applicable action.

    Nodes are ordered by their current stop value. The expansion budget, rather
    than a depth limit, makes cyclic or improving belief graphs finite. A fresh
    seeded RNG is owned by each call; callers therefore get reproducible replanning
    after every real action without changing the model, beliefs, or objective ``G``.
    """
    assert max_expansions >= 0, "max_expansions must be non-negative"
    assert num_samples >= 1, "num_samples must be positive"
    assert math.isfinite(summed_cost) and summed_cost >= 0, (
        "summed_cost must be finite and non-negative"
    )
    rng = np.random.default_rng(seed)

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
    best_action: ActionT | StopAction = STOP_ACTION
    root_values: dict[ActionT, float] = {}
    frontier: list[tuple[float, int, EnvironmentStateT, BeliefStateT, float, object, float]] = []
    root_key = model.search_cache_key(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=0,
    )
    values_by_key = {root_key: root_value}
    provenance_by_key: dict[object, list[ActionT]] = {root_key: []}
    children_by_key: dict[object, list[object]] = {root_key: []}
    expanded_nodes = 0
    generated_nodes = 0
    merged_nodes = 0
    action_evaluations = 0
    chance_outcomes = 0
    sequence = 0

    def add_provenance(*, key: object, actions: list[ActionT]) -> None:
        """Credit cached descendants to every root action that reaches them."""
        nonlocal best_action, best_value
        pending = [(key, action) for action in actions]
        while pending:
            current_key, action = pending.pop()
            provenance = provenance_by_key[current_key]
            if action in provenance:
                continue
            provenance.append(action)
            value = values_by_key[current_key]
            root_values[action] = max(root_values.get(action, -math.inf), value)
            if value > best_value:
                best_value = value
                best_action = action
            pending.extend((child_key, action) for child_key in children_by_key[current_key])

    if trace is not None:
        trace.record(event="stop_value", node=0, value=root_value)

    # Once a hard-budget G rejects the root, non-negative action costs cannot
    # restore feasibility, so skip model traversal just as expectimax does.
    while frontier or (expanded_nodes == 0 and root_value != -math.inf):
        if expanded_nodes >= max_expansions:
            break
        if frontier:
            (
                _priority,
                _sequence,
                current_environment,
                current_belief,
                current_cost,
                current_key,
                _value,
            ) = heapq.heappop(frontier)
            current_first_actions = provenance_by_key[current_key]
        else:
            current_environment = environment_state
            current_belief = belief_state
            current_cost = summed_cost
            current_key = root_key
            current_first_actions = []
        expanded_nodes += 1

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
            next_environment, sampled_cost, _probability = outcomes[outcome_index]
            assert math.isfinite(sampled_cost) and sampled_cost >= 0, (
                "sampled_cost must be finite and non-negative"
            )
            next_cost = current_cost + sampled_cost
            next_belief = model.update_belief_state(
                belief_state=current_belief,
                environment_state=current_environment,
                potential_next_environment_state=next_environment,
                practice_action=action,
            )
            generated_nodes += 1
            first_actions = [action] if current_key == root_key else current_first_actions
            key = model.search_cache_key(
                environment_state=next_environment,
                summed_cost=next_cost,
                belief_state=next_belief,
                horizon=0,
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
            if cached_value is not None:
                continue
            sequence += 1
            heapq.heappush(
                frontier,
                (
                    -value,
                    sequence,
                    next_environment,
                    next_belief,
                    next_cost,
                    key,
                    value,
                ),
            )

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
            action_evaluations=action_evaluations,
            chance_outcomes=chance_outcomes,
            frontier_nodes=len(frontier),
            max_expansions=max_expansions,
        )
    return best_value, best_action
