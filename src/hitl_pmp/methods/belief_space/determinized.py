"""Bounded best-first search through a sampled belief-space determinization.

The traversal follows the open/closed/cost structure of Algorithm 3 in Curtis
et al. (2025), while leaving the heuristic domain-independent and injectable:
https://proceedings.mlr.press/v305/curtis25a.html
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .planner import BeliefSpacePlanner
from .types.determinized import DeterminizedHeuristic, DeterminizedSearchNode
from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


class DeterminizedAStarPlanner(
    BeliefSpacePlanner[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]
):
    """Compute-bounded Algorithm-3 search with an injected generic heuristic."""

    name = "determinized_astar"

    def __init__(
        self,
        *,
        max_stop_value_evaluations: int | None = None,
        seed: int,
        max_seconds: float | None = None,
        heuristic: DeterminizedHeuristic[EnvironmentStateT, BeliefStateT] | None = None,
    ) -> None:
        self.max_stop_value_evaluations = max_stop_value_evaluations
        self.seed = seed
        self.max_seconds = max_seconds
        self.heuristic = heuristic or zero_heuristic

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
        # ``horizon`` belongs to the shared interface for exact expectimax. It
        # deliberately does not bound this compute-budgeted planner.
        del horizon
        return solve_belief_space_determinized_astar(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            model=model,
            max_stop_value_evaluations=self.max_stop_value_evaluations,
            max_seconds=self.max_seconds,
            seed=self.seed,
            heuristic=self.heuristic,
            num_samples=num_samples,
            trace=trace,
        )


def solve_belief_space_determinized_astar(
    *,
    environment_state: EnvironmentStateT,
    summed_cost: float,
    belief_state: BeliefStateT,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    max_stop_value_evaluations: int | None = None,
    max_seconds: float | None = None,
    seed: int,
    heuristic: DeterminizedHeuristic[EnvironmentStateT, BeliefStateT] | None = None,
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, ActionT | StopAction]:
    """Search one sampled deterministic successor per applicable action.

    Each unique cache-key/action edge is expanded at most once and samples one
    successor from the model's complete outcome distribution. A fixed seed and
    fixed traversal order are reproducible; changing the traversal order can
    assign random draws to different edges. Nodes are ordered by accumulated
    objective loss plus the injected heuristic. Search ends at the stop-value
    evaluation or wall-clock budget, whichever comes first.

    Additional model sampling remains controlled by the experiment's
    master-seeded model. At least one compute budget is required, so cyclic
    search spaces do not require a separate depth cutoff.
    """
    assert max_stop_value_evaluations is None or max_stop_value_evaluations >= 1, (
        "max_stop_value_evaluations must be positive"
    )
    assert num_samples >= 1, "num_samples must be positive"
    assert max_seconds is None or (math.isfinite(max_seconds) and max_seconds >= 0.0), (
        "max_seconds must be finite and non-negative"
    )
    assert max_stop_value_evaluations is not None or max_seconds is not None, (
        "at least one compute budget is required"
    )
    assert math.isfinite(summed_cost) and summed_cost >= 0, (
        "summed_cost must be finite and non-negative"
    )
    heuristic = heuristic or zero_heuristic
    rng = np.random.default_rng(seed)
    started_at = time.perf_counter()

    root_value = _stop_value(
        model=model, belief_state=belief_state, summed_cost=summed_cost, num_samples=num_samples
    )
    best_value = root_value
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
        # Determinized search is compute-bounded, not horizon-bounded. The
        # model protocol still accepts a remaining-horizon discriminator for
        # exact expectimax; zero is the canonical depth-independent sentinel.
        horizon=0,
    )
    values_by_key = {root_key: root_value}
    queued_keys = {root_key}
    expanded_keys: set[object] = set()
    provenance_by_key: dict[object, set[ActionT]] = {root_key: set()}
    children_by_key: dict[object, set[object]] = {root_key: set()}
    expanded_nodes = 0
    traversed_nodes = 0
    generated_successors = 0
    merged_nodes = 0
    action_transitions_evaluated = 0
    chance_outcomes_enumerated = 0
    sequence = 0
    max_frontier_size = 0
    max_depth_reached = 0
    termination_reason = "frontier_exhausted"

    if trace is not None:
        trace.record(event="stop_value", node=0, value=root_value)

    if root_value != -math.inf:
        root_priority = _priority_for(
            heuristic=heuristic,
            environment_state=environment_state,
            belief_state=belief_state,
            summed_cost=summed_cost,
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

    # A hard-budget G(C, theta) can make J(C, b) negative infinity at the root.
    # Since execution costs are non-negative, deeper nodes cannot restore feasibility.
    while frontier:
        if max_seconds is not None and time.perf_counter() - started_at >= max_seconds:
            termination_reason = "time_budget"
            break
        if (
            max_stop_value_evaluations is not None
            and len(values_by_key) >= max_stop_value_evaluations
        ):
            termination_reason = "stop_value_evaluation_budget"
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
        traversed_nodes += 1
        queued_keys.discard(current_key)
        if current_key in expanded_keys:
            continue
        expanded_keys.add(current_key)
        expanded_nodes += 1
        max_depth_reached = max(max_depth_reached, current_depth)

        for action in model.get_valid_actions(environment_state=current_environment):
            if max_seconds is not None and time.perf_counter() - started_at >= max_seconds:
                termination_reason = "time_budget"
                break
            if (
                max_stop_value_evaluations is not None
                and len(values_by_key) >= max_stop_value_evaluations
            ):
                termination_reason = "stop_value_evaluation_budget"
                break
            action_transitions_evaluated += 1
            outcomes = model.transition_outcomes(
                environment_state=current_environment,
                practice_action=action,
                belief_state=current_belief,
            )
            chance_outcomes_enumerated += len(outcomes)
            sampled_outcome = _sample_outcome(rng=rng, outcomes=outcomes, action=action)
            next_environment, sampled_cost, _outcome_probability = sampled_outcome
            next_cost = current_cost + sampled_cost
            next_depth = current_depth + 1
            next_belief = model.update_belief_state(
                belief_state=current_belief,
                environment_state=current_environment,
                potential_next_environment_state=next_environment,
                practice_action=action,
            )
            generated_successors += 1
            first_action = action if current_depth == 0 else current_first_action
            first_actions = {action} if current_depth == 0 else provenance_by_key[current_key]
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
                if max_seconds is not None and time.perf_counter() - started_at >= max_seconds:
                    termination_reason = "time_budget"
                    break
                value = _stop_value(
                    model=model,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
                )
                values_by_key[key] = value
                provenance_by_key[key] = set()
                children_by_key[key] = set()
            children_by_key[current_key].add(key)
            _add_provenance(
                key=key,
                actions=first_actions,
                provenance_by_key=provenance_by_key,
                root_values=root_values,
                values_by_key=values_by_key,
                children_by_key=children_by_key,
            )
            if value == -math.inf:
                continue
            # Objective differences telescope along the path, converting this
            # maximization problem to A*'s lowest-cost-first convention.
            next_path_cost = current_path_cost + current_value - value
            if value > best_value:
                best_value = value
                best_action = first_action
            if key in expanded_keys or key in queued_keys:
                continue
            queued_keys.add(key)
            sequence += 1
            heapq.heappush(
                frontier,
                (
                    _priority_for(
                        heuristic=heuristic,
                        environment_state=next_environment,
                        belief_state=next_belief,
                        summed_cost=next_cost,
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

        if termination_reason in {"time_budget", "stop_value_evaluation_budget"}:
            break

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
            solver="determinized_astar",
            expanded_nodes=expanded_nodes,
            traversed_nodes=traversed_nodes,
            generated_successors=generated_successors,
            unique_nodes=len(values_by_key),
            merged_nodes=merged_nodes,
            action_transitions_evaluated=action_transitions_evaluated,
            chance_outcomes_enumerated=chance_outcomes_enumerated,
            frontier_nodes=len(frontier),
            max_frontier_size=max_frontier_size,
            max_depth_reached=max_depth_reached,
            stop_value_evaluations=len(values_by_key),
            max_stop_value_evaluations=max_stop_value_evaluations,
            max_seconds=max_seconds,
            search_elapsed_seconds=(elapsed_seconds := time.perf_counter() - started_at),
            time_budget_overshoot_seconds=(
                max(0.0, elapsed_seconds - max_seconds) if max_seconds is not None else None
            ),
            termination_reason=termination_reason,
        )
    return best_value, best_action


def zero_heuristic(*, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]) -> float:
    """Neutral default: order the frontier only by accumulated path cost."""
    del node
    return 0.0


def _stop_value(
    *,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    belief_state: BeliefStateT,
    summed_cost: float,
    num_samples: int,
) -> float:
    """Monte Carlo estimate of the paper's stopping objective ``J(C, b)``.

    ``model.G`` evaluates ``G(C, theta)`` for one sampled latent model; the
    mean below estimates its expectation under the belief ``b``.
    """
    policy_values = model.sample_policy_values_from_belief(
        belief_state=belief_state, num_samples=num_samples
    )
    assert len(policy_values) == num_samples
    values = np.fromiter(
        (model.G(policy_value=float(value), summed_cost=summed_cost) for value in policy_values),
        dtype=np.float64,
        count=num_samples,
    )
    assert all(not math.isnan(value) and value != math.inf for value in values), (
        "stop values must be finite or negative infinity"
    )
    return float(np.mean(values))


def _priority_for(
    *,
    heuristic: DeterminizedHeuristic[EnvironmentStateT, BeliefStateT],
    environment_state: EnvironmentStateT,
    belief_state: BeliefStateT,
    summed_cost: float,
    depth: int,
    value: float,
    path_cost_so_far: float,
) -> float:
    """Build and score the public node view used by an injected heuristic."""
    estimate = heuristic(
        node=DeterminizedSearchNode(
            environment_state=environment_state,
            belief_state=belief_state,
            summed_cost=summed_cost,
            depth=depth,
            stop_value=value,
            path_cost=path_cost_so_far,
        )
    )
    assert math.isfinite(estimate), "heuristic must return a finite value"
    return path_cost_so_far + estimate


def _sample_outcome(
    *,
    rng: np.random.Generator,
    outcomes: list[tuple[EnvironmentStateT, float, float]],
    action: ActionT,
) -> tuple[EnvironmentStateT, float, float]:
    """Validate a chance distribution and sample one determinized successor."""
    assert outcomes, f"action {action!r} has no chance outcomes"
    probabilities = np.fromiter((outcome[2] for outcome in outcomes), dtype=np.float64)
    assert np.all(np.isfinite(probabilities)) and np.all(probabilities > 0.0), (
        "chance probabilities must be finite and positive"
    )
    total_probability = float(np.sum(probabilities))
    assert math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12), (
        f"chance probabilities sum to {total_probability}, not 1"
    )
    outcome_index = int(rng.choice(len(outcomes), p=probabilities / total_probability))
    outcome = outcomes[outcome_index]
    assert math.isfinite(outcome[1]) and outcome[1] >= 0, (
        "sampled_cost must be finite and non-negative"
    )
    return outcome


def _add_provenance(
    *,
    key: object,
    actions: set[ActionT],
    provenance_by_key: dict[object, set[ActionT]],
    root_values: dict[ActionT, float],
    values_by_key: dict[object, float],
    children_by_key: dict[object, set[object]],
) -> None:
    """Credit a shared descendant to every root action that reaches it."""
    pending = [(key, action) for action in actions]
    while pending:
        current_key, action = pending.pop()
        provenance = provenance_by_key[current_key]
        if action in provenance:
            continue
        provenance.add(action)
        root_values[action] = max(root_values.get(action, -math.inf), values_by_key[current_key])
        pending.extend((child_key, action) for child_key in children_by_key[current_key])
