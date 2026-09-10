"""Iteration-bounded best-first search through a belief-space determinization.

The traversal follows Algorithm 3 in Curtis et al. (2025), while leaving the
heuristic domain-independent:
https://drive.google.com/file/d/1_yxDfN1BDxr7jvdZEnyZJcUyP1QnyLRr/view?pli=1
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Generic

import numpy as np
from pydantic import BaseModel, ConfigDict

from .planner import BeliefSpacePlanner
from .types.determinized import DeterminizedSearchNode
from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


class _QueueEntry(
    BaseModel,
    Generic[EnvironmentStateT, BeliefStateT, ActionT],
):
    """One entry in Algorithm 3's ``Open`` priority queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    priority: float
    sequence: int
    path_cost: float
    environment_state: EnvironmentStateT
    belief_state: BeliefStateT
    summed_cost: float
    depth: int
    key: object
    stop_value: float
    first_action: ActionT | StopAction

    def __lt__(self, other: object, /) -> bool:  # noqa: PLR0917
        if not isinstance(other, _QueueEntry):
            return NotImplemented
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class DeterminizedAStarPlanner(
    BeliefSpacePlanner[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]
):
    """Algorithm-3 search using the module's generic heuristic."""

    name = "determinized_astar"

    def __init__(
        self,
        *,
        max_iterations: int,
        seed: int,
    ) -> None:
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

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
        # deliberately does not bound this iteration-budgeted planner.
        del horizon
        return solve_belief_space_determinized_astar(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            model=model,
            max_iterations=self.max_iterations,
            rng=self.rng,
            num_samples=num_samples,
            trace=trace,
        )


def solve_belief_space_determinized_astar(
    *,
    environment_state: EnvironmentStateT,
    summed_cost: float,
    belief_state: BeliefStateT,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    max_iterations: int,
    rng: np.random.Generator,
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, ActionT | StopAction]:
    """Search one sampled deterministic successor per applicable action.

    Each unique cache-key/action edge is expanded at most once and samples one
    successor from the model's complete outcome distribution. A fixed seed and
    fixed traversal order are reproducible; changing the traversal order can
    assign random draws to different edges. Nodes are ordered by accumulated
    objective loss plus ``heuristic``. As in Algorithm 3, ``max_iterations``
    bounds the number of priority-queue iterations (pops), not search depth.

    Additional model sampling remains controlled by the experiment's
    master-seeded model. The iteration limit bounds cyclic search spaces without
    imposing a depth cutoff or wall-clock stopping condition.
    """
    assert max_iterations >= 1, "max_iterations must be positive"
    assert num_samples >= 1, "num_samples must be positive"
    assert math.isfinite(summed_cost) and summed_cost >= 0, (
        "summed_cost must be finite and non-negative"
    )
    started_at = time.perf_counter()

    root_value = _stop_value(
        model=model, belief_state=belief_state, summed_cost=summed_cost, num_samples=num_samples
    )
    best_value = root_value
    best_action: ActionT | StopAction = STOP_ACTION
    root_values: dict[ActionT, float] = {}
    # Algorithm 3: Open <- {(b0, g=0)}.
    frontier: list[_QueueEntry[EnvironmentStateT, BeliefStateT, ActionT]] = []
    root_key = model.search_cache_key(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        # Determinized search is compute-bounded, not horizon-bounded. The
        # model protocol still accepts a remaining-horizon discriminator for
        # exact expectimax; zero is the canonical depth-independent sentinel.
        horizon=0,
    )
    # Our sampled stopping objective J(C, b), cached per belief-space node.
    values_by_key = {root_key: root_value}
    # Algorithm 3: Closed.
    expanded_keys: set[object] = set()
    # Algorithm 3: Cost <- {b0: 0}.
    cost_by_key = {root_key: 0.0}
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
    iterations = 0
    termination_reason = "frontier_exhausted"

    if trace is not None:
        trace.record(event="stop_value", node=0, value=root_value)

    if root_value != -math.inf:
        root_priority = _priority_for(
            environment_state=environment_state,
            belief_state=belief_state,
            summed_cost=summed_cost,
            depth=0,
            value=root_value,
            path_cost_so_far=0.0,
        )
        heapq.heappush(
            frontier,
            _QueueEntry(
                priority=root_priority,
                sequence=sequence,
                path_cost=0.0,
                environment_state=environment_state,
                belief_state=belief_state,
                summed_cost=summed_cost,
                depth=0,
                key=root_key,
                stop_value=root_value,
                first_action=STOP_ACTION,
            ),
        )
        max_frontier_size = 1

    # A hard-budget G(C, theta) can make J(C, b) negative infinity at the root.
    # Since execution costs are non-negative, deeper nodes cannot restore feasibility.
    # Algorithm 3: while Open != empty and iterations < H.
    while frontier and iterations < max_iterations:
        current = heapq.heappop(frontier)
        iterations += 1
        traversed_nodes += 1
        # Algorithm 3, lines 5 and 9: pop lowest(Open), then skip Closed nodes.
        if current.key in expanded_keys:
            continue
        # Algorithm 3, line 12: add b to Closed.
        expanded_keys.add(current.key)
        expanded_nodes += 1
        max_depth_reached = max(max_depth_reached, current.depth)

        # Algorithm 3: for action a applicable in belief b.
        for action in model.get_valid_actions(environment_state=current.environment_state):
            action_transitions_evaluated += 1
            # Algorithm 3, lines 14-15: draw next states and observations. The
            # domain model exposes their joint distribution; determinization
            # samples one joint outcome for this edge.
            outcomes = model.transition_outcomes(
                environment_state=current.environment_state,
                practice_action=action,
                belief_state=current.belief_state,
            )
            chance_outcomes_enumerated += len(outcomes)
            sampled_outcome = _sample_outcome(rng=rng, outcomes=outcomes, action=action)
            next_environment, sampled_cost, _outcome_probability = sampled_outcome
            next_cost = current.summed_cost + sampled_cost
            next_depth = current.depth + 1
            # Algorithm 3, line 16: b' <- Branch(b, a, o).
            next_belief = model.update_belief_state(
                belief_state=current.belief_state,
                environment_state=current.environment_state,
                potential_next_environment_state=next_environment,
                practice_action=action,
            )
            generated_successors += 1
            first_action = action if current.depth == 0 else current.first_action
            first_actions = {action} if current.depth == 0 else provenance_by_key[current.key]
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
                value = _stop_value(
                    model=model,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
                )
                values_by_key[key] = value
                provenance_by_key[key] = set()
                children_by_key[key] = set()
            children_by_key[current.key].add(key)
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
            # Algorithm 3, lines 17-19: update g'. Our model's J(C, b)
            # already combines expected deployment value and action cost, so
            # J(current) - J(child) is the corresponding incremental search
            # cost. These differences telescope along a path.
            next_path_cost = current.path_cost + current.stop_value - value
            if value > best_value:
                best_value = value
                best_action = first_action
            # Algorithm 3, lines 20-22: insert b' only for a newly discovered
            # or strictly cheaper path.
            previous_path_cost = cost_by_key.get(key)
            if previous_path_cost is not None and previous_path_cost <= next_path_cost:
                continue
            cost_by_key[key] = next_path_cost
            sequence += 1
            heapq.heappush(
                frontier,
                _QueueEntry(
                    priority=_priority_for(
                        environment_state=next_environment,
                        belief_state=next_belief,
                        summed_cost=next_cost,
                        depth=next_depth,
                        value=value,
                        path_cost_so_far=next_path_cost,
                    ),
                    sequence=sequence,
                    path_cost=next_path_cost,
                    environment_state=next_environment,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    depth=next_depth,
                    key=key,
                    stop_value=value,
                    first_action=first_action,
                ),
            )
            max_frontier_size = max(max_frontier_size, len(frontier))

    if frontier and iterations >= max_iterations:
        termination_reason = "iteration_budget"

    if trace is not None:
        for action, value in root_values.items():
            trace.record(
                event="action_value",
                node=0,
                action=action.model_dump(mode="json", fallback=str),
                value=value,
            )
        # Algorithm 3, line 26: the minimum-g node is equivalent here to the
        # maximum-J node because the incremental costs telescope from J(b0).
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
            iterations=iterations,
            max_iterations=max_iterations,
            stop_value_evaluations=len(values_by_key),
            search_elapsed_seconds=time.perf_counter() - started_at,
            termination_reason=termination_reason,
        )
    return best_value, best_action


def heuristic(*, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]) -> float:
    """Estimate remaining cost; this generic default adds no domain knowledge."""
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
    environment_state: EnvironmentStateT,
    belief_state: BeliefStateT,
    summed_cost: float,
    depth: int,
    value: float,
    path_cost_so_far: float,
) -> float:
    """Build and score the node view used by :func:`heuristic`."""
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
