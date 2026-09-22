"""Iteration-bounded best-first search through a belief-space determinization.

The traversal follows Algorithm 3 in Curtis et al. (2025), while leaving the
heuristic domain-independent:
https://drive.google.com/file/d/1_yxDfN1BDxr7jvdZEnyZJcUyP1QnyLRr/view?pli=1
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .planner import BeliefSpacePlanner
from .types.determinized import (
    DeterminizedNodeCacheInfo,
    DeterminizedNodeDiagnosticInfo,
    DeterminizedPathRecoveryInfo,
    DeterminizedSearchBelief,
    DeterminizedSearchDiagnostics,
    DeterminizedSearchNode,
    DeterminizedSearchQueueEntry,
)
from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


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
        observation_probability_weight: float = 0.1,
    ) -> None:
        assert math.isfinite(observation_probability_weight)
        assert observation_probability_weight >= 0.0
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        # Algorithm 3's lambda: weight on the sampled observation log probability.
        self.observation_probability_weight = observation_probability_weight

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
        remaining_actions: int | None = None,
    ) -> tuple[float, ActionT | StopAction]:
        """Search one sampled deterministic successor per applicable action.

        Each unique cache-key/action edge is expanded at most once and samples one
        successor from the model's complete outcome distribution. A fixed seed and
        fixed traversal order are reproducible; changing the traversal order can
        assign random draws to different edges. Nodes are ordered by accumulated
        objective loss plus ``heuristic``. As in Algorithm 3, ``max_iterations``
        bounds the number of priority-queue iterations (pops), not search depth.

        Additional model sampling remains controlled by the experiment's
        master-seeded model. Optional ``remaining_actions`` separately limits plans
        to the real practice session's remaining action slots. Without it, the
        iteration limit bounds cyclic search without imposing a depth cutoff.
        """
        # ``horizon`` belongs to the shared interface for exact expectimax. It
        # deliberately does not bound this iteration-budgeted planner.
        del horizon
        assert remaining_actions is None or remaining_actions >= 0, (
            "remaining_actions must be non-negative"
        )
        assert self.max_iterations >= 1, "max_iterations must be positive"
        assert num_samples >= 1, "num_samples must be positive"
        assert math.isfinite(summed_cost) and summed_cost >= 0, (
            "summed_cost must be finite and non-negative"
        )
        started_at = time.perf_counter()

        root_value = model.J(
            belief_state=belief_state, summed_cost=summed_cost, num_samples=num_samples
        )
        assert not math.isnan(root_value) and root_value != math.inf, (
            "root J(C, b) must be finite or negative infinity"
        )
        # Algorithm 3 line 26 returns the path to the minimum-g node in Open U Closed.
        # Popped nodes leave Open and Closed stores only keys, so retain that minimum
        # and its return information instead of storing every complete search path.
        # ``best_g`` selects the path; ``best_objective_value`` is that node's J(C, b),
        # which the common BeliefSpacePlanner interface returns alongside the action.
        best_g = 0.0  # Algorithm 3's cumulative search cost g.
        best_objective_value = root_value
        best_action: ActionT | StopAction = STOP_ACTION
        best_path_depth = 0
        # Algorithm 3, line 3: Open <- {(b0, g=0)}.
        open_nodes: list[
            DeterminizedSearchQueueEntry[EnvironmentStateT, BeliefStateT, ActionT]
        ] = []
        # Algorithm 3's initial belief b_0 comprises the environment state, latent
        # belief state, and cumulative practice cost in our factored representation.
        root_key = model.search_cache_key(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            # An identical state reached with fewer action slots has different
            # continuations. None preserves unrestricted graph-search merging.
            horizon=remaining_actions,
        )
        # Our sampled stopping objective J(C, b), cached per belief-space node.
        values_by_key = {root_key: root_value}
        # Algorithm 3, line 3: Closed <- empty; Cost <- {b0: 0}. Here zero is
        # b0's cumulative search cost g (cost-to-come), not its objective value J.
        closed: set[object] = set()
        cost = {root_key: 0.0}
        # Potential differences in J can make edges negative. A cheaper path to
        # an expanded node must therefore propagate through its descendants.
        # Retain its sampled edges so reopening cannot redraw the determinization.
        sampled_edges: dict[
            object,
            list[tuple[ActionT, EnvironmentStateT, BeliefStateT, float, float]],
        ] = {}
        reopened_nodes = 0
        cached_successor_relaxations = 0
        stale_queue_entries = 0
        diagnostics = DeterminizedSearchDiagnostics()
        sequence = 0
        # Algorithm 3 uses one iteration per Open pop; this is also our traversed-node count.
        iterations = 0

        if trace is not None:
            trace.record(event="stop_value", node=0, value=root_value)

        if root_value != -math.inf:
            # The queue entry contains Algorithm 3's (b_0, g=0), plus only the
            # metadata needed for cache lookup, line-26 path recovery, and diagnostics.
            heapq.heappush(
                open_nodes,
                DeterminizedSearchQueueEntry(
                    belief=DeterminizedSearchBelief(
                        environment_state=environment_state,
                        belief_state=belief_state,
                        summed_cost=summed_cost,
                    ),
                    g=0.0,
                    diagnostic_info=DeterminizedNodeDiagnosticInfo(depth=0),
                    cache_info=DeterminizedNodeCacheInfo(
                        key=root_key,
                        stop_value=root_value,
                        queue_sequence=sequence,
                    ),
                    path_recovery_info=DeterminizedPathRecoveryInfo(first_action=STOP_ACTION),
                ),
            )
            diagnostics.observe_frontier(size=1)

        # Algorithm 3: while Open != empty and iterations < H.
        while open_nodes and iterations < self.max_iterations:
            current_node = heapq.heappop(open_nodes)
            iterations += 1
            if self.strictly_improves(
                candidate=cost[current_node.cache_info.key], previous=current_node.g
            ):
                stale_queue_entries += 1
                continue
            # Algorithm 3, lines 5 and 9: pop lowest(Open), then skip Closed nodes.
            if current_node.cache_info.key in closed:
                continue
            # Algorithm 3, line 12: add b to Closed.
            closed.add(current_node.cache_info.key)
            diagnostics.observe_expansion(depth=current_node.diagnostic_info.depth)

            if remaining_actions is not None and (
                current_node.diagnostic_info.depth >= remaining_actions
            ):
                diagnostics.action_horizon_terminal_nodes += 1
                continue

            edges = sampled_edges.get(current_node.cache_info.key)
            sample_new_edges = edges is None
            if edges is None:
                edges = []
                sampled_edges[current_node.cache_info.key] = edges
                actions = model.get_valid_actions(
                    environment_state=current_node.belief.environment_state
                )
            else:
                actions = [edge[0] for edge in edges]
                cached_successor_relaxations += len(edges)

            # Keep the first expansion's original sample/evaluate order even if
            # the model shares random state between transitions and J.
            for edge_index, action in enumerate(actions):
                if sample_new_edges:
                    diagnostics.action_transitions_evaluated += 1
                    # Algorithm 3, lines 14-15: jointly sample one state/cost/
                    # observation outcome, once per unique state/action edge.
                    joint_transition_outcomes = model.transition_outcomes(
                        environment_state=current_node.belief.environment_state,
                        practice_action=action,
                        belief_state=current_node.belief.belief_state,
                    )
                    diagnostics.chance_outcomes_enumerated += len(joint_transition_outcomes)
                    (
                        next_environment_state,
                        sampled_cost,
                        estimated_observation_probability,
                    ) = self.sample_outcome(outcomes=joint_transition_outcomes, action=action)
                    next_belief = model.compute_next_belief_state(
                        belief_state=current_node.belief.belief_state,
                        environment_state=current_node.belief.environment_state,
                        potential_next_environment_state=next_environment_state,
                        practice_action=action,
                    )
                    edges.append((
                        action,
                        next_environment_state,
                        next_belief,
                        sampled_cost,
                        estimated_observation_probability,
                    ))
                    diagnostics.generated_successors += 1
                else:
                    (
                        _,
                        next_environment_state,
                        next_belief,
                        sampled_cost,
                        estimated_observation_probability,
                    ) = edges[edge_index]
                next_cost = current_node.belief.summed_cost + sampled_cost
                next_depth = current_node.diagnostic_info.depth + 1
                # Only a root edge establishes the first action. Descendants inherit
                # it so line 26 can return the first action on the selected path.
                first_action = (
                    action
                    if current_node.diagnostic_info.depth == 0
                    else current_node.path_recovery_info.first_action
                )
                key, value = self.compute_and_cache_stop_value(
                    values_by_key=values_by_key,
                    diagnostics=diagnostics,
                    model=model,
                    environment_state=next_environment_state,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
                    remaining_actions=(
                        None if remaining_actions is None else remaining_actions - next_depth
                    ),
                )
                if value == -math.inf:
                    continue
                # Algorithm 3, lines 17-19: update g'. Here the estimated
                # reward is improvement in J. For linear G,
                #
                #   J(C, b) - J(C + c(a), b')
                #     = -(E[F(theta')]-E[F(theta)]) + lambda*c(a),
                #
                # so the paper's +cost(a) is already present as +lambda*c(a).
                # Adding it separately would double-count action cost. Algorithm
                # 3's other lambda weights observation surprise; it is a distinct
                # hyperparameter from our linear-G cost coefficient.
                estimated_reward = value - current_node.cache_info.stop_value  # Algorithm 3: r_hat.
                # Algorithm 3, line 18: p_hat is the probability of the particular
                # observation selected by determinization. Its negative log is
                # conventionally called surprisal: likely outcomes add less cost.
                observation_surprise = -self.observation_probability_weight * math.log(
                    estimated_observation_probability
                )  # Algorithm 3: -lambda*log(p_hat).
                child = DeterminizedSearchNode(
                    environment_state=next_environment_state,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    depth=next_depth,
                    stop_value=value,
                    g=current_node.g,
                )
                # Algorithm 3, line 18: self.heuristic returns the complete alpha*h_hat term.
                heuristic_cost = self.heuristic(node=child)
                assert math.isfinite(heuristic_cost), "heuristic must return a finite value"
                next_path_cost = (
                    current_node.g - estimated_reward + observation_surprise + heuristic_cost
                )  # Algorithm 3: g'.
                # Algorithm 3, lines 20-22: insert b' only for a newly discovered
                # or strictly cheaper path.
                if key not in cost or self.strictly_improves(
                    candidate=next_path_cost, previous=cost[key]
                ):
                    cost[key] = next_path_cost
                    if key in closed:
                        closed.remove(key)
                        reopened_nodes += 1
                    if next_path_cost < best_g:
                        best_g = next_path_cost
                        best_objective_value = value
                        best_action = first_action
                        best_path_depth = next_depth
                    sequence += 1
                    heapq.heappush(
                        open_nodes,
                        DeterminizedSearchQueueEntry(
                            belief=DeterminizedSearchBelief(
                                environment_state=next_environment_state,
                                belief_state=next_belief,
                                summed_cost=next_cost,
                            ),
                            g=next_path_cost,
                            diagnostic_info=DeterminizedNodeDiagnosticInfo(depth=next_depth),
                            cache_info=DeterminizedNodeCacheInfo(
                                key=key,
                                stop_value=value,
                                queue_sequence=sequence,
                            ),
                            path_recovery_info=DeterminizedPathRecoveryInfo(
                                first_action=first_action
                            ),
                        ),
                    )
                    diagnostics.observe_frontier(size=len(open_nodes))

        if open_nodes and iterations >= self.max_iterations:
            diagnostics.termination_reason = "iteration_budget"
        elif diagnostics.action_horizon_terminal_nodes:
            diagnostics.termination_reason = "action_horizon_exhausted"

        if trace is not None:
            # Algorithm 3, line 26: return the first action on the minimum-g
            # sampled path found in Open or Closed. ``best_g`` tracks
            # that node as children are generated.
            trace.record(
                event="choice",
                node=0,
                action="STOP"
                if best_action == STOP_ACTION
                else best_action.model_dump(mode="json", fallback=str),
                value=best_objective_value,
                selected_path_depth=best_path_depth,
                reason="minimum_g_sampled_path_stop_wins_ties",
            )
            trace.record(
                event="search_summary",
                node=0,
                solver="determinized_astar",
                expanded_nodes=diagnostics.expanded_nodes,
                traversed_nodes=iterations,
                generated_successors=diagnostics.generated_successors,
                unique_nodes=len(values_by_key),
                merged_nodes=diagnostics.merged_nodes,
                reopened_nodes=reopened_nodes,
                cached_successor_relaxations=cached_successor_relaxations,
                stale_queue_entries=stale_queue_entries,
                action_transitions_evaluated=diagnostics.action_transitions_evaluated,
                chance_outcomes_enumerated=diagnostics.chance_outcomes_enumerated,
                frontier_nodes=len(open_nodes),
                max_frontier_size=diagnostics.max_frontier_size,
                max_depth_reached=diagnostics.max_depth_reached,
                effective_action_horizon=remaining_actions,
                selected_path_depth=best_path_depth,
                action_horizon_terminal_nodes=diagnostics.action_horizon_terminal_nodes,
                iterations=iterations,
                max_iterations=self.max_iterations,
                observation_probability_weight=self.observation_probability_weight,
                stop_value_evaluations=len(values_by_key),
                search_elapsed_seconds=time.perf_counter() - started_at,
                termination_reason=diagnostics.termination_reason,
            )
        return best_objective_value, best_action

    @staticmethod
    def strictly_improves(*, candidate: float, previous: float) -> bool:
        """Ignore roundoff in telescoping potential costs when relaxing a path."""
        return candidate < previous and not math.isclose(
            candidate, previous, rel_tol=1e-12, abs_tol=1e-15
        )

    def heuristic(self, *, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]) -> float:
        """Generic hook used at line 19 where Algorithm 3 adds ``alpha * h_hat``.

        This implementation currently returns zero; it does not compute the paper's
        entropy heuristic. The method is the single place where a later generic
        heuristic will be implemented.
        """
        del node
        return 0.0

    def compute_and_cache_stop_value(
        self,
        *,
        values_by_key: dict[object, float],
        diagnostics: DeterminizedSearchDiagnostics,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        environment_state: EnvironmentStateT,
        belief_state: BeliefStateT,
        summed_cost: float,
        num_samples: int,
        remaining_actions: int | None = None,
    ) -> tuple[object, float]:
        """Return the belief key and cached J(C, b), computing J on a cache miss."""
        key = model.search_cache_key(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=remaining_actions,
        )
        cached_value = values_by_key.get(key)
        if cached_value is not None:
            diagnostics.merged_nodes += 1
            return key, cached_value
        value = model.J(
            belief_state=belief_state,
            summed_cost=summed_cost,
            num_samples=num_samples,
        )
        assert not math.isnan(value) and value != math.inf, (
            "J(C, b) must be finite or negative infinity"
        )
        values_by_key[key] = value
        return key, value

    def sample_outcome(
        self,
        *,
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
        outcome_index = int(self.rng.choice(len(outcomes), p=probabilities / total_probability))
        outcome = outcomes[outcome_index]
        assert math.isfinite(outcome[1]) and outcome[1] >= 0, (
            "sampled_cost must be finite and non-negative"
        )
        return outcome
