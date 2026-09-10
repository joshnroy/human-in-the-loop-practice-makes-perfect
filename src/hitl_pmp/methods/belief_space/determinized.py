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
        # ``horizon`` belongs to the shared interface for exact expectimax. It
        # deliberately does not bound this iteration-budgeted planner.
        del horizon
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
        best_g = 0.0  # Algorithm 3's cumulative search cost g.
        best_value = root_value
        best_action: ActionT | StopAction = STOP_ACTION
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
            # Determinized search is compute-bounded, not horizon-bounded. The
            # model protocol still accepts a remaining-horizon discriminator for
            # exact expectimax; None is the depth-independent sentinel.
            horizon=None,
        )
        # Our sampled stopping objective J(C, b), cached per belief-space node.
        values_by_key = {root_key: root_value}
        # Algorithm 3, line 3: Closed <- empty; Cost <- {b0: 0}. Here zero is
        # b0's cumulative search cost g (cost-to-come), not its objective value J.
        closed: set[object] = set()
        cost = {root_key: 0.0}
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
                    g=0.0,
                    sequence=sequence,
                    environment_state=environment_state,
                    belief_state=belief_state,
                    summed_cost=summed_cost,
                    depth=0,
                    key=root_key,
                    stop_value=root_value,
                    first_action=STOP_ACTION,
                ),
            )
            diagnostics.observe_frontier(size=1)

        # A hard-budget G(C, theta) can make J(C, b) negative infinity at the root.
        # Since execution costs are non-negative, deeper nodes cannot restore feasibility.
        # Algorithm 3: while Open != empty and iterations < H.
        while open_nodes and iterations < self.max_iterations:
            current_node = heapq.heappop(open_nodes)
            iterations += 1
            # Algorithm 3, lines 5 and 9: pop lowest(Open), then skip Closed nodes.
            if current_node.key in closed:
                continue
            # Algorithm 3, line 12: add b to Closed.
            closed.add(current_node.key)
            diagnostics.observe_expansion(depth=current_node.depth)

            # Algorithm 3: for action a applicable in belief b.
            for action in model.get_valid_actions(environment_state=current_node.environment_state):
                diagnostics.action_transitions_evaluated += 1
                # Algorithm 3, lines 14-15 -- the determinization step. Each entry is
                # one possible joint (next environment state, realized cost,
                # observation probability), not a latent particle. Sampling one entry
                # jointly selects the next state from line 14 and observation from line 15.
                joint_transition_outcomes = model.transition_outcomes(
                    environment_state=current_node.environment_state,
                    practice_action=action,
                    belief_state=current_node.belief_state,
                )
                diagnostics.chance_outcomes_enumerated += len(joint_transition_outcomes)
                (
                    next_environment_state,
                    sampled_cost,
                    estimated_observation_probability,
                ) = self.sample_outcome(outcomes=joint_transition_outcomes, action=action)
                next_cost = current_node.summed_cost + sampled_cost
                next_depth = current_node.depth + 1
                # Algorithm 3, line 16: b' <- Branch(b, a, o).
                next_belief = model.compute_next_belief_state(
                    belief_state=current_node.belief_state,
                    environment_state=current_node.environment_state,
                    potential_next_environment_state=next_environment_state,
                    practice_action=action,
                )
                diagnostics.generated_successors += 1
                # Only a root edge establishes the first action. Descendants inherit
                # it so line 26 can return the first action on the selected path.
                first_action = action if current_node.depth == 0 else current_node.first_action
                key, value = self.compute_and_cache_stop_value(
                    values_by_key=values_by_key,
                    diagnostics=diagnostics,
                    model=model,
                    environment_state=next_environment_state,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
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
                estimated_reward = value - current_node.stop_value  # Algorithm 3: r_hat.
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
                previous_path_cost = cost.get(key)
                if previous_path_cost is not None and previous_path_cost <= next_path_cost:
                    continue
                cost[key] = next_path_cost
                if next_path_cost < best_g:
                    best_g = next_path_cost
                    best_value = value
                    best_action = first_action
                sequence += 1
                heapq.heappush(
                    open_nodes,
                    DeterminizedSearchQueueEntry(
                        g=next_path_cost,
                        sequence=sequence,
                        environment_state=next_environment_state,
                        belief_state=next_belief,
                        summed_cost=next_cost,
                        depth=next_depth,
                        key=key,
                        stop_value=value,
                        first_action=first_action,
                    ),
                )
                diagnostics.observe_frontier(size=len(open_nodes))

        if open_nodes and iterations >= self.max_iterations:
            diagnostics.termination_reason = "iteration_budget"

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
                value=best_value,
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
                action_transitions_evaluated=diagnostics.action_transitions_evaluated,
                chance_outcomes_enumerated=diagnostics.chance_outcomes_enumerated,
                frontier_nodes=len(open_nodes),
                max_frontier_size=diagnostics.max_frontier_size,
                max_depth_reached=diagnostics.max_depth_reached,
                iterations=iterations,
                max_iterations=self.max_iterations,
                observation_probability_weight=self.observation_probability_weight,
                stop_value_evaluations=len(values_by_key),
                search_elapsed_seconds=time.perf_counter() - started_at,
                termination_reason=diagnostics.termination_reason,
            )
        return best_value, best_action

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
    ) -> tuple[object, float]:
        """Return the belief key and cached J(C, b), computing J on a cache miss."""
        key = model.search_cache_key(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=None,
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
