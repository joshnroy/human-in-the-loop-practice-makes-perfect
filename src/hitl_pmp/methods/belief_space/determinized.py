"""Iteration-bounded best-first search through a belief-space determinization.

The traversal follows Algorithm 3 in Curtis et al. (2025), while leaving the
heuristic domain-independent:
https://drive.google.com/file/d/1_yxDfN1BDxr7jvdZEnyZJcUyP1QnyLRr/view?pli=1
"""

from __future__ import annotations

import heapq
import math
import time
from typing import cast

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
        log_full_search_tree: bool = False,
    ) -> None:
        assert math.isfinite(observation_probability_weight)
        assert observation_probability_weight >= 0.0
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        # Algorithm 3's lambda: weight on the sampled observation log probability.
        self.observation_probability_weight = observation_probability_weight
        self.log_full_search_tree = log_full_search_tree

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
        # ``best_g`` selects the path; ``best_objective_value`` is that node's J(C, b),
        # which the common BeliefSpacePlanner interface returns alongside the action.
        root_cost_penalty = self.cost_penalty(model=model, summed_cost=summed_cost)
        root_terminal_value = root_value + root_cost_penalty
        # Continuing nodes carry accumulated action/surprise cost in ``g``.  STOP is
        # an explicit terminal choice whose score is g - V_stop; lower is better.
        best_g = -root_terminal_value
        best_objective_value = root_value
        best_action: ActionT | StopAction = STOP_ACTION
        # Best discovered continuation grouped by its first root action. These
        # diagnostics explain why each applicable action did or did not beat STOP.
        root_action_paths: list[dict[str, object]] = []
        rejected_root_actions: list[tuple[ActionT, float, float]] = []
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
        search_node_ids = {root_key: 0}
        next_search_node_id = 1
        sequence = 0
        # Algorithm 3 uses one iteration per Open pop; this is also our traversed-node count.
        iterations = 0

        if trace is not None:
            trace.record(
                event="stop_value",
                node=0,
                value=root_terminal_value,
                cost_adjusted_value=root_value,
                terminal_score=best_g,
            )
            if self.log_full_search_tree:
                trace.record(
                    event="tree_node",
                    node=0,
                    depth=0,
                    g=0.0,
                    stop_value=root_value,
                    summed_cost=summed_cost,
                    atoms=self.environment_atoms(environment_state),
                )

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
            current_search_node_id = search_node_ids[current_node.cache_info.key]
            already_closed = current_node.cache_info.key in closed
            if trace is not None and self.log_full_search_tree:
                trace.record(
                    event="tree_pop",
                    node=current_search_node_id,
                    iteration=iterations,
                    depth=current_node.diagnostic_info.depth,
                    g=current_node.g,
                    frontier_size_after_pop=len(open_nodes),
                    skipped_already_closed=already_closed,
                )
            # Algorithm 3, lines 5 and 9: pop lowest(Open), then skip Closed nodes.
            if already_closed:
                continue
            # Algorithm 3, line 12: add b to Closed.
            closed.add(current_node.cache_info.key)
            diagnostics.observe_expansion(depth=current_node.diagnostic_info.depth)
            if current_node.diagnostic_info.depth > 0:
                expanded_root_path = next(
                    path
                    for path in root_action_paths
                    if path["action"] == current_node.path_recovery_info.first_action
                )
                expanded_root_path["expanded_nodes"] = (
                    cast(int, expanded_root_path["expanded_nodes"]) + 1
                )
                expanded_root_path["max_expanded_depth"] = max(
                    cast(int, expanded_root_path["max_expanded_depth"]),
                    current_node.diagnostic_info.depth,
                )

            # Algorithm 3: for action a applicable in belief b.
            for action in model.get_valid_actions(
                environment_state=current_node.belief.environment_state
            ):
                diagnostics.action_transitions_evaluated += 1
                # Algorithm 3, lines 14-15 -- the determinization step. Each entry is
                # one possible joint (next environment state, realized cost,
                # observation probability), not a latent particle. Sampling one entry
                # jointly selects the next state from line 14 and observation from line 15.
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
                next_cost = current_node.belief.summed_cost + sampled_cost
                next_depth = current_node.diagnostic_info.depth + 1
                # Algorithm 3, line 16: b' <- Branch(b, a, o).
                next_belief = model.compute_next_belief_state(
                    belief_state=current_node.belief.belief_state,
                    environment_state=current_node.belief.environment_state,
                    potential_next_environment_state=next_environment_state,
                    practice_action=action,
                )
                diagnostics.generated_successors += 1
                # Only a root edge establishes the first action. Descendants inherit
                # it so line 26 can return the first action on the selected path.
                first_action = (
                    action
                    if current_node.diagnostic_info.depth == 0
                    else current_node.path_recovery_info.first_action
                )
                action_path = current_node.path_recovery_info.action_path + (action,)
                key, value = self.compute_and_cache_stop_value(
                    values_by_key=values_by_key,
                    diagnostics=diagnostics,
                    model=model,
                    environment_state=next_environment_state,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
                )
                child_is_new = key not in search_node_ids
                if child_is_new:
                    child_search_node_id = next_search_node_id
                    next_search_node_id += 1
                    search_node_ids[key] = child_search_node_id
                else:
                    child_search_node_id = search_node_ids[key]
                if value == -math.inf:
                    if trace is not None and self.log_full_search_tree:
                        trace.record(
                            event="tree_edge",
                            node=current_search_node_id,
                            child_node=child_search_node_id,
                            depth=next_depth,
                            action=self.compact_action(action),
                            first_action=self.compact_action(first_action),
                            sampled_cost=sampled_cost,
                            sampled_observation_probability=estimated_observation_probability,
                            disposition="infeasible",
                        )
                    if current_node.diagnostic_info.depth == 0:
                        rejected_root_actions.append(
                            (action, sampled_cost, estimated_observation_probability)
                        )
                    continue
                # Non-STOP actions receive reward -lambda_cost*c(a). Equivalently,
                # Algorithm 3's ``-r_hat`` adds the marginal scaled action cost to g.
                # The deployment value is received only by the STOP candidate below.
                current_cost_penalty = self.cost_penalty(
                    model=model, summed_cost=current_node.belief.summed_cost
                )
                next_cost_penalty = self.cost_penalty(model=model, summed_cost=next_cost)
                action_cost_penalty = next_cost_penalty - current_cost_penalty
                estimated_reward = -action_cost_penalty
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
                )  # Algorithm 3: continuation g'.
                terminal_value = value + next_cost_penalty
                terminal_score = next_path_cost - terminal_value
                improves_cost = key not in cost or next_path_cost < cost[key]
                if trace is not None and self.log_full_search_tree:
                    if child_is_new:
                        trace.record(
                            event="tree_node",
                            node=child_search_node_id,
                            depth=next_depth,
                            g=next_path_cost,
                            stop_value=value,
                            summed_cost=next_cost,
                            atoms=self.environment_atoms(next_environment_state),
                        )
                    trace.record(
                        event="tree_edge",
                        node=current_search_node_id,
                        child_node=child_search_node_id,
                        depth=next_depth,
                        action=self.compact_action(action),
                        first_action=self.compact_action(first_action),
                        sampled_cost=sampled_cost,
                        sampled_observation_probability=estimated_observation_probability,
                        parent_stop_value=current_node.cache_info.stop_value,
                        child_stop_value=value,
                        estimated_reward=estimated_reward,
                        action_cost_penalty=action_cost_penalty,
                        observation_surprise=observation_surprise,
                        heuristic_cost=heuristic_cost,
                        child_g=next_path_cost,
                        terminal_value=terminal_value,
                        terminal_score=terminal_score,
                        previous_best_g=cost.get(key),
                        disposition="enqueued" if improves_cost else "dominated",
                    )
                existing_root_path = next(
                    (
                        path
                        for path in root_action_paths
                        if path["action"] == first_action
                    ),
                    None,
                )
                if existing_root_path is None:
                    path_diagnostic: dict[str, object] = {
                        "action": first_action,
                        "generated_nodes": 0,
                        "expanded_nodes": 0,
                        "max_generated_depth": 0,
                        "max_expanded_depth": 0,
                        "best_action_path": [],
                        "path_cost_g": terminal_score,
                        "continuation_g": next_path_cost,
                        "terminal_value": terminal_value,
                        "value": value,
                        "objective_improvement": terminal_value - root_terminal_value,
                        "observation_surprise": next_path_cost
                        - (next_cost_penalty - root_cost_penalty)
                        - heuristic_cost,
                        "heuristic_cost": heuristic_cost,
                        "practice_cost": next_cost - summed_cost,
                        "depth": next_depth,
                        "sampled_observation_probability": (
                            estimated_observation_probability
                        ),
                    }
                    root_action_paths.append(path_diagnostic)
                    existing_root_path = path_diagnostic
                existing_root_path["generated_nodes"] = (
                    cast(int, existing_root_path["generated_nodes"]) + 1
                )
                existing_root_path["max_generated_depth"] = max(
                    cast(int, existing_root_path["max_generated_depth"]), next_depth
                )
                if terminal_score <= cast(float, existing_root_path["path_cost_g"]):
                    existing_root_path.update(
                        {
                            "path_cost_g": terminal_score,
                            "continuation_g": next_path_cost,
                            "terminal_value": terminal_value,
                            "value": value,
                            "objective_improvement": terminal_value - root_terminal_value,
                            "observation_surprise": next_path_cost
                            - (next_cost_penalty - root_cost_penalty)
                            - heuristic_cost,
                            "heuristic_cost": heuristic_cost,
                            "practice_cost": next_cost - summed_cost,
                            "depth": next_depth,
                            "sampled_observation_probability": (
                                estimated_observation_probability
                            ),
                            "best_action_path": list(action_path),
                        }
                    )
                # Algorithm 3, lines 20-22: insert b' only for a newly discovered
                # or strictly cheaper path.
                if improves_cost:
                    cost[key] = next_path_cost
                    if terminal_score < best_g:
                        best_g = terminal_score
                        best_objective_value = value
                        best_action = first_action
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
                                first_action=first_action,
                                action_path=action_path,
                            ),
                        ),
                    )
                    diagnostics.observe_frontier(size=len(open_nodes))

        if open_nodes and iterations >= self.max_iterations:
            diagnostics.termination_reason = "iteration_budget"

        if trace is not None:
            for action, rejected_cost, rejected_probability in rejected_root_actions:
                trace.record(
                    event="action_rejected",
                    node=0,
                    action=action.model_dump(mode="json", fallback=str),
                    reason="infeasible_stop_value",
                    sampled_cost=rejected_cost,
                    sampled_observation_probability=rejected_probability,
                )
            for path in root_action_paths:
                action = cast(ActionT, path["action"])
                assert action != STOP_ACTION
                trace.record(
                    event="action_value",
                    node=0,
                    action=action.model_dump(mode="json", fallback=str),
                    value=path["value"],
                    path_cost_g=path["path_cost_g"],
                    objective_improvement=path["objective_improvement"],
                    observation_surprise=path["observation_surprise"],
                    heuristic_cost=path["heuristic_cost"],
                    practice_cost=path["practice_cost"],
                    depth=path["depth"],
                    sampled_observation_probability=path[
                        "sampled_observation_probability"
                    ],
                    generated_nodes=path["generated_nodes"],
                    expanded_nodes=path["expanded_nodes"],
                    max_generated_depth=path["max_generated_depth"],
                    max_expanded_depth=path["max_expanded_depth"],
                    best_action_path=[
                        action.model_dump(mode="json", fallback=str)
                        for action in cast(list[ActionT], path["best_action_path"])
                    ],
                    beats_stop=cast(float, path["path_cost_g"]) < 0.0,
                )
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
        return best_objective_value, best_action

    @staticmethod
    def compact_action(action: ActionT | StopAction) -> dict[str, object] | str:
        """Serialize only the grounded action identity needed to draw a debug tree."""
        if action == STOP_ACTION:
            return "STOP"
        payload = action.model_dump(mode="json", fallback=str)
        return {
            "skill": payload["skill"]["name"],
            "objects": [obj["name"] for obj in payload["objects"]],
        }

    @staticmethod
    def environment_atoms(environment_state: EnvironmentStateT) -> list[str] | None:
        """Return a compact symbolic-state label when the domain exposes atoms."""
        atoms = getattr(environment_state, "atoms", None)
        return list(atoms) if atoms is not None else None

    def heuristic(self, *, node: DeterminizedSearchNode[EnvironmentStateT, BeliefStateT]) -> float:
        """Generic hook used at line 19 where Algorithm 3 adds ``alpha * h_hat``.

        This implementation currently returns zero; it does not compute the paper's
        entropy heuristic. The method is the single place where a later generic
        heuristic will be implemented.
        """
        del node
        return 0.0

    @staticmethod
    def cost_penalty(
        *,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        summed_cost: float,
    ) -> float:
        """Return G's penalty for ``summed_cost`` independently of deployment value.

        For linear G this is ``lambda_cost * summed_cost``.  Expressing it through
        G keeps the planner generic and lets us recover the unpenalized terminal
        value from the already-sampled J without performing another Monte Carlo
        evaluation.
        """
        zero_cost_value = model.G(policy_value=0.0, summed_cost=0.0)
        cost_adjusted_value = model.G(policy_value=0.0, summed_cost=summed_cost)
        penalty = zero_cost_value - cost_adjusted_value
        assert math.isfinite(penalty) and penalty >= 0.0, (
            "determinized terminal-reward search requires a finite, non-negative "
            "cost penalty for every feasible node"
        )
        return penalty

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
