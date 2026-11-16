"""Iteration-bounded best-first search through a belief-space determinization.

The traversal follows Algorithm 3 in Curtis et al. (2025), while leaving the
heuristic domain-independent:
https://drive.google.com/file/d/1_yxDfN1BDxr7jvdZEnyZJcUyP1QnyLRr/view?pli=1
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable, Iterator
from typing import Any, get_args

import numpy as np

from .decision_log_encoding import DecisionLogEncoding
from .planner import BeliefSpacePlanner
from .types.determinized import (
    DeterminizedNodeCacheInfo,
    DeterminizedNodeDiagnosticInfo,
    DeterminizedPathRecoveryInfo,
    DeterminizedSearchBelief,
    DeterminizedSearchDiagnostics,
    DeterminizedSearchNode,
    DeterminizedSearchQueueEntry,
    DeterminizedVariant,
    DrawScope,
)
from .types.protocol import ActionT, BeliefSpaceModel, BeliefStateT, EnvironmentStateT, ThetaT
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction
from .types.value_resolution import ValueResolution


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
        draw_scope: DrawScope = "node",
        variant: DeterminizedVariant = "algorithm3",
        num_rollouts: int | None = None,
    ) -> None:
        assert math.isfinite(observation_probability_weight)
        assert observation_probability_weight >= 0.0
        assert draw_scope in get_args(DrawScope), f"unknown draw scope {draw_scope!r}"
        assert variant in get_args(DeterminizedVariant), f"unknown variant {variant!r}"
        assert variant == "algorithm3" or draw_scope == "node", "curtis draws no shared outcome"
        assert num_rollouts is None or num_rollouts >= 1
        # "algorithm3": one sampled outcome per (node, action), nodes keyed with the
        # accumulated cost. "curtis": Curtis et al.'s PO_DAStar.plan_next_action -- one
        # child per distinct outcome of `num_rollouts` draws (None: every outcome, the
        # many-rollout limit), each charged -w log of its own branch probability, and
        # nodes keyed by the belief alone, the cost living only in g.
        self.variant = variant
        self.num_rollouts = num_rollouts
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        # Algorithm 3's lambda: weight on the sampled observation log probability.
        self.observation_probability_weight = observation_probability_weight
        self.log_full_search_tree = log_full_search_tree
        # "node": one independent draw per (search node, action) edge, so a node that a
        # cheap reset makes new re-rolls the same skill's outcome. Otherwise one uniform
        # per `model.outcome_draw_key(scope=...)` -- what the outcome distribution reads,
        # with ("state") or without ("skill") the atoms -- shared by every node with it.
        self.draw_scope = draw_scope

    def solve(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        horizon: int | None = None,
        num_samples: int = NUM_SAMPLES,
        trace: SearchTrace | None = None,
        remaining_actions: int | None = None,
        remaining_steps: float | None = None,
        action_steps: Callable[[ActionT], float] | None = None,
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
        ``horizon`` must be None: there is no search depth to set.

        Optional ``remaining_steps`` limits plans to the practice steps that are left,
        counted with ``action_steps``, the caller's forecast of the control steps one
        action takes: a path never takes an action that would overshoot, and a node
        whose path has spent them all is not expanded. Such a node is still valued by
        its stop value, as under ``remaining_actions``. The steps a path has spent are
        part of its node's identity (``node_key``), so two routes to one belief that
        spent different steps stay two nodes, each cut by its own count. With None no
        step count is read and every key is the model's own, unchanged.
        """
        # ``horizon`` belongs to the shared interface for exact expectimax. This
        # planner has no depth to bound, so a caller passing one is configured wrongly.
        if horizon is not None:
            raise ValueError(
                f"determinized_astar takes no search depth (got {horizon}); it is bounded "
                "only by max_iterations and remaining_actions"
            )
        assert remaining_actions is None or remaining_actions >= 0, (
            "remaining_actions must be non-negative"
        )
        if remaining_steps is not None and action_steps is None:
            raise ValueError("remaining_steps needs action_steps, the forecast steps per action")
        assert remaining_steps is None or remaining_steps >= 0, (
            "remaining_steps must be non-negative"
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
        best_path_steps = 0.0
        over_budget_root_actions: list[ActionT] = []
        # Algorithm 3, line 3: Open <- {(b0, g=0)}.
        open_nodes: list[
            DeterminizedSearchQueueEntry[EnvironmentStateT, BeliefStateT, ActionT]
        ] = []
        # Algorithm 3's initial belief b_0 comprises the environment state, latent
        # belief state, and cumulative practice cost in our factored representation.
        root_key = self.node_key(
            model=model,
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            # An identical state reached with fewer action slots has different
            # continuations. None preserves unrestricted graph-search merging.
            horizon=remaining_actions,
            steps=None if remaining_steps is None else 0.0,
        )
        # Our sampled stopping objective J(C, b), cached per belief-space node with
        # the cost it was evaluated at (a curtis node is reached at several costs). J
        # does not read the steps spent, so this cache is keyed without them.
        values_by_key = {
            self.node_key(
                model=model,
                environment_state=environment_state,
                summed_cost=summed_cost,
                belief_state=belief_state,
                horizon=remaining_actions,
            ): (root_value, summed_cost)
        }
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
        # The shared uniforms of the "state" and "skill" draw scopes, and how many
        # draws reused one (under the key `reused`).
        outcome_uniforms: dict[object, float] = {}
        reopened_nodes = 0
        cached_successor_relaxations = 0
        stale_queue_entries = 0
        diagnostics = DeterminizedSearchDiagnostics()
        search_node_ids = {root_key: 0}
        root_action_paths: list[dict[str, Any]] = []
        sequence = 0
        # Algorithm 3 uses one iteration per Open pop; this is also our traversed-node count.
        iterations = 0

        if trace is not None:
            trace.record(event="stop_value", node=0, value=root_value)
            if self.log_full_search_tree:
                trace.record(
                    event="tree_node",
                    node=0,
                    depth=0,
                    g=0.0,
                    stop_value=root_value,
                    summed_cost=summed_cost,
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
            if trace is not None and self.log_full_search_tree:
                trace.record(
                    event="tree_pop",
                    node=current_search_node_id,
                    iteration=iterations,
                    depth=current_node.diagnostic_info.depth,
                    g=current_node.g,
                    frontier_size_after_pop=len(open_nodes),
                    skipped_already_closed=current_node.cache_info.key in closed,
                    skipped_stale=self.strictly_improves(
                        candidate=cost[current_node.cache_info.key], previous=current_node.g
                    ),
                )
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
            if (
                remaining_steps is not None
                and current_node.path_recovery_info.steps >= remaining_steps
            ):
                diagnostics.step_budget_terminal_nodes += 1
                continue

            edges = sampled_edges.get(current_node.cache_info.key)
            sample_new_edges = edges is None
            if edges is None:
                edges = []
                sampled_edges[current_node.cache_info.key] = edges
                children = self.generate_children(
                    model=model,
                    node=current_node.belief,
                    edges=edges,
                    diagnostics=diagnostics,
                    outcome_uniforms=outcome_uniforms,
                )
            else:
                children = iter(list(edges))
                cached_successor_relaxations += len(edges)

            # Lazily generated, so the first expansion keeps its original sample/
            # evaluate order even if the model shares random state between
            # transitions and J.
            for (
                action,
                next_environment_state,
                next_belief,
                sampled_cost,
                estimated_observation_probability,
            ) in children:
                next_steps = current_node.path_recovery_info.steps
                if remaining_steps is not None:
                    assert action_steps is not None
                    forecast_steps = action_steps(action)
                    assert math.isfinite(forecast_steps) and forecast_steps >= 0, (
                        "forecast steps must be finite and non-negative"
                    )
                    next_steps += forecast_steps
                    # Checked per path, not when the edge is sampled: a cached edge can
                    # be relaxed again from a path that has spent fewer steps.
                    if next_steps > remaining_steps:
                        if (
                            trace is not None
                            and current_node.diagnostic_info.depth == 0
                            and action not in over_budget_root_actions
                        ):
                            over_budget_root_actions.append(action)
                            trace.record(
                                event="action_rejected",
                                node=0,
                                action=DecisionLogEncoding.value(value=action),
                                reason="exceeds_remaining_steps",
                                forecast_steps=forecast_steps,
                                remaining_steps=remaining_steps,
                            )
                        continue
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
                    model=model,
                    environment_state=next_environment_state,
                    belief_state=next_belief,
                    summed_cost=next_cost,
                    num_samples=num_samples,
                    remaining_actions=(
                        None if remaining_actions is None else remaining_actions - next_depth
                    ),
                    steps=None if remaining_steps is None else next_steps,
                )
                child_is_new = key not in search_node_ids
                if not child_is_new:
                    diagnostics.merged_nodes += 1
                child_search_node_id = search_node_ids.setdefault(key, len(search_node_ids))
                if value == -math.inf:
                    if trace is not None and current_node.diagnostic_info.depth == 0:
                        trace.record(
                            event="action_rejected",
                            node=0,
                            action=DecisionLogEncoding.value(value=action),
                            reason="infeasible_stop_value",
                            sampled_cost=sampled_cost,
                            sampled_observation_probability=estimated_observation_probability,
                        )
                    if trace is not None and self.log_full_search_tree:
                        trace.record(
                            event="tree_edge",
                            node=current_search_node_id,
                            child_node=child_search_node_id,
                            depth=next_depth,
                            action=DecisionLogEncoding.value(value=action),
                            disposition="infeasible",
                        )
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
                action_path = current_node.path_recovery_info.action_path + (action,)
                cumulative_surprise = (
                    current_node.path_recovery_info.observation_surprise + observation_surprise
                )
                cumulative_heuristic = (
                    current_node.path_recovery_info.heuristic_cost + heuristic_cost
                )
                if trace is not None:
                    path = next(
                        (item for item in root_action_paths if item["root_action"] == first_action),
                        None,
                    )
                    if path is None:
                        path = {"root_action": first_action, "path_cost_g": math.inf}
                        root_action_paths.append(path)
                    if next_path_cost < path["path_cost_g"]:
                        path.update(
                            value=value,
                            path_cost_g=next_path_cost,
                            objective_improvement=value - root_value,
                            observation_surprise=cumulative_surprise,
                            heuristic_cost=cumulative_heuristic,
                            practice_cost=next_cost - summed_cost,
                            depth=next_depth,
                            sampled_observation_probability=estimated_observation_probability,
                            best_action_path=[
                                DecisionLogEncoding.value(value=a) for a in action_path
                            ],
                            beats_stop=next_path_cost < 0.0,
                        )
                        if remaining_steps is not None:
                            path.update(forecast_steps=next_steps)
                    if self.log_full_search_tree:
                        if child_is_new:
                            trace.record(
                                event="tree_node",
                                node=child_search_node_id,
                                depth=next_depth,
                                g=next_path_cost,
                                stop_value=value,
                                summed_cost=next_cost,
                            )
                        trace.record(
                            event="tree_edge",
                            node=current_search_node_id,
                            child_node=child_search_node_id,
                            depth=next_depth,
                            action=DecisionLogEncoding.value(value=action),
                            sampled_cost=sampled_cost,
                            sampled_observation_probability=estimated_observation_probability,
                            parent_stop_value=current_node.cache_info.stop_value,
                            child_stop_value=value,
                            estimated_reward=estimated_reward,
                            observation_surprise=observation_surprise,
                            heuristic_cost=heuristic_cost,
                            child_g=next_path_cost,
                            previous_best_g=cost.get(key),
                            cached_successor=not sample_new_edges,
                            disposition="enqueued"
                            if key not in cost
                            or self.strictly_improves(candidate=next_path_cost, previous=cost[key])
                            else "dominated",
                        )
                # Algorithm 3, lines 20-22: insert b' only for a newly discovered
                # or strictly cheaper path.
                if key not in cost or self.strictly_improves(
                    candidate=next_path_cost, previous=cost[key]
                ):
                    cost[key] = next_path_cost
                    if key in closed:
                        closed.remove(key)
                        reopened_nodes += 1
                    # On the shared grid, so a path cheaper only by roundoff does
                    # not displace the first one found (ValueResolution).
                    if ValueResolution.rank(value=next_path_cost) < ValueResolution.rank(
                        value=best_g
                    ):
                        best_g = next_path_cost
                        best_objective_value = value
                        best_action = first_action
                        best_path_depth = next_depth
                        best_path_steps = next_steps
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
                                observation_surprise=cumulative_surprise,
                                heuristic_cost=cumulative_heuristic,
                                steps=next_steps,
                            ),
                        ),
                    )
                    diagnostics.observe_frontier(size=len(open_nodes))

        if open_nodes and iterations >= self.max_iterations:
            diagnostics.termination_reason = "iteration_budget"
        elif diagnostics.action_horizon_terminal_nodes:
            diagnostics.termination_reason = "action_horizon_exhausted"
        elif diagnostics.step_budget_terminal_nodes:
            diagnostics.termination_reason = "step_budget_exhausted"

        if trace is not None:
            for path in root_action_paths:
                root_action = path.pop("root_action")
                trace.record(
                    event="action_value",
                    node=0,
                    action=DecisionLogEncoding.value(value=root_action),
                    **path,
                )
            # Algorithm 3, line 26: return the first action on the minimum-g
            # sampled path found in Open or Closed. ``best_g`` tracks
            # that node as children are generated.
            trace.record(
                event="choice",
                node=0,
                action="STOP"
                if best_action == STOP_ACTION
                else DecisionLogEncoding.value(value=best_action),
                value=best_objective_value,
                selected_path_depth=best_path_depth,
                reason="minimum_g_sampled_path_stop_wins_ties",
            )
            draw_fields: dict[str, object] = {}
            if self.draw_scope != "node":
                draw_fields.update(
                    draw_scope=self.draw_scope,
                    reused_outcome_draws=diagnostics.reused_outcome_draws,
                )
            if self.variant != "algorithm3":
                draw_fields.update(
                    determinized_variant=self.variant, num_rollouts=self.num_rollouts
                )
            if remaining_steps is not None:
                draw_fields.update(
                    remaining_steps=remaining_steps,
                    selected_path_steps=best_path_steps,
                    step_budget_terminal_nodes=diagnostics.step_budget_terminal_nodes,
                )
            trace.record(
                event="search_summary",
                node=0,
                solver="determinized_astar",
                expanded_nodes=diagnostics.expanded_nodes,
                traversed_nodes=iterations,
                generated_successors=diagnostics.generated_successors,
                unique_nodes=len(search_node_ids),
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
                **draw_fields,
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

    def node_key(
        self,
        *,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int | None,
        steps: float | None = None,
    ) -> object:
        """A search node's identity: with the accumulated cost (ours) or, as in
        PO_DAStar, the belief alone, the cost being carried by g.

        Under a practice step budget ``steps``, the forecast steps the path has spent,
        joins either key. Neither key determines it: the curtis key carries no cost at
        all, and one cost is many step counts once a path mixes a person's ask (dear,
        one step) with robot steps (cheap, many). Merged, a node keeps the steps of
        whichever route reached it first or cheapest, and the budget cut then prunes
        or admits continuations for the other route on the wrong count. With None the
        key is the model's own."""
        if self.variant == "curtis":
            key = model.belief_cache_key(
                environment_state=environment_state, belief_state=belief_state, horizon=horizon
            )
        else:
            key = model.search_cache_key(
                environment_state=environment_state,
                summed_cost=summed_cost,
                belief_state=belief_state,
                horizon=horizon,
            )
        if steps is None:
            return key
        return key, self.steps_key(steps=steps)

    @staticmethod
    def steps_key(*, steps: float) -> float:
        """Steps spent as a merge key, independent of summation order: one multiset of
        actions sums its forecasts (means of observed step counts, so inexact in
        binary) to totals that differ in the last bits by order. Nine significant
        digits, as for the model's cost key, keep those together and keep apart any
        two totals a ten-thousandth of a step apart up to 1e5 steps."""
        return float(f"{steps:.9g}")

    def generate_children(
        self,
        *,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        node: DeterminizedSearchBelief[EnvironmentStateT, BeliefStateT],
        edges: list[tuple[ActionT, EnvironmentStateT, BeliefStateT, float, float]],
        diagnostics: DeterminizedSearchDiagnostics,
        outcome_uniforms: dict[object, float],
    ) -> Iterator[tuple[ActionT, EnvironmentStateT, BeliefStateT, float, float]]:
        """Each valid action's children, appended to ``edges`` as they are yielded.

        Ours: one sampled outcome per action (Algorithm 3, lines 14-15). Curtis: one
        child per distinct outcome, with its branch probability -- the model's, or its
        frequency among ``num_rollouts`` draws."""
        for action in model.get_valid_actions(environment_state=node.environment_state):
            diagnostics.action_transitions_evaluated += 1
            outcomes = model.transition_outcomes(
                environment_state=node.environment_state,
                practice_action=action,
                belief_state=node.belief_state,
            )
            diagnostics.chance_outcomes_enumerated += len(outcomes)
            if self.variant == "curtis":
                branches = self.branch_outcomes(outcomes=outcomes, action=action)
            else:
                uniform = None
                if self.draw_scope != "node":
                    draw_key = model.outcome_draw_key(
                        environment_state=node.environment_state,
                        belief_state=node.belief_state,
                        action=action,
                        scope=self.draw_scope,
                    )
                    uniform = outcome_uniforms.get(draw_key)
                    if uniform is None:
                        uniform = float(self.rng.random())
                        outcome_uniforms[draw_key] = uniform
                    else:
                        diagnostics.reused_outcome_draws += 1
                branches = [self.sample_outcome(outcomes=outcomes, action=action, uniform=uniform)]
            for next_environment_state, sampled_cost, probability in branches:
                next_belief = model.compute_next_belief_state(
                    belief_state=node.belief_state,
                    environment_state=node.environment_state,
                    potential_next_environment_state=next_environment_state,
                    practice_action=action,
                )
                edge = (action, next_environment_state, next_belief, sampled_cost, probability)
                edges.append(edge)
                diagnostics.generated_successors += 1
                yield edge

    def branch_outcomes(
        self,
        *,
        outcomes: list[tuple[EnvironmentStateT, float, float]],
        action: ActionT,
    ) -> list[tuple[EnvironmentStateT, float, float]]:
        """PO_DAStar's branches: every outcome at its own probability, or the distinct
        outcomes of ``num_rollouts`` draws at their empirical frequency."""
        assert outcomes, f"action {action!r} has no chance outcomes"
        if self.num_rollouts is None:
            return list(outcomes)
        probabilities = np.fromiter((outcome[2] for outcome in outcomes), dtype=np.float64)
        draws = self.rng.choice(
            len(outcomes), size=self.num_rollouts, p=probabilities / probabilities.sum()
        )
        counts = np.bincount(draws, minlength=len(outcomes))
        return [
            (outcome[0], outcome[1], count / self.num_rollouts)
            for outcome, count in zip(outcomes, counts, strict=True)
            if count
        ]

    def compute_and_cache_stop_value(
        self,
        *,
        values_by_key: dict[object, tuple[float, float]],
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        environment_state: EnvironmentStateT,
        belief_state: BeliefStateT,
        summed_cost: float,
        num_samples: int,
        remaining_actions: int | None = None,
        steps: float | None = None,
    ) -> tuple[object, float]:
        """Return the node key and J(C, b), computing J on a cache miss.

        J is cached by the model's key alone; ``steps`` only joins the node key that is
        returned, so one belief reached after different step counts is valued once.

        A curtis node is one belief reached at several costs. G separates into a
        policy-value term and a cost term, so J(C, b) = J(C', b) + G(0, C) - G(0, C'):
        the cost difference between two routes is exactly what their edge rewards
        already charged. A cached -inf (over a hard budget) carries no value to shift
        from, so it is recomputed."""
        key = self.node_key(
            model=model,
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=remaining_actions,
        )
        node_key = key if steps is None else (key, self.steps_key(steps=steps))
        cached = values_by_key.get(key)
        if cached is not None:
            cached_value, cached_cost = cached
            # Ours keys the cost itself, so a hit is the same cost up to `cost_key`.
            if self.variant != "curtis" or cached_cost == summed_cost:
                return node_key, cached_value
            if cached_value != -math.inf:
                shift = model.G(policy_value=0.0, summed_cost=summed_cost) - model.G(
                    policy_value=0.0, summed_cost=cached_cost
                )
                if not math.isnan(shift):
                    return node_key, cached_value + shift
        value = model.J(
            belief_state=belief_state,
            summed_cost=summed_cost,
            num_samples=num_samples,
        )
        assert not math.isnan(value) and value != math.inf, (
            "J(C, b) must be finite or negative infinity"
        )
        if cached is None or value != -math.inf:
            values_by_key[key] = (value, summed_cost)
        return node_key, value

    def sample_outcome(
        self,
        *,
        outcomes: list[tuple[EnvironmentStateT, float, float]],
        action: ActionT,
        uniform: float | None = None,
    ) -> tuple[EnvironmentStateT, float, float]:
        """Validate a chance distribution and sample one determinized successor.

        A given ``uniform`` goes through the CDF exactly as ``Generator.choice`` maps
        its own draw, so equal distributions given equal uniforms agree."""
        assert outcomes, f"action {action!r} has no chance outcomes"
        probabilities = np.fromiter((outcome[2] for outcome in outcomes), dtype=np.float64)
        assert np.all(np.isfinite(probabilities)) and np.all(probabilities > 0.0), (
            "chance probabilities must be finite and positive"
        )
        total_probability = float(np.sum(probabilities))
        assert math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12), (
            f"chance probabilities sum to {total_probability}, not 1"
        )
        if uniform is None:
            outcome_index = int(self.rng.choice(len(outcomes), p=probabilities / total_probability))
        else:
            assert 0.0 <= uniform < 1.0
            cdf = np.cumsum(probabilities / total_probability)
            cdf /= cdf[-1]
            outcome_index = min(int(cdf.searchsorted(uniform, side="right")), len(outcomes) - 1)
        outcome = outcomes[outcome_index]
        assert math.isfinite(outcome[1]) and outcome[1] >= 0, (
            "sampled_cost must be finite and non-negative"
        )
        return outcome
