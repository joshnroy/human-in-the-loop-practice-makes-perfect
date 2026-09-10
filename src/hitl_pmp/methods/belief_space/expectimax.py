"""Finite-horizon belief-space expectimax."""

import math
import time

import numpy as np

from .planner import BeliefSpacePlanner
from .types.protocol import (
    ActionT,
    BeliefSpaceModel,
    BeliefStateT,
    EnvironmentStateT,
    ThetaT,
)
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


class ExpectimaxPlanner(BeliefSpacePlanner[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]):
    """Exact finite-horizon belief-space expectimax.

    This is the reference implementation from ``understanding/pomdp_formulation.py``.
    It evaluates the complete tree through ``horizon``; it has no node or wall-clock
    stopping condition. Each call owns a fresh cache, so later decisions resample
    latent parameters and see the latest model state.
    """

    name = "expectimax"

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
        """Return the exact finite-horizon value and first action."""
        assert num_samples >= 1, "num_samples must be positive"
        assert horizon >= 0, f"horizon must be non-negative, got {horizon}"
        assert math.isfinite(summed_cost) and summed_cost >= 0, (
            "summed_cost must be finite and non-negative"
        )

        self._model = model
        self._num_samples = num_samples
        self._trace = trace
        self._memo: dict[object, tuple[float, ActionT | StopAction]] = {}
        self._next_node = 0
        self._expanded_nodes = 0
        self._cache_requests = 0
        self._cache_hits = 0
        self._action_transitions_evaluated = 0
        self._chance_outcomes_enumerated = 0
        self._nodes_by_horizon: dict[int, int] = {}
        started_at = time.perf_counter()

        result = self._cached_solve(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        if trace is not None:
            trace.record(
                event="search_summary",
                node=0,
                solver=self.name,
                horizon=horizon,
                expanded_nodes=self._expanded_nodes,
                traversed_nodes=self._cache_requests,
                generated_successors=max(0, self._cache_requests - 1),
                unique_nodes=self._next_node,
                stop_value_evaluations=self._next_node,
                frontier_nodes=0,
                max_frontier_size=0,
                cache_requests=self._cache_requests,
                cache_hits=self._cache_hits,
                action_transitions_evaluated=self._action_transitions_evaluated,
                chance_outcomes_enumerated=self._chance_outcomes_enumerated,
                nodes_by_horizon=dict(sorted(self._nodes_by_horizon.items(), reverse=True)),
                max_depth_reached=(
                    horizon - min(self._nodes_by_horizon) if self._nodes_by_horizon else 0
                ),
                search_elapsed_seconds=time.perf_counter() - started_at,
                termination_reason="horizon_or_objective_exhausted",
            )
        return result

    def _cached_solve(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
    ) -> tuple[float, ActionT | StopAction]:
        self._cache_requests += 1
        key = self._model.search_cache_key(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        cached = self._memo.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        result = self._solve_node(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        self._memo[key] = result
        return result

    def _solve_node(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
    ) -> tuple[float, ActionT | StopAction]:
        node = self._next_node
        self._next_node += 1
        self._nodes_by_horizon[horizon] = self._nodes_by_horizon.get(horizon, 0) + 1
        policy_values = self._model.sample_policy_values_from_belief(
            belief_state=belief_state, num_samples=self._num_samples
        )
        assert len(policy_values) == self._num_samples
        sample_values = np.fromiter(
            (
                self._model.G(policy_value=float(policy_value), summed_cost=summed_cost)
                for policy_value in policy_values
            ),
            dtype=np.float64,
            count=self._num_samples,
        )
        assert all(not math.isnan(value) and value != math.inf for value in sample_values), (
            "stop value must be finite or negative infinity"
        )
        if self._trace is not None and node == 0:
            self._trace.record(
                event="sample_summary",
                node=node,
                count=len(sample_values),
                policy_value_mean=float(np.mean(policy_values)),
                policy_value_min=float(np.min(policy_values)),
                policy_value_max=float(np.max(policy_values)),
                pomdp_value_mean=float(np.mean(sample_values)),
                pomdp_value_min=float(np.min(sample_values)),
                pomdp_value_max=float(np.max(sample_values)),
            )

        current_best_value = float(np.mean(sample_values))
        current_best_action: ActionT | StopAction = STOP_ACTION
        if self._trace is not None and node == 0:
            self._trace.record(event="stop_value", node=node, value=current_best_value)
        if current_best_value == -math.inf:
            if self._trace is not None and node == 0:
                self._trace.record(
                    event="choice",
                    node=node,
                    action="STOP",
                    value=current_best_value,
                    reason="objective_infeasible",
                )
            return current_best_value, current_best_action
        if horizon == 0:
            if self._trace is not None and node == 0:
                self._trace.record(
                    event="choice",
                    node=node,
                    action="STOP",
                    value=current_best_value,
                    reason="horizon_exhausted",
                )
            return current_best_value, current_best_action

        self._expanded_nodes += 1
        for practice_action in self._model.get_valid_actions(environment_state=environment_state):
            self._action_transitions_evaluated += 1
            value_of_state = 0.0
            total_probability = 0.0
            next_states_and_probabilities = self._model.transition_outcomes(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
            assert next_states_and_probabilities, (
                f"action {practice_action!r} has no chance outcomes"
            )
            self._chance_outcomes_enumerated += len(next_states_and_probabilities)
            for (
                potential_next_environment_state,
                sampled_cost,
                probability,
            ) in next_states_and_probabilities:
                assert math.isfinite(sampled_cost) and sampled_cost >= 0, (
                    "sampled_cost must be finite and non-negative"
                )
                next_belief_state = self._model.update_belief_state(
                    belief_state=belief_state,
                    environment_state=environment_state,
                    potential_next_environment_state=potential_next_environment_state,
                    practice_action=practice_action,
                )
                value_of_next_state, _ = self._cached_solve(
                    environment_state=potential_next_environment_state,
                    summed_cost=summed_cost + sampled_cost,
                    belief_state=next_belief_state,
                    horizon=horizon - 1,
                )
                assert math.isfinite(probability) and probability > 0.0, (
                    f"chance probability must be finite and positive, got {probability}"
                )
                value_of_state += probability * value_of_next_state
                total_probability += probability
                if self._trace is not None and node == 0:
                    self._trace.record(
                        event="branch",
                        node=node,
                        action=practice_action.model_dump(mode="json", fallback=str),
                        horizon=horizon - 1,
                        summed_cost=summed_cost + sampled_cost,
                        sampled_cost=sampled_cost,
                        probability=probability,
                        successor_value=value_of_next_state,
                        contribution=probability * value_of_next_state,
                    )

            assert math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12), (
                f"chance probabilities sum to {total_probability}, not 1"
            )
            if self._trace is not None and node == 0:
                self._trace.record(
                    event="action_value",
                    node=node,
                    action=practice_action.model_dump(mode="json", fallback=str),
                    value=value_of_state,
                )
            if current_best_value < value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

        if self._trace is not None and node == 0:
            self._trace.record(
                event="choice",
                node=node,
                action=(
                    "STOP"
                    if current_best_action == STOP_ACTION
                    else current_best_action.model_dump(mode="json", fallback=str)
                ),
                value=current_best_value,
                reason="max_value_stop_wins_ties",
            )
        return current_best_value, current_best_action


def solve_belief_space_expectimax(
    *,
    environment_state: EnvironmentStateT,
    summed_cost: float,
    belief_state: BeliefStateT,
    horizon: int,
    model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, ActionT | StopAction]:
    """Compatibility entry point backed by :class:`ExpectimaxPlanner`."""
    return ExpectimaxPlanner[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]().solve(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=horizon,
        model=model,
        num_samples=num_samples,
        trace=trace,
    )
