"""Finite-horizon belief-space expectimax."""

import math
from typing import Generic

import numpy as np

from .types.protocol import (
    ActionT,
    BeliefSpaceModel,
    BeliefStateT,
    EnvironmentStateT,
    ThetaT,
)
from .types.search_trace import SearchTrace
from .types.stop_action import NUM_SAMPLES, STOP_ACTION, StopAction


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
    """Implementation of understanding/pomdp_formulation.py.

    Pseudocode and review:
    https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/290
    Algorithm notes (Google Drive reference from the pseudocode):
    https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a

    Model methods have the pseudocode's names and keyword arguments. Costs and
    policy values allow floats. Theta is sampled num_samples times per unique search state.
    Stopping wins ties. Each call owns a fresh recursive cache, so later searches
    resample theta and see updated model parameters.
    """
    assert num_samples >= 1, "num_samples must be positive"
    solver = ExpectimaxSearch(model=model, num_samples=num_samples, trace=trace)
    result = solver.cached_solve_belief_space_expectimax(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=horizon,
    )
    if trace is not None:
        trace.record(
            event="search_summary",
            node=0,
            expanded_nodes=solver.next_node,
            cache_requests=solver.cache_requests,
            cache_hits=solver.cache_hits,
            action_evaluations=solver.action_evaluations,
            chance_outcomes=solver.chance_outcomes,
            nodes_by_horizon=dict(sorted(solver.nodes_by_horizon.items(), reverse=True)),
        )
    return result


class ExpectimaxSearch(Generic[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]):
    """One search's model and cached recursion."""

    def __init__(
        self,
        *,
        model: BeliefSpaceModel[EnvironmentStateT, BeliefStateT, ThetaT, ActionT],
        num_samples: int,
        trace: SearchTrace | None = None,
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.memo: dict[object, tuple[float, ActionT | StopAction]] = {}
        self.trace = trace
        self.next_node = 0
        self.cache_requests = 0
        self.cache_hits = 0
        self.action_evaluations = 0
        self.chance_outcomes = 0
        self.nodes_by_horizon: dict[int, int] = {}

    def cached_solve_belief_space_expectimax(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
    ) -> tuple[float, ActionT | StopAction]:
        self.cache_requests += 1
        key = self.model.search_cache_key(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        cached = self.memo.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        result = self.solve_belief_space_expectimax(
            environment_state=environment_state,
            summed_cost=summed_cost,
            belief_state=belief_state,
            horizon=horizon,
        )
        self.memo[key] = result
        return result

    def solve_belief_space_expectimax(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
    ) -> tuple[float, ActionT | StopAction]:
        assert horizon >= 0, f"horizon must be non-negative, got {horizon}"
        assert math.isfinite(summed_cost) and summed_cost >= 0, (
            "summed_cost must be finite and non-negative"
        )

        node = self.next_node
        self.next_node += 1
        self.nodes_by_horizon[horizon] = self.nodes_by_horizon.get(horizon, 0) + 1
        policy_values = self.model.sample_policy_values_from_belief(
            belief_state=belief_state, num_samples=self.num_samples
        )
        assert len(policy_values) == self.num_samples
        sample_values = []
        for current_policy_value in policy_values:
            current_pomdp_value = self.model.G(
                policy_value=current_policy_value, summed_cost=summed_cost
            )
            assert not math.isnan(current_pomdp_value) and current_pomdp_value != math.inf, (
                f"stop value must be finite or negative infinity, got {current_pomdp_value}"
            )
            sample_values.append(current_pomdp_value)
        if self.trace is not None and node == 0:
            self.trace.record(
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
        if self.trace is not None and node == 0:
            self.trace.record(event="stop_value", node=node, value=current_best_value)
        if current_best_value == -math.inf:
            if self.trace is not None and node == 0:
                self.trace.record(
                    event="choice",
                    node=node,
                    action="STOP",
                    value=current_best_value,
                    reason="objective_infeasible",
                )
            return current_best_value, current_best_action
        if horizon == 0:
            if self.trace is not None and node == 0:
                self.trace.record(
                    event="choice",
                    node=node,
                    action="STOP",
                    value=current_best_value,
                    reason="horizon_exhausted",
                )
            return current_best_value, current_best_action

        for practice_action in self.model.get_valid_actions(environment_state=environment_state):
            self.action_evaluations += 1
            value_of_state = 0.0
            total_probability = 0.0
            # TODO: Should samples be drawn with or without replacement?
            next_states_and_probabilities = self.model.transition_outcomes(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
            assert next_states_and_probabilities, (
                f"action {practice_action!r} has no chance outcomes"
            )
            self.chance_outcomes += len(next_states_and_probabilities)
            for (
                potential_next_environment_state,
                sampled_cost,
                probability,
            ) in next_states_and_probabilities:
                assert math.isfinite(sampled_cost) and sampled_cost >= 0, (
                    "sampled_cost must be finite and non-negative"
                )
                next_belief_state = self.model.update_belief_state(
                    belief_state=belief_state,
                    environment_state=environment_state,
                    potential_next_environment_state=potential_next_environment_state,
                    practice_action=practice_action,
                )
                value_of_next_state, _ = self.cached_solve_belief_space_expectimax(
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
                if self.trace is not None and node == 0:
                    self.trace.record(
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
            # Compare only after summing every successor, including negative values.
            if self.trace is not None and node == 0:
                self.trace.record(
                    event="action_value",
                    node=node,
                    action=practice_action.model_dump(mode="json", fallback=str),
                    value=value_of_state,
                )
            if current_best_value < value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

        if self.trace is not None and node == 0:
            self.trace.record(
                event="choice",
                node=node,
                action="STOP"
                if current_best_action == STOP_ACTION
                else current_best_action.model_dump(mode="json", fallback=str),
                value=current_best_value,
                reason="max_value_stop_wins_ties",
            )
        return current_best_value, current_best_action
