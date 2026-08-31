"""Finite-horizon belief-space expectimax."""

import math
from functools import cache

import numpy as np

from .types import (
    NUM_SAMPLES,
    STOP_ACTION,
    BeliefSpaceModel,
    BeliefState,
    EnvironmentState,
    POMDPAction,
    SearchTrace,
)


def solve_belief_space_expectimax(
    *,
    environment_state: EnvironmentState,
    summed_cost: float,
    belief_state: BeliefState,
    horizon: int,
    model: BeliefSpaceModel,
    num_samples: int = NUM_SAMPLES,
    trace: SearchTrace | None = None,
) -> tuple[float, POMDPAction]:
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
    return solver.cached_solve_belief_space_expectimax(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=horizon,
    )


class ExpectimaxSearch:
    """One search's model and cached recursion."""

    def __init__(
        self, *, model: BeliefSpaceModel, num_samples: int, trace: SearchTrace | None = None
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.trace = trace
        # A bound-method cache belongs to this search, not to the class or model.
        self.cached_solve_belief_space_expectimax = cache(self.solve_belief_space_expectimax)

    def solve_belief_space_expectimax(
        self,
        *,
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int,
    ) -> tuple[float, POMDPAction]:
        assert horizon >= 0, f"horizon must be non-negative, got {horizon}"
        assert math.isfinite(summed_cost) and summed_cost >= 0, (
            "summed_cost must be finite and non-negative"
        )

        node = len(self.trace.events) if self.trace is not None else 0
        if self.trace is not None:
            self.trace.record(
                event="node",
                node=node,
                horizon=horizon,
                summed_cost=summed_cost,
                environment_state=environment_state.model_dump(mode="json"),
                belief_state=belief_state.model_dump(mode="json"),
            )
        sample_values = []
        for _ in range(self.num_samples):
            sampled_theta = self.model.sample_theta_from_belief(belief_state=belief_state)
            current_policy_value = self.model.evaluate_policy(sampled_theta=sampled_theta)
            current_pomdp_value = self.model.score_pomdp_value_from_policy_value_and_cost(
                policy_value=current_policy_value, summed_cost=summed_cost
            )
            assert math.isfinite(current_pomdp_value), (
                f"stop value must be finite, got {current_pomdp_value}"
            )
            sample_values.append(current_pomdp_value)
            if self.trace is not None:
                self.trace.record(
                    event="sample",
                    node=node,
                    theta=sampled_theta.model_dump(mode="json"),
                    policy_value=current_policy_value,
                    pomdp_value=current_pomdp_value,
                )

        current_best_value = float(np.mean(sample_values))
        current_best_action = STOP_ACTION
        if self.trace is not None:
            self.trace.record(event="stop_value", node=node, value=current_best_value)
        if horizon == 0:
            if self.trace is not None:
                self.trace.record(
                    event="choice",
                    node=node,
                    action="STOP",
                    value=current_best_value,
                    reason="horizon_exhausted",
                )
            return current_best_value, current_best_action

        for practice_action in self.model.get_valid_actions(environment_state=environment_state):
            value_of_state = 0.0
            total_probability = 0.0
            # TODO: Should samples be drawn with or without replacement?
            next_states = self.model.sample_next_states(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
            assert next_states, f"action {practice_action!r} has no chance outcomes"
            for potential_next_environment_state, sampled_cost in next_states:
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
                probability = self.model.transition_probability(
                    potential_next_environment_state=potential_next_environment_state,
                    sampled_cost=sampled_cost,
                    environment_state=environment_state,
                    practice_action=practice_action,
                    belief_state=belief_state,
                )
                assert math.isfinite(probability) and probability > 0.0, (
                    f"chance probability must be finite and positive, got {probability}"
                )
                value_of_state += probability * value_of_next_state
                total_probability += probability
                if self.trace is not None:
                    self.trace.record(
                        event="branch",
                        node=node,
                        action=practice_action.model_dump(mode="json"),
                        successor=potential_next_environment_state.model_dump(mode="json"),
                        belief_state=next_belief_state.model_dump(mode="json"),
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
            if self.trace is not None:
                self.trace.record(
                    event="action_value",
                    node=node,
                    action=practice_action.model_dump(mode="json"),
                    value=value_of_state,
                )
            if current_best_value < value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

        if self.trace is not None:
            self.trace.record(
                event="choice",
                node=node,
                action="STOP"
                if current_best_action == STOP_ACTION
                else current_best_action.model_dump(mode="json"),
                value=current_best_value,
                reason="max_value_stop_wins_ties",
            )
        return current_best_value, current_best_action
