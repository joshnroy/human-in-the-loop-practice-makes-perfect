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
)


def solve_belief_space_expectimax(
    *,
    environment_state: EnvironmentState,
    summed_cost: float,
    belief_state: BeliefState,
    horizon: int,
    model: BeliefSpaceModel,
    num_samples: int = NUM_SAMPLES,
) -> tuple[float, POMDPAction]:
    """Implementation of understanding/pomdp_formulation.py.

    Pseudocode and review:
    https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/290
    Algorithm notes (Google Drive reference from the pseudocode):
    https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a

    Model methods have the pseudocode's names and keyword arguments. Costs and
    policy values allow floats. Successors are enumerated with probabilities
    summing to one; theta is sampled num_samples times per unique search state.
    Stopping wins ties. Each call owns a fresh recursive cache, so later searches
    resample theta and see updated model parameters.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    solver = ExpectimaxSearch(model=model, num_samples=num_samples)
    return solver.cached_solve_belief_space_expectimax(
        environment_state=environment_state,
        summed_cost=summed_cost,
        belief_state=belief_state,
        horizon=horizon,
    )


class ExpectimaxSearch:
    """One search's model and cached recursion."""

    def __init__(self, *, model: BeliefSpaceModel, num_samples: int) -> None:
        self.model = model
        self.num_samples = num_samples
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
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        if not math.isfinite(summed_cost) or summed_cost < 0:
            raise ValueError("summed_cost must be finite and non-negative")

        sample_values = []
        for _ in range(self.num_samples):
            sampled_theta = self.model.sample_theta_from_belief(belief_state=belief_state)
            current_policy_value = self.model.evaluate_policy(sampled_theta=sampled_theta)
            current_pomdp_value = self.model.score_pomdp_value_from_policy_value_and_cost(
                policy_value=current_policy_value, summed_cost=summed_cost
            )
            if not math.isfinite(current_pomdp_value):
                raise ValueError(f"stop value must be finite, got {current_pomdp_value}")
            sample_values.append(current_pomdp_value)

        # Scale before summing so large finite samples do not overflow the average.
        current_best_value = float(np.sum(np.asarray(sample_values) / self.num_samples))
        current_best_action = STOP_ACTION
        if horizon == 0:
            return current_best_value, current_best_action

        for practice_action in self.model.get_valid_actions(environment_state=environment_state):
            value_of_state = 0.0
            total_probability = 0.0
            next_states = self.model.sample_next_states(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
            if not next_states:
                raise ValueError(f"action {practice_action!r} has no chance outcomes")
            if len(set(next_states)) != len(next_states):
                raise ValueError("sample_next_states must return distinct successors and costs")

            for potential_next_environment_state, sampled_cost in next_states:
                if not math.isfinite(sampled_cost) or sampled_cost < 0:
                    raise ValueError("sampled_cost must be finite and non-negative")
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
                if not math.isfinite(probability) or probability <= 0.0:
                    raise ValueError(
                        f"chance probability must be finite and positive, got {probability}"
                    )
                value_of_state += probability * value_of_next_state
                total_probability += probability

            if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"chance probabilities sum to {total_probability}, not 1")
            # Compare only after summing every successor, including negative values.
            if current_best_value < value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

        return current_best_value, current_best_action
