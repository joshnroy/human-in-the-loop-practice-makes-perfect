"""
This contains the outline for the expectimax algorithm in https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a
This is sorta-python valid pseduocode. Any undefined types should be Pydantic types
"""

import numpy as np
from pydantic import BaseModel


class EnvironmentState(BaseModel):
    """Placeholder for the real environment MDP state type."""


class BeliefState(BaseModel):
    """Placeholder for the real belief-over-theta state type."""


class POMDPAction(BaseModel):
    """Placeholder for the real POMDP action type."""


class Theta(BaseModel):
    """Placeholder for the real sampled-skill-parameters type."""


NUM_SAMPLES = 1  # Placeholder for the real number of theta samples to draw
STOP_ACTION = POMDPAction()  # Placeholder for the real "stop practicing" action


def solve_belief_space_expectimax(
    *,
    environment_state: EnvironmentState,
    summed_cost: int,
    belief_state: BeliefState,
    horizon: int,
) -> tuple[float, POMDPAction]:
    """Returns best value, action"""

    current_best_value = float("-inf")
    current_best_action = None

    # This corresponds to line 2 in algorithm 1 in https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a
    sample_values = []
    for _ in range(NUM_SAMPLES):
        sampled_theta = sample_theta_from_belief(belief_state=belief_state)
        current_policy_value = evaluate_policy(sampled_theta=sampled_theta)

        current_pomdp_value = score_pomdp_value_from_policy_value_and_cost(
            policy_value=current_policy_value, summed_cost=summed_cost
        )
        sample_values.append(current_pomdp_value)

    # In the POMDP, if the best action is to stop practicing, we should stop practicing.
    # Note: this does _not_ mean that the best action in the environment MDP is to do
    # nothing. It means that the best environment policy is the one we can construct
    # based on the information we have now and further practice will degrade the value
    current_best_value = np.mean(sample_values)
    current_best_action = STOP_ACTION

    assert horizon >= 0, (
        "Planning horizon should never be negative, since it decrements by 1 each time we recurse"
    )
    if horizon == 0:
        return current_best_value, current_best_action

    # This corresponds to the loop in line 6. Our goal is to (recursively) see if any
    # practice action will improve the policy
    for practice_action in get_valid_actions(environment_state=environment_state):
        value_of_state = 0
        # TODO: I'm not sure if this is sampling the next environment state or the next POMDP state
        for potential_next_environment_state, sampled_cost in sample_next_states(
            environment_state=environment_state,
            practice_action=practice_action,
            belief_state=belief_state,
        ):
            next_belief_state = update_belief_state(
                belief_state=belief_state,
                environment_state=environment_state,
                potential_next_environment_state=potential_next_environment_state,
                practice_action=practice_action,
                # The algorithm in the PDF doesn't have practice action as a parameter
                # because it's the subscript of \tau
            )

            # At this point, we want to calculate the optimal value if we take the
            # practice action and then follow the best policy. The best policy is the
            # improving policy, not the current policy, so this recurses
            value_of_next_state, _ = solve_belief_space_expectimax(
                environment_state=potential_next_environment_state,
                summed_cost=summed_cost + sampled_cost,
                belief_state=next_belief_state,
                horizon=horizon - 1,
            )

            # We need to weight the value of the next state by the transition
            # probability so that we end up with
            # value_of_state = average(value_of_next_state) over all possible next states
            value_of_state += (
                transition_probability(
                    potential_next_environment_state=potential_next_environment_state,
                    sampled_cost=sampled_cost,
                    environment_state=environment_state,
                    practice_action=practice_action,
                    belief_state=belief_state,
                )
                * value_of_next_state
            )

            if current_best_value <= value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

    return current_best_value, current_best_action


def sample_theta_from_belief(*, belief_state: BeliefState) -> Theta:
    raise NotImplementedError("This is a placeholder for the sampling function")


def evaluate_policy(*, sampled_theta: Theta) -> tuple[int]:
    """This function should take the sampled theta (which contains each skill's
    (estimated competence, estimated learning rate, estimated cost) and solve
    the environment MDP to find the optimal policy's value. This is essentially
    the planning step that Practice Makes Perfect implements by default."""

    raise NotImplementedError("This is a placeholder for the policy evaluation function")


def score_pomdp_value_from_policy_value_and_cost(*, policy_value: int, summed_cost: int) -> int:
    """This function is the `G` from https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a.
    Essentially, it encodes how to incorporate the cost required to find the policy into
    the value of the policy itself

    Assumption: For now, this means that we are not incorporating the cost of the skills
    in the run of the optimal MDP. We may eventually want to change this (e.g. to weight
    toward a shorter solution. But I will leave this as a TODO for now"""

    raise NotImplementedError("This is a placeholder for the scoring function")


def get_valid_actions(*, environment_state: EnvironmentState) -> list[POMDPAction]:
    """This are the valid _environment_ mdp actions. Though in reality, the POMDP
    actions are the same as the environment actions"""

    raise NotImplementedError("This is a placeholder for the valid actions function")


def sample_next_states(
    *,
    environment_state: EnvironmentState,
    practice_action: POMDPAction,
    belief_state: BeliefState,
) -> list[tuple[EnvironmentState, int]]:
    """TODO: I'm not sure if this is supposed to sample the next environment state or the
    next POMDP state."""

    raise NotImplementedError("This is a placeholder for the next states sampling function")


def update_belief_state(
    *,
    belief_state: BeliefState,
    environment_state: EnvironmentState,
    potential_next_environment_state: EnvironmentState,
    practice_action: POMDPAction,
) -> BeliefState:
    """See equation 11 of https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a"""

    raise NotImplementedError("This is a placeholder for the belief update function")


def transition_probability(
    *,
    potential_next_environment_state: EnvironmentState,
    sampled_cost: int,
    environment_state: EnvironmentState,
    practice_action: POMDPAction,
    belief_state: BeliefState,
) -> float:
    """See equation 11 of https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a"""

    raise NotImplementedError("This is a placeholder for the transition probability function")
