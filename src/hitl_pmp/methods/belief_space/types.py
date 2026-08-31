"""The state, action, and parameter types in the belief-space pseudocode."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class EnvironmentState(BaseModel):
    """Environment MDP state; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class BeliefState(BaseModel):
    """Belief over theta; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class POMDPAction(BaseModel):
    """Practice action; domain subclasses supply the action's parameters."""

    model_config = ConfigDict(frozen=True)


class Theta(BaseModel):
    """Sampled skill parameters used to construct and evaluate a policy."""

    model_config = ConfigDict(frozen=True)


STOP_ACTION = POMDPAction()
NUM_SAMPLES = 1


class BeliefSpaceModel(Protocol):
    """Domain implementations of the pseudocode's seven model functions."""

    def sample_theta_from_belief(self, *, belief_state: BeliefState) -> Theta: ...

    def evaluate_policy(self, *, sampled_theta: Theta) -> float: ...

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float: ...

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[POMDPAction]: ...

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float]]:
        """Return potential next environment states and their sampled costs."""
        ...

    def update_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: POMDPAction,
    ) -> BeliefState: ...

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentState,
        sampled_cost: float,
        environment_state: EnvironmentState,
        practice_action: POMDPAction,
        belief_state: BeliefState,
    ) -> float: ...
