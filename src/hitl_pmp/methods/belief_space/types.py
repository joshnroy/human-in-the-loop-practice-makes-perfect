"""The state, action, and parameter types in the belief-space pseudocode."""

from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict


class EnvironmentState(BaseModel):
    """Environment MDP state; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class BeliefState(BaseModel):
    """Belief over theta; domain subclasses supply hashable fields."""

    model_config = ConfigDict(frozen=True)


class Theta(BaseModel):
    """Sampled skill parameters used to construct and evaluate a policy."""

    model_config = ConfigDict(frozen=True)


class StopAction(BaseModel):
    """Sentinel selected when further practice has no value."""

    model_config = ConfigDict(frozen=True)


STOP_ACTION = StopAction()
NUM_SAMPLES = 1
ActionT = TypeVar("ActionT")


class BeliefSpaceModel(Protocol[ActionT]):
    """Domain implementations of the pseudocode's seven model functions."""

    def sample_theta_from_belief(self, *, belief_state: BeliefState) -> Theta: ...

    def sample_thetas_from_belief(
        self, *, belief_state: BeliefState, num_samples: int
    ) -> list[Theta]: ...

    def search_cache_key(
        self,
        *,
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int,
    ) -> object: ...

    def evaluate_policy(self, *, sampled_theta: Theta) -> float: ...

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float: ...

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[ActionT]: ...

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: ActionT,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float]]:
        """Return potential next environment states and their sampled costs."""
        ...

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: ActionT,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float, float]]: ...

    def update_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: ActionT,
    ) -> BeliefState: ...

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentState,
        sampled_cost: float,
        environment_state: EnvironmentState,
        practice_action: ActionT,
        belief_state: BeliefState,
    ) -> float: ...
