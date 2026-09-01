"""Generic model protocol for belief-space expectimax."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

EnvironmentStateT = TypeVar("EnvironmentStateT", bound=BaseModel)
BeliefStateT = TypeVar("BeliefStateT", bound=BaseModel)
ThetaT = TypeVar("ThetaT", bound=BaseModel)
ActionT = TypeVar("ActionT", bound=BaseModel)


class BeliefSpaceModel(Protocol[EnvironmentStateT, BeliefStateT, ThetaT, ActionT]):
    """Domain implementations of the pseudocode's seven model functions."""

    def sample_theta_from_belief(self, *, belief_state: BeliefStateT) -> ThetaT: ...

    def sample_thetas_from_belief(
        self, *, belief_state: BeliefStateT, num_samples: int
    ) -> list[ThetaT]: ...

    def search_cache_key(
        self,
        *,
        environment_state: EnvironmentStateT,
        summed_cost: float,
        belief_state: BeliefStateT,
        horizon: int,
    ) -> object: ...

    def evaluate_policy(self, *, sampled_theta: ThetaT) -> float: ...

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float: ...

    def get_valid_actions(self, *, environment_state: EnvironmentStateT) -> list[ActionT]: ...

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentStateT,
        practice_action: ActionT,
        belief_state: BeliefStateT,
    ) -> list[tuple[EnvironmentStateT, float]]:
        """Return potential next environment states and their sampled costs."""
        ...

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentStateT,
        practice_action: ActionT,
        belief_state: BeliefStateT,
    ) -> list[tuple[EnvironmentStateT, float, float]]: ...

    def update_belief_state(
        self,
        *,
        belief_state: BeliefStateT,
        environment_state: EnvironmentStateT,
        potential_next_environment_state: EnvironmentStateT,
        practice_action: ActionT,
    ) -> BeliefStateT: ...

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentStateT,
        sampled_cost: float,
        environment_state: EnvironmentStateT,
        practice_action: ActionT,
        belief_state: BeliefStateT,
    ) -> float: ...
