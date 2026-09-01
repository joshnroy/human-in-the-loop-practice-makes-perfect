import pytest
from pydantic import BaseModel, Field

from hitl_pmp.methods.belief_space.expectimax import solve_belief_space_expectimax
from hitl_pmp.methods.belief_space.types import (
    STOP_ACTION,
    StopAction,
)
class EnvironmentState(BaseModel):
    model_config = {"frozen": True}

    name: str


class BeliefState(BaseModel):
    model_config = {"frozen": True}

    value: float


class Action(BaseModel):
    model_config = {"frozen": True}

    name: str


class Theta(BaseModel):
    model_config = {"frozen": True}

    value: float


INITIAL = EnvironmentState(name="initial")
READY = EnvironmentState(name="ready")
SUCCESS = EnvironmentState(name="success")
FAILURE = EnvironmentState(name="failure")
SETUP = Action(name="setup")
PRACTICE = Action(name="practice")


class Model(BaseModel):
    transitions: dict[
        tuple[EnvironmentState, Action],
        list[tuple[EnvironmentState, float, float]],
    ] = Field(default_factory=dict)
    beliefs: dict[EnvironmentState, BeliefState] = Field(default_factory=dict)
    action_beliefs: dict[Action, BeliefState] = Field(default_factory=dict)
    samples: list[float] = Field(default_factory=list)
    visits: list[BeliefState] = Field(default_factory=list)
    scale: float = 1.0

    def sample_theta_from_belief(self, *, belief_state: BeliefState) -> Theta:
        assert isinstance(belief_state, BeliefState)
        value = self.samples[len(self.visits)] if self.samples else belief_state.value
        self.visits.append(belief_state)
        return Theta(value=value)

    def sample_thetas_from_belief(
        self, *, belief_state: BeliefState, num_samples: int
    ) -> list[Theta]:
        return [
            self.sample_theta_from_belief(belief_state=belief_state) for _ in range(num_samples)
        ]

    def search_cache_key(
        self,
        *,
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int,
    ) -> object:
        return environment_state, summed_cost, belief_state, horizon

    def evaluate_policy(self, *, sampled_theta: Theta) -> float:
        assert isinstance(sampled_theta, Theta)
        return sampled_theta.value * self.scale

    def score_pomdp_value_from_policy_value_and_cost(
        self, *, policy_value: float, summed_cost: float
    ) -> float:
        return policy_value - summed_cost

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[Action]:
        return [action for state, action in self.transitions if state == environment_state]

    def sample_next_states(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: Action,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float]]:
        return [
            (state, cost) for state, cost, _ in self.transitions[environment_state, practice_action]
        ]

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: Action,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float, float]]:
        del belief_state
        return self.transitions[environment_state, practice_action]

    def update_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: Action,
    ) -> BeliefState:
        return self.action_beliefs.get(
            practice_action, self.beliefs.get(potential_next_environment_state, belief_state)
        )

    def transition_probability(
        self,
        *,
        potential_next_environment_state: EnvironmentState,
        sampled_cost: float,
        environment_state: EnvironmentState,
        practice_action: Action,
        belief_state: BeliefState,
    ) -> float:
        return next(
            probability
            for state, cost, probability in self.transitions[environment_state, practice_action]
            if state == potential_next_environment_state and cost == sampled_cost
        )


def test_horizon_zero_stops_and_subtracts_existing_cost() -> None:
    model = Model(transitions={(INITIAL, PRACTICE): [(SUCCESS, 0.0, 1.0)]})
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.25,
        belief_state=BeliefState(value=1.0),
        horizon=0,
        model=model,
    ) == (0.75, STOP_ACTION)
    assert len(model.visits) == 1


def test_averages_theta_samples_after_policy_evaluation_and_scoring() -> None:
    model = Model(samples=[1.0, 3.0, 8.0], scale=2.0)
    value, action = solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.5,
        belief_state=BeliefState(value=0.0),
        horizon=0,
        model=model,
        num_samples=3,
    )
    assert value == pytest.approx(7.5)
    assert action == STOP_ACTION
    assert len(model.visits) == 3


@pytest.mark.parametrize("horizon, expected", [(1, (2.0, STOP_ACTION)), (2, (3.0, SETUP))])
def test_looks_past_an_initially_unhelpful_setup_action(
    *, horizon: int, expected: tuple[float, Action | StopAction]
) -> None:
    model = Model(
        transitions={
            (INITIAL, SETUP): [(READY, 1.0, 1.0)],
            (READY, PRACTICE): [(SUCCESS, 1.0, 0.75), (FAILURE, 1.0, 0.25)],
        },
        beliefs={SUCCESS: BeliefState(value=6.0), FAILURE: BeliefState(value=2.0)},
    )
    assert (
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=0.0,
            belief_state=BeliefState(value=2.0),
            horizon=horizon,
            model=model,
        )
        == expected
    )


def test_stop_wins_an_exact_tie() -> None:
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=2.0),
        horizon=1,
        model=Model(transitions={(INITIAL, PRACTICE): [(INITIAL, 0.0, 1.0)]}),
    ) == (2.0, STOP_ACTION)


def test_compares_actions_only_after_summing_all_outcomes() -> None:
    model = Model(
        transitions={(INITIAL, PRACTICE): [(SUCCESS, 0.0, 0.5), (FAILURE, 0.0, 0.5)]},
        beliefs={SUCCESS: BeliefState(value=4.0), FAILURE: BeliefState(value=-4.0)},
    )
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=1.0),
        horizon=1,
        model=model,
    ) == (1.0, STOP_ACTION)


def test_shared_successor_is_evaluated_once_at_each_depth() -> None:
    model = Model(
        transitions={
            (INITIAL, SETUP): [(SUCCESS, 0.0, 1.0)],
            (INITIAL, PRACTICE): [(SUCCESS, 0.0, 1.0)],
        },
        beliefs={SUCCESS: BeliefState(value=2.0)},
    )
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=1,
        model=model,
    ) == (2.0, SETUP)
    assert model.visits == [BeliefState(value=0.0), BeliefState(value=2.0)]


def test_cache_distinguishes_accumulated_cost() -> None:
    model = Model(
        transitions={
            (INITIAL, SETUP): [(SUCCESS, 1.0, 1.0)],
            (INITIAL, PRACTICE): [(SUCCESS, 0.5, 1.0)],
        },
        beliefs={SUCCESS: BeliefState(value=2.0)},
    )
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=1,
        model=model,
    ) == (1.5, PRACTICE)
    assert len(model.visits) == 3


def test_separate_searches_resample_and_do_not_reuse_stale_values() -> None:
    model = Model(samples=[1.0, 3.0])
    first = solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=0,
        model=model,
    )
    model.scale = 2.0
    second = solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=0,
        model=model,
    )
    assert first == (1.0, STOP_ACTION)
    assert second == (6.0, STOP_ACTION)


def test_cache_distinguishes_beliefs_at_the_same_environment_state_and_cost() -> None:
    model = Model(
        transitions={
            (INITIAL, SETUP): [(SUCCESS, 0.0, 1.0)],
            (INITIAL, PRACTICE): [(SUCCESS, 0.0, 1.0)],
        },
        action_beliefs={SETUP: BeliefState(value=1.0), PRACTICE: BeliefState(value=2.0)},
    )
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=1,
        model=model,
    ) == (2.0, PRACTICE)
    assert model.visits == [
        BeliefState(value=0.0),
        BeliefState(value=1.0),
        BeliefState(value=2.0),
    ]


def test_averaging_identical_samples_preserves_value() -> None:
    assert solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.75),
        horizon=0,
        model=Model(),
        num_samples=2,
    ) == (0.75, STOP_ACTION)


@pytest.mark.parametrize(
    "branches",
    [
        [],
        [(SUCCESS, 0.0, 0.4), (FAILURE, 0.0, 0.4)],
        [(SUCCESS, 0.0, 0.0), (FAILURE, 0.0, 1.0)],
        [(SUCCESS, 0.0, -0.1)],
        [(SUCCESS, 0.0, float("nan"))],
        [(SUCCESS, 0.0, float("inf"))],
    ],
)
def test_rejects_invalid_chance_outcomes(
    *, branches: list[tuple[EnvironmentState, float, float]]
) -> None:
    with pytest.raises(AssertionError):
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=0.0,
            belief_state=BeliefState(value=0.0),
            horizon=1,
            model=Model(transitions={(INITIAL, PRACTICE): branches}),
        )


def test_sampled_successors_need_not_be_distinct() -> None:
    model = Model(
        transitions={(INITIAL, PRACTICE): [(SUCCESS, 0.0, 0.5), (SUCCESS, 0.0, 0.5)]},
        beliefs={SUCCESS: BeliefState(value=2.0)},
    )
    value, action = solve_belief_space_expectimax(
        environment_state=INITIAL,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.0),
        horizon=1,
        model=model,
    )
    assert value == 2.0
    assert action == PRACTICE


def test_rejects_negative_horizon_before_evaluating_model() -> None:
    model = Model()
    with pytest.raises(AssertionError, match="horizon must be non-negative"):
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=0.0,
            belief_state=BeliefState(value=0.0),
            horizon=-1,
            model=model,
        )
    assert not model.visits


@pytest.mark.parametrize("num_samples", [0, -1])
def test_rejects_nonpositive_sample_count(*, num_samples: int) -> None:
    with pytest.raises(AssertionError, match="num_samples must be positive"):
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=0.0,
            belief_state=BeliefState(value=0.0),
            horizon=0,
            model=Model(),
            num_samples=num_samples,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_stop_value(*, value: float) -> None:
    with pytest.raises(AssertionError, match="stop value must be finite"):
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=0.0,
            belief_state=BeliefState(value=value),
            horizon=0,
            model=Model(),
        )


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("accumulated", [True, False])
def test_rejects_invalid_costs(*, cost: float, accumulated: bool) -> None:
    model = Model(transitions={(INITIAL, PRACTICE): [(SUCCESS, cost, 1.0)]})
    with pytest.raises(AssertionError, match="cost must be finite and non-negative"):
        solve_belief_space_expectimax(
            environment_state=INITIAL,
            summed_cost=cost if accumulated else 0.0,
            belief_state=BeliefState(value=0.0),
            horizon=1,
            model=model,
        )
