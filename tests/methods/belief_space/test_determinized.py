import numpy as np
import pytest
from pydantic import BaseModel, Field

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.types.determinized import DeterminizedSearchNode
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from hitl_pmp.methods.belief_space.types.stop_action import STOP_ACTION


class EnvironmentState(BaseModel):
    model_config = {"frozen": True}
    name: str


class BeliefState(BaseModel):
    model_config = {"frozen": True}
    value: float


class Action(BaseModel):
    model_config = {"frozen": True}
    name: str


class Model(BaseModel):
    transitions: dict[
        tuple[EnvironmentState, Action], list[tuple[EnvironmentState, float, float]]
    ] = Field(default_factory=dict)
    beliefs: dict[EnvironmentState, BeliefState] = Field(default_factory=dict)
    evaluations: int = 0

    def sample_policy_values_from_belief(
        self, *, belief_state: BeliefState, num_samples: int
    ) -> np.ndarray:
        self.evaluations += 1
        return np.full(num_samples, belief_state.value)

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        return policy_value - summed_cost

    def J(self, *, belief_state: BeliefState, summed_cost: float, num_samples: int) -> float:
        values = self.sample_policy_values_from_belief(
            belief_state=belief_state, num_samples=num_samples
        )
        return float(
            np.mean([self.G(policy_value=value, summed_cost=summed_cost) for value in values])
        )

    def get_valid_actions(self, *, environment_state: EnvironmentState) -> list[Action]:
        return [action for state, action in self.transitions if state == environment_state]

    def transition_outcomes(
        self,
        *,
        environment_state: EnvironmentState,
        practice_action: Action,
        belief_state: BeliefState,
    ) -> list[tuple[EnvironmentState, float, float]]:
        del belief_state
        return self.transitions[environment_state, practice_action]

    def compute_next_belief_state(
        self,
        *,
        belief_state: BeliefState,
        environment_state: EnvironmentState,
        potential_next_environment_state: EnvironmentState,
        practice_action: Action,
    ) -> BeliefState:
        del belief_state, environment_state, practice_action
        return self.beliefs[potential_next_environment_state]

    def search_cache_key(
        self,
        *,
        environment_state: EnvironmentState,
        summed_cost: float,
        belief_state: BeliefState,
        horizon: int | None,
    ) -> object:
        return environment_state, summed_cost, belief_state, horizon


ROOT = EnvironmentState(name="root")
LOW = EnvironmentState(name="low")
HIGH = EnvironmentState(name="high")
GOAL = EnvironmentState(name="goal")
LEFT = Action(name="left")
RIGHT = Action(name="right")
FINISH = Action(name="finish")


def test_best_first_returns_first_action_on_best_discovered_path() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={
            LOW: BeliefState(value=0.3),
            HIGH: BeliefState(value=0.6),
            GOAL: BeliefState(value=0.9),
        },
    )

    planner = DeterminizedAStarPlanner(max_iterations=5, seed=4)
    value, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
    )

    assert value == pytest.approx(0.9)
    assert action == RIGHT


def test_samples_one_weighted_outcome_per_action_reproducibly() -> None:
    model = Model(
        transitions={(ROOT, LEFT): [(LOW, 0.0, 0.25), (HIGH, 0.0, 0.75)]},
        beliefs={LOW: BeliefState(value=0.1), HIGH: BeliefState(value=0.8)},
    )
    args = dict(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        horizon=0,
        num_samples=1,
    )

    first = DeterminizedAStarPlanner(max_iterations=2, seed=7).solve(**args)
    second = DeterminizedAStarPlanner(max_iterations=2, seed=7).solve(**args)
    assert first == second
    assert first == (pytest.approx(0.8), LEFT)


def test_observation_probability_penalizes_an_unlikely_determinization() -> None:
    model = Model(
        transitions={(ROOT, LEFT): [(LOW, 0.0, 0.1), (HIGH, 0.0, 0.9)]},
        beliefs={LOW: BeliefState(value=0.25), HIGH: BeliefState(value=0.25)},
    )
    args = dict(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
    )

    unweighted = DeterminizedAStarPlanner(
        max_iterations=1, seed=3, observation_probability_weight=0.0
    ).solve(**args)
    weighted = DeterminizedAStarPlanner(
        max_iterations=1, seed=3, observation_probability_weight=0.1
    ).solve(**args)

    assert unweighted == (pytest.approx(0.25), LEFT)
    assert weighted == (pytest.approx(0.2), STOP_ACTION)


def test_graph_search_merges_duplicate_states_and_emits_compact_metrics() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(HIGH, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.8)},
    )
    trace = SearchTrace()

    planner = DeterminizedAStarPlanner(max_iterations=3, seed=0)
    result = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
        trace=trace,
    )

    assert result == (pytest.approx(0.8), LEFT)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["expanded_nodes"] == 2
    assert summary["generated_successors"] == 2
    assert summary["unique_nodes"] == 2
    assert summary["merged_nodes"] == 1
    assert model.evaluations == 2  # root plus the one unique successor
    assert not any(event["event"] == "branch" for event in trace.events)


def test_merged_node_is_evaluated_once() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(HIGH, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.5), GOAL: BeliefState(value=0.9)},
    )
    trace = SearchTrace()

    planner = DeterminizedAStarPlanner(max_iterations=4, seed=0)
    planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
        trace=trace,
    )

    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["merged_nodes"] == 1


def test_nonpositive_iteration_budget_is_rejected() -> None:
    model = Model()
    args = dict(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        num_samples=1,
        horizon=0,
    )
    for max_iterations in (0, -1):
        planner = DeterminizedAStarPlanner(max_iterations=max_iterations, seed=0)
        with pytest.raises(AssertionError, match="positive"):
            planner.solve(**args)


def test_one_iteration_expands_only_the_root() -> None:
    model = Model(
        transitions={(ROOT, LEFT): [(HIGH, 0.0, 1.0)]},
        beliefs={HIGH: BeliefState(value=0.8)},
    )
    trace = SearchTrace()

    planner = DeterminizedAStarPlanner(max_iterations=1, seed=0)
    result = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
        trace=trace,
    )

    assert result == (0.8, LEFT)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["iterations"] == 1
    assert summary["generated_successors"] == 1
    assert summary["termination_reason"] == "iteration_budget"


def test_generic_heuristic_adds_no_domain_knowledge() -> None:
    planner = DeterminizedAStarPlanner(max_iterations=1, seed=0)
    assert (
        planner.heuristic(
            node=DeterminizedSearchNode(
                environment_state=ROOT,
                belief_state=BeliefState(value=0.2),
                summed_cost=0.0,
                depth=0,
                stop_value=0.2,
                g=0.0,
            )
        )
        == 0.0
    )


def test_planner_compute_budget_is_independent_of_expectimax_horizon() -> None:
    model = Model(
        transitions={
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.6), GOAL: BeliefState(value=0.9)},
    )
    planner = DeterminizedAStarPlanner[EnvironmentState, BeliefState, BaseModel, Action](
        max_iterations=3, seed=0
    )

    value, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,  # type: ignore[arg-type]
        num_samples=1,
    )

    assert value == pytest.approx(0.9)
    assert action == RIGHT
