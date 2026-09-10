import numpy as np
import pytest
from pydantic import BaseModel, Field

from hitl_pmp.methods.belief_space.determinized import (
    DeterminizedAStarPlanner,
    DeterminizedSearchNode,
    solve_belief_space_determinized_astar,
)
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

    def update_belief_state(
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
        horizon: int,
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

    value, action = solve_belief_space_determinized_astar(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        max_stop_value_evaluations=5,
        num_samples=1,
        seed=4,
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
        max_stop_value_evaluations=2,
        num_samples=1,
        seed=7,
    )

    assert solve_belief_space_determinized_astar(**args) == solve_belief_space_determinized_astar(
        **args
    )
    assert solve_belief_space_determinized_astar(**args) == (pytest.approx(0.8), LEFT)


def test_graph_search_merges_duplicate_states_and_emits_compact_metrics() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(HIGH, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.8)},
    )
    trace = SearchTrace()

    result = solve_belief_space_determinized_astar(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        max_stop_value_evaluations=3,
        num_samples=1,
        seed=0,
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


def test_merged_node_propagates_deeper_value_to_every_root_action() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(HIGH, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.5), GOAL: BeliefState(value=0.9)},
    )
    trace = SearchTrace()

    solve_belief_space_determinized_astar(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        max_stop_value_evaluations=4,
        num_samples=1,
        seed=0,
        trace=trace,
    )

    action_values = {
        event["action"]["name"]: event["value"]
        for event in trace.events
        if event["event"] == "action_value"
    }
    assert action_values == {"left": pytest.approx(0.9), "right": pytest.approx(0.9)}


def test_zero_expansions_stops_and_invalid_budget_is_rejected() -> None:
    model = Model()
    args = dict(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        num_samples=1,
        seed=0,
    )
    with pytest.raises(AssertionError, match="positive"):
        solve_belief_space_determinized_astar(max_stop_value_evaluations=0, **args)
    with pytest.raises(AssertionError, match="positive"):
        solve_belief_space_determinized_astar(max_stop_value_evaluations=-1, **args)
    with pytest.raises(AssertionError, match="max_seconds"):
        solve_belief_space_determinized_astar(
            max_stop_value_evaluations=1, max_seconds=-1.0, **args
        )
    with pytest.raises(AssertionError, match="at least one compute budget"):
        solve_belief_space_determinized_astar(
            max_stop_value_evaluations=None, max_seconds=None, **args
        )


def test_zero_time_budget_returns_stop_with_summary() -> None:
    trace = SearchTrace()
    result = solve_belief_space_determinized_astar(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=Model(),
        max_stop_value_evaluations=10,
        max_seconds=0.0,
        num_samples=1,
        seed=0,
        trace=trace,
    )

    assert result == (0.2, STOP_ACTION)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["expanded_nodes"] == 0
    assert summary["termination_reason"] == "time_budget"


def test_stop_value_budget_terminates_before_generating_more_successors() -> None:
    model = Model(
        transitions={(ROOT, LEFT): [(HIGH, 0.0, 1.0)]},
        beliefs={HIGH: BeliefState(value=0.8)},
    )
    trace = SearchTrace()

    result = solve_belief_space_determinized_astar(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        model=model,
        max_stop_value_evaluations=1,
        num_samples=1,
        seed=0,
        trace=trace,
    )

    assert result == (0.2, STOP_ACTION)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["stop_value_evaluations"] == 1
    assert summary["generated_successors"] == 0
    assert summary["termination_reason"] == "stop_value_evaluation_budget"


def test_generic_heuristic_controls_frontier_order() -> None:
    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (LOW, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={
            LOW: BeliefState(value=0.3),
            HIGH: BeliefState(value=0.6),
            GOAL: BeliefState(value=0.9),
        },
    )

    def prefer_low(*, node: DeterminizedSearchNode[EnvironmentState, BeliefState]) -> float:
        return -1.0 if node.environment_state == LOW else 0.0

    planner = DeterminizedAStarPlanner[EnvironmentState, BeliefState, BaseModel, Action](
        max_stop_value_evaluations=5, seed=0, heuristic=prefer_low
    )
    value, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=2,
        model=model,  # type: ignore[arg-type]
        num_samples=1,
    )

    assert value == pytest.approx(0.9)
    assert action == LEFT


def test_planner_compute_budget_is_independent_of_expectimax_horizon() -> None:
    model = Model(
        transitions={
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.6), GOAL: BeliefState(value=0.9)},
    )
    planner = DeterminizedAStarPlanner[EnvironmentState, BeliefState, BaseModel, Action](
        max_stop_value_evaluations=3, seed=0
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
