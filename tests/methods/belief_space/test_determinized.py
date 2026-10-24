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


@pytest.mark.parametrize("remaining_actions", [0, 1, 2])
def test_real_action_budget_limits_paths_and_stops_at_zero(*, remaining_actions: int) -> None:
    model = Model(
        transitions={
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={HIGH: BeliefState(value=0.6), GOAL: BeliefState(value=0.9)},
    )
    trace = SearchTrace()
    value, action = DeterminizedAStarPlanner(max_iterations=100, seed=0).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=99,
        remaining_actions=remaining_actions,
        model=model,
        num_samples=1,
        trace=trace,
    )

    assert value == pytest.approx([0.2, 0.6, 0.9][remaining_actions])
    assert action == (STOP_ACTION if remaining_actions == 0 else RIGHT)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["effective_action_horizon"] == remaining_actions
    assert summary["selected_path_depth"] == remaining_actions
    assert summary["max_depth_reached"] == remaining_actions
    assert summary["action_transitions_evaluated"] == remaining_actions
    assert summary["action_horizon_terminal_nodes"] == 1
    assert summary["termination_reason"] == "action_horizon_exhausted"


def test_same_state_with_a_remaining_action_is_not_closed_by_a_terminal_visit() -> None:
    class DepthPenaltyPlanner(DeterminizedAStarPlanner):
        def heuristic(self, *, node: DeterminizedSearchNode) -> float:
            # Visit HIGH through LOW first, exhausting the two-action budget.
            # The direct visit must still expand HIGH -> GOAL with its final slot.
            return 0.15 if node.environment_state == HIGH and node.depth == 1 else 0.0

    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (LOW, FINISH): [(HIGH, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={
            LOW: BeliefState(value=0.3),
            HIGH: BeliefState(value=0.4),
            GOAL: BeliefState(value=0.9),
        },
    )
    value, action = DepthPenaltyPlanner(max_iterations=100, seed=0).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        remaining_actions=2,
        model=model,
        num_samples=1,
    )

    assert value == pytest.approx(0.9)
    assert action == RIGHT


def test_negative_remaining_action_budget_is_rejected() -> None:
    with pytest.raises(AssertionError, match="remaining_actions must be non-negative"):
        DeterminizedAStarPlanner(max_iterations=100, seed=0).solve(
            environment_state=ROOT,
            summed_cost=0.0,
            belief_state=BeliefState(value=0.2),
            horizon=0,
            remaining_actions=-1,
            model=Model(),
            num_samples=1,
        )


@pytest.mark.parametrize("remaining_actions", [None, 3])
@pytest.mark.parametrize("log_full_search_tree", [False, True])
def test_cheaper_closed_path_propagates_without_resampling(
    *, remaining_actions: int | None, log_full_search_tree: bool
) -> None:
    merged = EnvironmentState(name="merged")
    safe = EnvironmentState(name="safe")
    dead = EnvironmentState(name="dead")
    safe_action = Action(name="safe")
    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 0.5), (dead, 0.0, 0.5)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (ROOT, safe_action): [(safe, 0.0, 1.0)],
            (LOW, FINISH): [(merged, 0.0, 1.0)],
            (HIGH, FINISH): [(merged, 0.0, 1.0)],
            (merged, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={
            LOW: BeliefState(value=0.6),
            HIGH: BeliefState(value=0.3),
            merged: BeliefState(value=0.7),
            safe: BeliefState(value=0.77),
            dead: BeliefState(value=0.0),
            GOAL: BeliefState(value=0.8),
        },
    )
    trace = SearchTrace()
    planner = DeterminizedAStarPlanner(
        max_iterations=100,
        seed=3,
        observation_probability_weight=0.1,
        log_full_search_tree=log_full_search_tree,
    )
    value, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=3,
        remaining_actions=remaining_actions,
        model=model,
        num_samples=1,
        trace=trace,
    )

    # LEFT closes merged and GOAL before RIGHT is expanded. RIGHT reaches
    # merged at the same depth with no surprise penalty; propagate that saving.
    assert (value, action) == (pytest.approx(0.8), RIGHT)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["reopened_nodes"] == 2
    assert summary["cached_successor_relaxations"] == 1
    assert summary["generated_successors"] == 6
    assert summary["action_transitions_evaluated"] == 6
    assert summary["selected_path_depth"] == 3
    assert model.evaluations == 6  # cached J is also reused on reopening
    paths = [row for row in trace.events if row["event"] == "action_value"]
    assert len(paths) == 3
    right_path = next(row for row in paths if row["action"] == RIGHT.model_dump())
    assert right_path["value"] == pytest.approx(0.8)
    assert right_path["path_cost_g"] == pytest.approx(-0.6)
    assert right_path["objective_improvement"] == pytest.approx(0.6)
    assert right_path["observation_surprise"] == pytest.approx(0.0)
    assert len(right_path["best_action_path"]) == 3
    tree_edges = [row for row in trace.events if row["event"] == "tree_edge"]
    assert bool(tree_edges) == log_full_search_tree
    if log_full_search_tree:
        assert sum(row.get("cached_successor", False) for row in tree_edges) == 1
    expected_rng = np.random.default_rng(3)
    expected_rng.random(6)
    assert planner.rng.bit_generator.state == expected_rng.bit_generator.state


def test_cheaper_queued_path_discards_the_stale_entry() -> None:
    dead = EnvironmentState(name="dead")
    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 0.5), (dead, 0.0, 0.5)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
            (LOW, FINISH): [(GOAL, 0.0, 1.0)],
            (HIGH, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={
            LOW: BeliefState(value=0.4),
            HIGH: BeliefState(value=0.3),
            dead: BeliefState(value=0.0),
            GOAL: BeliefState(value=0.2),
        },
    )
    trace = SearchTrace()
    DeterminizedAStarPlanner(max_iterations=100, seed=3).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=3,
        remaining_actions=3,
        model=model,
        num_samples=1,
        trace=trace,
    )
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["stale_queue_entries"] == 1
    assert summary["reopened_nodes"] == 0
    assert summary["expanded_nodes"] == 4


def test_path_relaxation_ignores_telescoping_roundoff() -> None:
    assert not DeterminizedAStarPlanner.strictly_improves(candidate=0.01 - 3.5e-18, previous=0.01)
    assert DeterminizedAStarPlanner.strictly_improves(candidate=0.01 - 1e-8, previous=0.01)


def test_first_expansion_keeps_model_sample_and_value_call_order() -> None:
    class TracedModel(Model):
        calls: list[str] = Field(default_factory=list)

        def J(self, *, belief_state: BeliefState, summed_cost: float, num_samples: int) -> float:
            self.calls.append(f"J:{belief_state.value}")
            return super().J(
                belief_state=belief_state, summed_cost=summed_cost, num_samples=num_samples
            )

        def transition_outcomes(
            self,
            *,
            environment_state: EnvironmentState,
            practice_action: Action,
            belief_state: BeliefState,
        ) -> list[tuple[EnvironmentState, float, float]]:
            self.calls.append(f"outcomes:{practice_action.name}")
            return super().transition_outcomes(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )

    model = TracedModel(
        transitions={
            (ROOT, LEFT): [(LOW, 0.0, 1.0)],
            (ROOT, RIGHT): [(HIGH, 0.0, 1.0)],
        },
        beliefs={LOW: BeliefState(value=0.3), HIGH: BeliefState(value=0.6)},
    )
    DeterminizedAStarPlanner(max_iterations=1, seed=0).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
    )
    assert model.calls == ["J:0.2", "outcomes:left", "J:0.3", "outcomes:right", "J:0.6"]


class DrawKeyedModel(Model):
    """A model whose pick outcome reads one belief, shared by every state.

    Under the "state" scope a state names its draw by `draw_keys` (a reset that changes
    only bookkeeping maps to the state it left); under "skill" only the action does."""

    draw_keys: dict[str, str] = Field(default_factory=dict)

    def outcome_draw_key(
        self,
        *,
        environment_state: EnvironmentState,
        belief_state: BeliefState,
        action: Action,
        scope: str,
    ) -> object:
        del belief_state
        if scope == "skill":
            return action
        return self.draw_keys.get(environment_state.name, environment_state.name), action


AFTER_RESET = EnvironmentState(name="after_reset")
MOVED = EnvironmentState(name="moved")
PICK = Action(name="pick")
RESET = Action(name="reset")
MOVE = Action(name="move")


def _picks(*, state: EnvironmentState) -> list[tuple[EnvironmentState, float, float]]:
    return [
        (EnvironmentState(name=f"picked_{state.name}"), 0.0, 0.5),
        (EnvironmentState(name=f"dropped_{state.name}"), 0.0, 0.5),
    ]


def _reroll_model(*, cls: type[DrawKeyedModel] = DrawKeyedModel) -> DrawKeyedModel:
    """Two cheap resets: one leaves the state as it was but for bookkeeping, one moves
    it to a different state. The pick's outcome reads the same belief from all three."""
    states = (ROOT, AFTER_RESET, MOVED)
    return cls(
        transitions={
            (ROOT, RESET): [(AFTER_RESET, 0.01, 1.0)],
            (ROOT, MOVE): [(MOVED, 0.01, 1.0)],
            **{(state, PICK): _picks(state=state) for state in states},
        },
        beliefs={
            AFTER_RESET: BeliefState(value=0.5),
            MOVED: BeliefState(value=0.5),
            **{
                outcome: BeliefState(value=0.9 if outcome.name.startswith("picked") else 0.3)
                for state in states
                for outcome, _, _ in _picks(state=state)
            },
        },
        draw_keys={AFTER_RESET.name: ROOT.name},
    )


def _reroll_choices(*, draw_scope: str) -> set[object]:
    return {
        DeterminizedAStarPlanner(
            max_iterations=20,
            seed=seed,
            observation_probability_weight=0.0,
            draw_scope=draw_scope,
        ).solve(
            environment_state=ROOT,
            summed_cost=0.0,
            belief_state=BeliefState(value=0.5),
            horizon=0,
            model=_reroll_model(),
            num_samples=1,
        )[1]
        for seed in range(60)
    }


def test_per_node_draws_let_either_cheap_reset_re_roll_the_pick() -> None:
    assert {RESET, MOVE} <= _reroll_choices(draw_scope="node")


def test_per_state_draws_stop_only_the_reset_that_leaves_the_state_unchanged() -> None:
    choices = _reroll_choices(draw_scope="state")
    assert RESET not in choices
    assert MOVE in choices


def test_per_skill_draws_stop_every_reset_re_roll() -> None:
    assert _reroll_choices(draw_scope="skill") == {PICK, STOP_ACTION}


def test_shared_draws_are_traced() -> None:
    trace = SearchTrace()
    DeterminizedAStarPlanner(
        max_iterations=20, seed=0, observation_probability_weight=0.0, draw_scope="skill"
    ).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.5),
        horizon=0,
        model=_reroll_model(),
        num_samples=1,
        trace=trace,
        remaining_actions=None,
    )
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["draw_scope"] == "skill"
    assert summary["reused_outcome_draws"] == 2


def test_an_unknown_draw_scope_is_rejected() -> None:
    with pytest.raises(AssertionError):
        DeterminizedAStarPlanner(max_iterations=1, seed=0, draw_scope="belief")


def test_consistent_draws_map_one_uniform_through_the_cdf_like_choice() -> None:
    outcomes = [(LOW, 0.0, 0.25), (HIGH, 0.0, 0.75)]
    planner = DeterminizedAStarPlanner(max_iterations=1, seed=0)
    assert planner.sample_outcome(outcomes=outcomes, action=LEFT, uniform=0.2)[0] == LOW
    assert planner.sample_outcome(outcomes=outcomes, action=LEFT, uniform=0.25)[0] == HIGH
    assert planner.sample_outcome(outcomes=outcomes, action=LEFT, uniform=0.99)[0] == HIGH
    # The uniform path agrees with Generator.choice given the same stream.
    choice = DeterminizedAStarPlanner(max_iterations=1, seed=5)
    uniform = np.random.default_rng(5).random()
    assert choice.sample_outcome(outcomes=outcomes, action=LEFT) == planner.sample_outcome(
        outcomes=outcomes, action=LEFT, uniform=uniform
    )


class BeliefKeyedModel(DrawKeyedModel):
    """Adds the cost-free node key Curtis et al.'s PO_DAStar uses (belief only)."""

    def belief_cache_key(
        self, *, environment_state: EnvironmentState, belief_state: BeliefState, horizon: int | None
    ) -> object:
        return environment_state, belief_state, horizon


def _curtis(
    *, weight: float, num_rollouts: int | None = None, seed: int = 0, iterations: int = 20
) -> DeterminizedAStarPlanner:
    return DeterminizedAStarPlanner(
        max_iterations=iterations,
        seed=seed,
        observation_probability_weight=weight,
        variant="curtis",
        num_rollouts=num_rollouts,
    )


def _solve(
    *, planner: DeterminizedAStarPlanner, model: Model, trace: SearchTrace | None = None
) -> tuple[float, object]:
    return planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.5),
        horizon=0,
        model=model,
        num_samples=1,
        trace=trace,
    )


def _two_outcome_model() -> BeliefKeyedModel:
    return BeliefKeyedModel(
        transitions={(ROOT, LEFT): [(LOW, 0.0, 0.25), (HIGH, 0.0, 0.75)]},
        beliefs={LOW: BeliefState(value=0.1), HIGH: BeliefState(value=0.8)},
    )


def test_curtis_enumerates_every_outcome_as_its_own_child() -> None:
    trace = SearchTrace()
    value, action = _solve(planner=_curtis(weight=0.0), model=_two_outcome_model(), trace=trace)
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    assert summary["generated_successors"] == 2
    assert summary["determinized_variant"] == "curtis"
    assert (value, action) == (pytest.approx(0.8), LEFT)


def test_curtis_charges_each_enumerated_outcome_its_own_surprise() -> None:
    # HIGH: g = -(0.8 - 0.5) - w ln 0.75 = -0.3 + 0.2877 w, which beats STOP (g = 0)
    # only below w = 1.043.
    assert _solve(planner=_curtis(weight=1.0), model=_two_outcome_model())[1] == LEFT
    assert _solve(planner=_curtis(weight=1.1), model=_two_outcome_model())[1] == STOP_ACTION


def test_curtis_never_resets_to_re_roll_a_pick() -> None:
    model = _reroll_model(cls=BeliefKeyedModel)
    for weight in (0.0, 0.01, 0.1):
        assert _solve(planner=_curtis(weight=weight), model=model)[1] == PICK


def _without_timing(*, events: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in event.items() if k not in ("timestamp", "elapsed_seconds")}
        for event in events
        if event["event"] != "search_summary"
    ]


def test_curtis_rollouts_branch_on_empirical_frequencies_reproducibly() -> None:
    traces = []
    for _ in range(2):
        trace = SearchTrace()
        _solve(
            planner=_curtis(weight=0.1, num_rollouts=10, seed=3, iterations=1),
            model=_two_outcome_model(),
            trace=trace,
        )
        traces.append(trace)
    (root,) = [event for event in traces[0].events if event["event"] == "action_value"]
    assert round(10 * root["sampled_observation_probability"]) == pytest.approx(
        10 * root["sampled_observation_probability"]
    )
    assert _without_timing(events=traces[0].events) == _without_timing(events=traces[1].events)


def test_one_rollout_from_a_known_state_is_charged_nothing() -> None:
    # PO_DAStar's p is the empirical branch frequency, so one rollout gives p = 1.
    trace = SearchTrace()
    _solve(planner=_curtis(weight=1.0, num_rollouts=1), model=_two_outcome_model(), trace=trace)
    (root,) = [event for event in trace.events if event["event"] == "action_value"]
    assert root["sampled_observation_probability"] == 1.0
    assert root["observation_surprise"] == 0.0


MIDDLE = EnvironmentState(name="middle")
CHEAP = Action(name="cheap")
DEAR = Action(name="dear")


def test_curtis_nodes_are_beliefs_so_routes_of_different_cost_merge() -> None:
    """Two routes reach MIDDLE at costs 0.1 and 0.3. Keyed with the cost they are two
    nodes; keyed by belief they are one, whose J at each route's cost is exact."""
    model = BeliefKeyedModel(
        transitions={
            (ROOT, CHEAP): [(MIDDLE, 0.1, 1.0)],
            (ROOT, DEAR): [(MIDDLE, 0.3, 1.0)],
            (MIDDLE, FINISH): [(GOAL, 0.0, 1.0)],
        },
        beliefs={MIDDLE: BeliefState(value=0.7), GOAL: BeliefState(value=0.9)},
    )
    as_is, curtis = SearchTrace(), SearchTrace()
    _solve(planner=DeterminizedAStarPlanner(max_iterations=10, seed=0), model=model, trace=as_is)
    value, action = _solve(planner=_curtis(weight=0.0, iterations=10), model=model, trace=curtis)

    def unique(*, events: list[dict]) -> int:
        return next(e for e in events if e["event"] == "search_summary")["unique_nodes"]

    assert unique(events=curtis.events) == unique(events=as_is.events) - 2
    assert action == CHEAP
    assert value == pytest.approx(0.9 - 0.1)
    dear = next(
        e for e in curtis.events if e["event"] == "action_value" and e["action"]["name"] == "dear"
    )
    assert dear["value"] == pytest.approx(0.7 - 0.3)


def test_curtis_refuses_shared_draws() -> None:
    with pytest.raises(AssertionError):
        DeterminizedAStarPlanner(max_iterations=1, seed=0, variant="curtis", draw_scope="skill")
