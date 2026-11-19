"""`--pomdp-mean-backup` (Josh, 2026-11-18): Curtis A* searches exactly as before, but a
decision is read off the explored graph with the outcomes of an action averaged rather
than the best one taken -- a node is worth max(stop, max_a sum_k p_k (V(child_k) - lambda
c_k)), an unexpanded node its stop value. The surprise weight then steers only which
nodes the search expands, never what they are worth; with every node expanded the
backup is expectimax's."""

import pytest

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from tests.methods.belief_space.test_determinized import Action, BeliefState, EnvironmentState
from tests.methods.belief_space.test_probability_weighted_gain import (
    RESET,
    ROOT,
    LinearModel,
    _model,
)

SAFE = Action(name="safe")
RISKY = Action(name="risky")
SAFE_DONE = EnvironmentState(name="safe_done")
RISKY_OK = EnvironmentState(name="risky_ok")
RISKY_FAILED = EnvironmentState(name="risky_failed")


def _bet(*, safe_cost: float = 0.0) -> LinearModel:
    """A sure 2.6, or a coin flip between 3.4 and 1.5 (mean 2.45); stopping is worth 2."""
    return LinearModel(
        transitions={
            (ROOT, SAFE): [(SAFE_DONE, safe_cost, 1.0)],
            (ROOT, RISKY): [(RISKY_OK, 0.0, 0.5), (RISKY_FAILED, 0.0, 0.5)],
        },
        beliefs={
            SAFE_DONE: BeliefState(value=2.6),
            RISKY_OK: BeliefState(value=3.4),
            RISKY_FAILED: BeliefState(value=1.5),
        },
    )


def _solve(*, model: LinearModel, root: float, w: float, mean: bool) -> tuple[float, object]:
    planner = DeterminizedAStarPlanner(
        max_iterations=50,
        seed=0,
        variant="curtis",
        observation_probability_weight=w,
        mean_backup=mean,
    )
    return planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=root),
        model=model,
        num_samples=1,
    )


def _expectimax(*, model: LinearModel, root: float) -> tuple[float, object]:
    return ExpectimaxPlanner(use_model_j=True).solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=root),
        model=model,
        horizon=4,
        num_samples=1,
    )


def test_curtis_bets_on_the_best_outcome_the_mean_backup_does_not() -> None:
    assert _solve(model=_bet(), root=2.0, w=0.2, mean=False)[1] == RISKY
    value, action = _solve(model=_bet(), root=2.0, w=0.2, mean=True)
    assert action == SAFE
    assert value == pytest.approx(2.6)


def test_with_every_node_expanded_it_is_expectimax() -> None:
    for model, root in ((_bet(), 2.0), (_bet(safe_cost=0.7), 2.0), (_model(), 2.38)):
        mean = _solve(model=model, root=root, w=0.2, mean=True)
        exact = _expectimax(model=model, root=root)
        assert mean[1] == exact[1]
        assert mean[0] == pytest.approx(exact[0])


def test_an_edge_cost_is_paid_on_every_outcome() -> None:
    # The sure 2.6 less its 0.7 is worth 1.9, under the coin flip's 2.45.
    assert _solve(model=_bet(safe_cost=0.7), root=2.0, w=0.2, mean=True)[1] == RISKY


def test_the_surprise_weight_no_longer_stops_practice() -> None:
    # The w = 0.6 stop decision: reset (p 0.5) then sweep3 (0.795 -> 2.76, else 2.0).
    # Averaged: 0.5 * max(2.38, 0.795 * 2.76 + 0.205 * 2.0) + 0.5 * 2.38 = 2.492 > 2.38.
    assert _solve(model=_model(), root=2.38, w=0.6, mean=False)[1] != RESET
    value, action = _solve(model=_model(), root=2.38, w=0.6, mean=True)
    assert action == RESET
    assert value == pytest.approx(0.5 * (0.795 * 2.76 + 0.205 * 2.0) + 0.5 * 2.38)


def test_it_is_a_curtis_option() -> None:
    with pytest.raises(ValueError, match="curtis"):
        DeterminizedAStarPlanner(max_iterations=5, seed=0, mean_backup=True)


def test_the_flag_reaches_the_method() -> None:
    from hitl_pmp.cli import Cli

    base = [
        "--env", "tworoutes", "--method", "pomdp", "--practice-step-budget", "10",
        "--practice-cost-config", "costs.json", "--pomdp-solver", "determinized_astar",
    ]  # fmt: skip
    assert Cli.parse_args(argv=base).pomdp_mean_backup is False
    assert Cli.parse_args(argv=[*base, "--pomdp-mean-backup"]).pomdp_mean_backup is True


def test_a_graph_deeper_than_the_recursion_limit_backs_up() -> None:
    # 2026-11-18: a real 3D Serving run explored a chain of resets (they spend no budget)
    # deeper than Python's recursion limit, and the recursive backup overflowed.
    depth = 5_000
    children = {n: {"step": {n + 1: (1.0, 0.0)}} for n in range(depth)}
    values = {n: 0.0 for n in range(depth + 1)}
    values[depth] = 1.0
    q = DeterminizedAStarPlanner.mean_backup_values(
        root_key=0, children_by_key=children, policy_value_by_key=values, cost_lambda=1.0
    )
    assert q == {"step": 1.0}


def test_a_node_reached_again_on_its_own_path_counts_as_stopping() -> None:
    children = {0: {"go": {1: (1.0, 0.0)}}, 1: {"back": {0: (1.0, 0.0)}}}
    q = DeterminizedAStarPlanner.mean_backup_values(
        root_key=0, children_by_key=children, policy_value_by_key={0: 5.0, 1: 2.0}, cost_lambda=1.0
    )
    assert q == {"go": 5.0}
