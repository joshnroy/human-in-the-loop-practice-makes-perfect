"""`--pomdp-probability-weighted-gain` (Josh, 2026-11-18): Curtis A* scores a path by
each step's deployment gain weighted by the chance of reaching it with the outcome the
path assumes, and each action's cost by the chance of getting to try it -- in place of
the flat surprise charge w * -log p, whose nats the gain's reward units never matched:

    g = sum_t [ P(<t) * lambda * c(a_t)  -  P(<=t) * (V(b_t) - V(b_t-1)) ]

The fixture is the stop decision that went wrong at w = 0.6 (H 0.3, seed 0): stopping is
worth 2.38; a reset that works half the time, then a sweep3 that succeeds 0.795 of the
time, lifts it to 2.76."""

import pytest

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from tests.methods.belief_space.test_determinized import (
    Action,
    BeliefState,
    EnvironmentState,
    Model,
)

ROOT = EnvironmentState(name="root")
RESET_OK = EnvironmentState(name="reset_ok")
RESET_FAILED = EnvironmentState(name="reset_failed")
SWEEP_OK = EnvironmentState(name="sweep_ok")
SWEEP_FAILED = EnvironmentState(name="sweep_failed")
RESET = Action(name="reset")
SWEEP = Action(name="sweep3")


class LinearModel(Model):
    linear_cost_lambda: float = 1.0

    def belief_cache_key(
        self, *, environment_state: EnvironmentState, belief_state: BeliefState, horizon: object
    ) -> object:
        return environment_state, belief_state, horizon


def _model(*, reset_cost: float = 0.0, sweep_cost: float = 0.0) -> LinearModel:
    return LinearModel(
        transitions={
            (ROOT, RESET): [(RESET_OK, reset_cost, 0.5), (RESET_FAILED, reset_cost, 0.5)],
            (RESET_OK, SWEEP): [(SWEEP_OK, sweep_cost, 0.795), (SWEEP_FAILED, sweep_cost, 0.205)],
        },
        beliefs={
            RESET_OK: BeliefState(value=2.38),
            RESET_FAILED: BeliefState(value=2.38),
            SWEEP_OK: BeliefState(value=2.76),
            SWEEP_FAILED: BeliefState(value=2.0),
        },
    )


def _decide(*, model: LinearModel, weighted: bool, w: float) -> object:
    planner = DeterminizedAStarPlanner(
        max_iterations=50,
        seed=0,
        variant="curtis",
        observation_probability_weight=w,
        probability_weighted_gain=weighted,
    )
    _, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=2.38),
        model=model,
        num_samples=1,
    )
    return action


def test_the_surprise_charge_stops_practice_the_weighted_gain_continues() -> None:
    # Surprise: -0.38 + 0.6 * (0.69 + 0.23) = +0.17 > 0. Weighted: -0.5 * 0.795 * 0.38 < 0.
    assert _decide(model=_model(), weighted=False, w=0.6) != RESET
    assert _decide(model=_model(), weighted=True, w=0.0) == RESET


def test_a_cost_is_weighted_by_the_chance_of_getting_to_pay_it() -> None:
    # Gain 0.5 * 0.795 * 0.38 = 0.151. Costs paid on reaching each action: the reset
    # always (1 * 0.1), the sweep only after the reset works (0.5 * 0.25): 0.225 > 0.151,
    # stop. Weighting each cost by its own outcome as well (0.05 + 0.099) would continue.
    assert _decide(model=_model(reset_cost=0.1, sweep_cost=0.25), weighted=True, w=0.0) != RESET
    # The sweep's cost alone, 0.5 * 0.25 = 0.125 < 0.151: continue.
    assert _decide(model=_model(sweep_cost=0.25), weighted=True, w=0.0) == RESET


def test_it_replaces_the_surprise_charge_rather_than_adding_to_it() -> None:
    with pytest.raises(ValueError, match="surprise"):
        DeterminizedAStarPlanner(
            max_iterations=5,
            seed=0,
            variant="curtis",
            observation_probability_weight=0.2,
            probability_weighted_gain=True,
        )


def test_it_needs_a_linear_cost_weight_to_separate_gain_from_cost() -> None:
    with pytest.raises(ValueError, match="linear"):
        DeterminizedAStarPlanner(
            max_iterations=5,
            seed=0,
            variant="curtis",
            observation_probability_weight=0.0,
            probability_weighted_gain=True,
        ).solve(
            environment_state=ROOT,
            summed_cost=0.0,
            belief_state=BeliefState(value=2.38),
            model=Model(transitions=_model().transitions, beliefs=_model().beliefs),
            num_samples=1,
        )


def test_the_flag_reaches_the_method() -> None:
    from hitl_pmp.cli import Cli

    base = [
        "--env", "tworoutes", "--method", "pomdp", "--practice-step-budget", "10",
        "--practice-cost-config", "costs.json", "--pomdp-solver", "determinized_astar",
    ]  # fmt: skip
    assert Cli.parse_args(argv=base).pomdp_probability_weighted_gain is False
    args = Cli.parse_args(argv=[*base, "--pomdp-probability-weighted-gain"])
    assert args.pomdp_probability_weighted_gain is True
