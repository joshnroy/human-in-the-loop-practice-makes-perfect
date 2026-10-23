"""Search decisions must not depend on the order a path's costs were summed in.

Costs that are multiples of 0.01 (the 2026-10-23 rescale) are inexact in binary, so the
same multiset of actions sums to values that differ in the last bit by order: 7.21 vs
7.209999999999997. Neither the merge key nor a tie between path costs may see that.
"""

import math

import pytest

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.determinized import (
    DeterminizedNodeCacheInfo,
    DeterminizedNodeDiagnosticInfo,
    DeterminizedPathRecoveryInfo,
    DeterminizedSearchBelief,
    DeterminizedSearchQueueEntry,
)
from hitl_pmp.methods.belief_space.types.stop_action import STOP_ACTION
from hitl_pmp.methods.belief_space.types.value_resolution import ValueResolution

from . import test_expectimax as ex
from .test_determinized import HIGH, LEFT, LOW, RIGHT, ROOT, BeliefState, Model

ONE_ORDER = (0.1 + 0.2) + 0.3
OTHER_ORDER = (0.3 + 0.2) + 0.1


def _key(*, cost: float, merge_beliefs_by_counts: bool = False) -> object:
    model = Tossing3DPracticeModel(merge_beliefs_by_counts=merge_beliefs_by_counts)
    state = Tossing3DBeliefState(skill_beliefs={}, accumulated_cost=cost)
    return model.search_cache_key(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=frozenset()),
        summed_cost=cost,
        belief_state=state,
        horizon=None,
    )


@pytest.mark.parametrize("merge_beliefs_by_counts", [False, True])
def test_two_summation_orders_of_the_same_costs_share_one_search_key(
    *, merge_beliefs_by_counts: bool
) -> None:
    assert ONE_ORDER != OTHER_ORDER, "the premise: these sums differ in the last bit"
    assert _key(cost=ONE_ORDER, merge_beliefs_by_counts=merge_beliefs_by_counts) == _key(
        cost=OTHER_ORDER, merge_beliefs_by_counts=merge_beliefs_by_counts
    )
    # Genuinely different costs stay apart, at either scale.
    assert _key(cost=7.21) != _key(cost=7.22)
    assert _key(cost=721.0) != _key(cost=722.0)
    assert _key(cost=0.0) != _key(cost=3e-6)


def _entry(*, g: float, sequence: int) -> DeterminizedSearchQueueEntry:
    return DeterminizedSearchQueueEntry(
        belief=DeterminizedSearchBelief(
            environment_state=ROOT, belief_state=BeliefState(value=0.0), summed_cost=0.0
        ),
        g=g,
        diagnostic_info=DeterminizedNodeDiagnosticInfo(depth=0),
        cache_info=DeterminizedNodeCacheInfo(key=sequence, stop_value=0.0, queue_sequence=sequence),
        path_recovery_info=DeterminizedPathRecoveryInfo(first_action=STOP_ACTION),
    )


def test_path_costs_tied_up_to_roundoff_pop_in_insertion_order() -> None:
    first = _entry(g=0.1 + 0.2, sequence=1)  # 0.30000000000000004
    second = _entry(g=0.3, sequence=2)
    assert first.g != second.g
    assert first < second
    assert not second < first
    # A real difference still orders by cost.
    assert _entry(g=0.3, sequence=2) < _entry(g=0.3 + 1e-6, sequence=1)


@pytest.mark.parametrize("ulp_heavier", [LEFT, RIGHT])
def test_a_roundoff_cheaper_path_does_not_change_the_decision(*, ulp_heavier) -> None:
    """Two paths of equal cost in exact arithmetic: whichever one rounding makes a last
    bit cheaper, the search returns the same action -- the first one found."""
    heavy, light = 0.1 + 0.2, 0.3
    model = Model(
        transitions={
            (ROOT, LEFT): [(LOW, heavy if ulp_heavier == LEFT else light, 1.0)],
            (ROOT, RIGHT): [(HIGH, heavy if ulp_heavier == RIGHT else light, 1.0)],
        },
        beliefs={LOW: BeliefState(value=0.9), HIGH: BeliefState(value=0.9)},
    )
    planner = DeterminizedAStarPlanner(max_iterations=5, seed=0)
    _, action = planner.solve(
        environment_state=ROOT,
        summed_cost=0.0,
        belief_state=BeliefState(value=0.2),
        horizon=0,
        model=model,
        num_samples=1,
    )
    assert action == LEFT


@pytest.mark.parametrize("ulp_heavier", ["setup", "practice"])
def test_expectimax_actions_tied_up_to_roundoff_go_to_the_first_action(*, ulp_heavier) -> None:
    heavy, light = 0.1 + 0.2, 0.3
    model = ex.Model(
        transitions={
            (ex.INITIAL, ex.SETUP): [(ex.SUCCESS, heavy if ulp_heavier == "setup" else light, 1.0)],
            (ex.INITIAL, ex.PRACTICE): [
                (ex.FAILURE, heavy if ulp_heavier == "practice" else light, 1.0)
            ],
        },
        beliefs={ex.SUCCESS: ex.BeliefState(value=0.9), ex.FAILURE: ex.BeliefState(value=0.9)},
    )
    _, action = ExpectimaxPlanner().solve(
        model=model,
        environment_state=ex.INITIAL,
        summed_cost=0.0,
        belief_state=ex.BeliefState(value=0.2),
        horizon=1,
    )
    assert action == ex.SETUP


def test_expectimax_stop_wins_a_tie_with_an_action_up_to_roundoff() -> None:
    model = ex.Model(
        transitions={(ex.INITIAL, ex.SETUP): [(ex.SUCCESS, 0.0, 1.0)]},
        beliefs={ex.SUCCESS: ex.BeliefState(value=0.1 + 0.2)},
    )
    value, action = ExpectimaxPlanner().solve(
        model=model,
        environment_state=ex.INITIAL,
        summed_cost=0.0,
        belief_state=ex.BeliefState(value=0.3),
        horizon=1,
    )
    assert action == STOP_ACTION
    assert value == 0.3


def test_an_infeasible_value_ranks_below_every_finite_one() -> None:
    assert ValueResolution.rank(value=-math.inf) < ValueResolution.rank(value=-1e9)
    assert ValueResolution.rank(value=0.1 + 0.2) == ValueResolution.rank(value=0.3)
