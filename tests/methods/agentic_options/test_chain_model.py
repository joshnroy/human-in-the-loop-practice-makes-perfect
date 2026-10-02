"""Exercise the real expectimax search against a small auditable skill-chain."""

import pytest

from hitl_pmp.methods.agentic_options.chain_model import SkillChainModel
from hitl_pmp.methods.agentic_options.types import ClusterState, OptionAction
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.stop_action import StopAction

from .support import CountForecast, library


def test_deeper_search_chooses_help_that_enables_two_future_practices() -> None:
    forecast = CountForecast()
    chain = SkillChainModel(manifest=library(), learner_belief=forecast)
    belief = Tossing3DBeliefState(skill_beliefs={})
    before = belief.model_dump_json()
    planner = ExpectimaxPlanner(use_model_j=True)
    arguments = {
        "environment_state": ClusterState(cluster_id="blocked"),
        "summed_cost": 0.0,
        "belief_state": belief,
        "model": chain,
    }

    shallow_value, shallow_action = planner.solve(**arguments, horizon=2)
    deep_value, deep_action = planner.solve(**arguments, horizon=3)

    assert isinstance(shallow_action, StopAction)
    assert shallow_value == 0.0
    assert deep_action == OptionAction(option_id="reset_near")
    # Two practices are worth 8, minus 5 for help and 2 for robot attempts.
    assert deep_value == pytest.approx(1.0)
    assert belief.model_dump_json() == before
    assert forecast.real_observations == []
    assert forecast.session_inputs == []


def test_success_and_failure_remain_distinct_when_destination_cluster_is_identical() -> None:
    chain = SkillChainModel(manifest=library(), learner_belief=CountForecast())
    arguments = {
        "environment_state": ClusterState(cluster_id="near"),
        "practice_action": OptionAction(option_id="toss"),
        "belief_state": Tossing3DBeliefState(skill_beliefs={}),
    }
    branches = chain.transition_outcomes(**arguments)

    assert len(branches) == 2
    assert {state.cluster_id for state, _, _ in branches} == {"near"}
    assert {state.outcome for state, _, _ in branches} == {"success", "failure"}
    for successor, cost, probability in branches:
        assert cost == 1.0
        assert probability == 0.5
        assert (
            chain.transition_probability(
                **arguments, potential_next_environment_state=successor, sampled_cost=cost
            )
            == 0.5
        )


def test_empirical_destinations_do_not_turn_failures_into_self_loops() -> None:
    chain = SkillChainModel(manifest=library(), learner_belief=CountForecast())
    for destination in ("near", "blocked", "blocked"):
        chain.record_destination(
            source="near", option_id="toss", outcome="failure", destination=destination
        )
    arguments = {
        "environment_state": ClusterState(cluster_id="near"),
        "practice_action": OptionAction(option_id="toss"),
        "belief_state": Tossing3DBeliefState(skill_beliefs={}),
    }
    outcomes = {
        (state.cluster_id, state.outcome): probability
        for state, _, probability in chain.transition_outcomes(**arguments)
    }
    assert outcomes == pytest.approx({
        ("near", "success"): 0.5,
        ("near", "failure"): 1 / 6,
        ("blocked", "failure"): 1 / 3,
    })
    chain.invalidate_code_revision()
    assert len(chain.transition_outcomes(**arguments)) == 2
    assert chain.counts == {}


def test_unresolved_or_unknown_destinations_are_rejected() -> None:
    chain = SkillChainModel(manifest=library(), learner_belief=CountForecast())
    for outcome, destination in (("unresolved", "near"), ("failure", "invented")):
        with pytest.raises(ValueError):
            chain.record_destination(
                source="near", option_id="toss", outcome=outcome, destination=destination
            )
    assert chain.counts == {}
