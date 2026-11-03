"""DP target identity, empirical prices, and supplied-library eligibility.

arxiv:2608.13415v1 sections 3, 4, and 5.3 motivate these contracts. DP practices
the grounding it allocated, and setup costs are charged to that grounding.
"""

import json

import pytest

from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import STOP_SKILL, _EesEpisode

from .test_dp_method import _build, _execute, _ground, _pick_setup


@pytest.mark.parametrize("groundings", ["observed", "all"])
def test_default_executes_selected_grounding_and_binds_setup_cost_to_it(*, monkeypatch, groundings):
    method, setup, target = _pick_setup(monkeypatch=monkeypatch, groundings=groundings)
    assert setup.preconditions != target.preconditions
    assert method.choose_practice_target() == [target, STOP_SKILL]
    before = method.dp_remaining_allocation()[target]
    monkeypatch.setattr(DpMethod, "plan_to", lambda self, **kwargs: [setup])
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True, pursue_goal=False)
    assert episode._practice_plan(true_atoms=setup.preconditions) == [setup, target]
    assert method._dp_active_target == target
    method.record_action_cost(ground_skill=setup)
    method.observe_execution_cost(cost=2, steps=1, complete=True)
    assert method.dp_episodes() == 0
    assert method.dp_remaining_allocation()[target] == before
    assert setup not in method._dp_costs
    method.record_action_cost(ground_skill=target)
    method.observe_execution_cost(cost=3, steps=1, complete=True)
    assert method.dp_episodes() == 1
    assert method.dp_remaining_allocation()[target] == before - 1
    assert method._dp_executed_since_solve == {target: 1}
    assert method.dp_expected_attempt_cost(ground_skill=target) == pytest.approx(5)
    assert setup not in method._dp_costs


def test_solver_price_and_diagnostics_follow_each_groundings_own_costs(*, monkeypatch, tmp_path):
    method, setup, target = _pick_setup(monkeypatch=monkeypatch, groundings="all")
    method.dp_budget_unit = "cost"
    method.dp_budget = 8
    method.dp_solver_log = tmp_path / "allocation.jsonl"
    _execute(method=method, ground=setup, cost=2)
    _execute(method=method, ground=target, cost=8)
    captured = []
    original = DpAllocator.allocate

    def allocate(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(DpAllocator, "allocate", staticmethod(allocate))
    assert method.choose_practice_target() == [target, STOP_SKILL]
    assert captured[0]["costs"][target] == pytest.approx(8)
    assert captured[0]["costs"][setup] == pytest.approx(2)
    assert captured[0].get("groups") is None
    assert method.dp_allocation()[target] == 1
    allocation = json.loads(method.dp_solver_log.read_text())
    row = next(
        row
        for row in allocation["allocation"]
        if row["objects"] == [obj.name for obj in target.objects]
    )
    assert row["attempt_price"] == pytest.approx(8)
    assert method.dp_expected_attempt_cost(ground_skill=setup) == pytest.approx(2)
    assert method.dp_expected_attempt_cost(ground_skill=target) == pytest.approx(8)


def test_unseen_reachable_deployment_skill_is_eligible_with_observed_groundings(*, monkeypatch):
    method = _build(budget=1)
    method.practice_groundings = "observed"
    target = _ground(method=method, name="PickWiper")
    assert target.skill.param_dim == 0
    monkeypatch.setattr(DpMethod, "skills", lambda self: (target.skill,))
    monkeypatch.setattr(DpMethod, "offered_partial_resets", lambda self: [])
    monkeypatch.setattr(
        DpMethod,
        "planning_progress_tasks",
        lambda self: [(target.preconditions, target.add_effects)],
    )
    assert method.practice_candidates() == []
    assert target not in method._competence_models
    assert target not in method._all_attempt_outcomes
    assert method.choose_practice_target() == [target, STOP_SKILL]
    assert method.dp_allocation()[target] > 0
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True, pursue_goal=False)
    assert episode._practice_plan(true_atoms=target.preconditions) == [target]
    assert method._dp_active_target == target
    assert target not in method._all_attempt_outcomes


def test_dp_selector_stays_outside_ees_only_behavior(*, monkeypatch):
    method = _build()
    target = _ground(method=method, name="PickWiper")
    monkeypatch.setattr(DpMethod, "choose_practice_target", lambda self: [target, STOP_SKILL])
    assert not method._uses_ees_selector()
    assert method.select_skill_to_practice(true_atoms=frozenset()) == [target, STOP_SKILL]
