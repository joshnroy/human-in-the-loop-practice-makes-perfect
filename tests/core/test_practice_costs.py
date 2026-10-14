import sys
from types import SimpleNamespace

import pytest

from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)


def test_unit_costs_are_per_control_period_not_controller_boundary():
    ledger = PracticeAccounting()
    for _ in range(300):
        ledger.record(charge=ledger.robot_charge())
    ledger.record(charge=ledger.human_charge(skill="reset_cube_far"))
    assert ledger.total_cost == 301
    assert ledger.practice_steps == 301
    assert ledger.robot_cost == 300
    assert ledger.human_cost == 1


def test_human_prices_and_durations_are_independent_per_skill():
    costs = PracticeCosts(
        human_skills={
            "far": HumanCharge(cost=ChargeFunction(value=7), duration=ChargeFunction(value=2)),
            "near": HumanCharge(cost=ChargeFunction(value=3), duration=ChargeFunction(value=1)),
        },
        human_weight=2,
    )
    ledger = PracticeAccounting(costs=costs)
    ledger.record(charge=ledger.human_charge(skill="far"))
    ledger.record(charge=ledger.human_charge(skill="near"))
    assert ledger.total_cost == 20
    assert ledger.practice_steps == 3
    assert ledger.human_invocations == {"far": 1, "near": 1}


def test_configured_functions_receive_pre_execution_context(*, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cost_example",
        SimpleNamespace(
            cost=lambda *, context: context["robot_steps"] + 2,
            duration=lambda *, context: 3 if context["skill"] == "far" else 1,
        ),
    )
    costs = PracticeCosts(
        robot_step=ChargeFunction(function="cost_example:cost"),
        human_skills={
            "far": HumanCharge(duration=ChargeFunction(function="cost_example:duration"))
        },
    )
    ledger = PracticeAccounting(costs=costs)
    for _ in range(2):
        ledger.record(charge=ledger.robot_charge())
    ledger.record(charge=ledger.human_charge(skill="far"))
    assert ledger.robot_cost == 5
    assert ledger.practice_steps == 5
    assert PracticeCosts.model_validate_json(costs.model_dump_json()) == costs


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_function_result_is_rejected_before_execution(*, monkeypatch, value):
    monkeypatch.setitem(sys.modules, "invalid_cost", SimpleNamespace(f=lambda *, context: value))
    ledger = PracticeAccounting(
        costs=PracticeCosts(
            robot_step=ChargeFunction(function="invalid_cost:f"),
        )
    )
    with pytest.raises(ValueError):
        ledger.robot_charge()
    assert ledger.robot_steps == 0


@pytest.mark.parametrize("duration", [0, 0.5])
def test_duration_is_a_positive_whole_number_of_counted_steps(*, duration):
    ledger = PracticeAccounting(
        costs=PracticeCosts(
            human_skills={
                "far": HumanCharge(duration=ChargeFunction(value=duration)),
            }
        )
    )
    with pytest.raises(ValueError):
        ledger.human_charge(skill="far")


def test_unknown_human_skill_does_not_fall_back_to_a_price():
    with pytest.raises(ValueError, match="Unknown human skill"):
        PracticeAccounting().human_charge(skill="typo")


def test_state_dependent_charge_receives_current_host_observation(*, monkeypatch):
    import sys
    import types

    from hitl_pmp.core.practice_costs import ChargeFunction, PracticeAccounting, PracticeCosts

    module = types.ModuleType("cost_observation_test")
    module.price = lambda **kw: kw["context"]["observation"]["price"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    state = {"price": 2}
    accounting = PracticeAccounting(
        costs=PracticeCosts(robot_step=ChargeFunction(function="cost_observation_test:price"))
    )
    accounting.set_observation_provider(provider=lambda: dict(state))
    assert accounting.robot_charge().cost == 2
    state["price"] = 9
    assert accounting.robot_charge().cost == 9
