"""Swept interruption prices reach the environment and host accounting unchanged."""

import argparse
import json
from types import SimpleNamespace

import pytest

from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment


@pytest.mark.parametrize("cost", [0, 5, 10000])
def test_constant_cost_file_reaches_both_ledgers_without_changing_competence(
    *, tmp_path, monkeypatch, cost
):
    config = {
        "robot_step": {"value": 0.01},
        "human_skills": {HUMAN_RESET: {"cost": {"value": cost}, "duration": {"value": 1}}},
        "human_weight": 1,
        "objective_lambda": 0.0003,
    }
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(config))
    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    args = parser.parse_args(["--human-reset-success-probability", "1"])
    args.seed = 7
    args.practice_cost_config = path
    env = KitchenCli.build_problem(args=args).env
    assert env.human_interruption_cost == cost
    assert env.human_reset_success_probability == 1
    assert env.human_reset_ticks == 1

    # Exercise the actual dispatch and direct-call ledger without constructing MuJoCo.
    # The any-state physical recovery itself is covered by the native reset tests.
    controller = SimpleNamespace(
        held="pitcher", scene=SimpleNamespace(cooking_observed=lambda: False)
    )
    monkeypatch.setattr(KitchenEnvironment, "controller", lambda self: controller)
    recoveries = []
    monkeypatch.setattr(
        KitchenEnvironment,
        "reset_from_any_state",
        lambda self: recoveries.append(1) or True,
    )
    accounting = PracticeAccounting(costs=PracticeCosts.load(path=path))
    for _ in range(2):
        charge = accounting.human_charge(skill=HUMAN_RESET)
        assert env.reset_movables(destination=HUMAN_RESET)
        assert env.last_info["success"]
        accounting.record(charge=charge)
    assert len(recoveries) == env.human_interventions == 2
    assert env.interruption_cost_total == accounting.human_cost == 2 * cost
    assert accounting.human_steps == 2
    assert accounting.human_invocations[HUMAN_RESET] == 2


def test_negative_interruption_prices_are_rejected():
    with pytest.raises(ValueError):
        KitchenEnvironment(human_interruption_cost=-0.01)
