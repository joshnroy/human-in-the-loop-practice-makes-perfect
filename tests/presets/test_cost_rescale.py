"""The 2026-10-23 rescale: every launched arm is charged 5 per human reset, and the
planner objective lambda * cost is unchanged (Josh: "this should stay mathematically
equivalent"). Before: reset 500 (planner) / 5 (EES), robot step 1, lambda 3e-6."""

import math
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from scripts.sweepoffshelf3d_launch import (
    EES_ALWAYS_ASK_COSTS,
    EES_COSTS,
    PLANNER_ALWAYS_ASK_COSTS,
    PLANNER_COSTS,
    ROBOCODE_COSTS,
)

OLD_LAMBDA, OLD_STEP, OLD_PLANNER_RESET = 3e-6, 1.0, 500.0
LAUNCHED = sorted({
    *(
        Tossing3DArmCosts.cost_config(method=method, always_ask=always_ask)
        for method, always_ask in Tossing3DArmCosts.FILES
    ),
    PLANNER_COSTS,
    PLANNER_ALWAYS_ASK_COSTS,
    EES_COSTS,
    EES_ALWAYS_ASK_COSTS,
    ROBOCODE_COSTS,
})


@pytest.mark.parametrize("path", LAUNCHED, ids=lambda path: Path(path).name)
def test_every_launched_cost_file_charges_5_per_reset_at_the_rescaled_prices(*, path: Path) -> None:
    costs = PracticeCosts.load(path=path)
    assert costs.robot_step.value == 0.01 and costs.robot_step.function is None
    assert costs.objective_lambda == 3e-4
    assert costs.human_weight == 1
    for charge in costs.human_skills.values():
        assert (charge.cost.value, charge.cost.function) == (5, None)
        assert charge.duration.value == 1


@pytest.mark.parametrize("path", LAUNCHED, ids=lambda path: Path(path).name)
def test_the_planner_objective_charge_of_a_step_and_a_reset_is_unchanged(*, path: Path) -> None:
    costs = PracticeCosts.load(path=path)
    step = costs.objective_lambda * costs.robot_step.value
    assert math.isclose(step, OLD_LAMBDA * OLD_STEP, rel_tol=1e-12)
    for charge in costs.human_skills.values():
        reset = costs.objective_lambda * costs.human_weight * charge.cost.value
        assert math.isclose(reset, OLD_LAMBDA * OLD_PLANNER_RESET, rel_tol=1e-12)
