import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.core.problem.human.types import SAMPLED_START_RESET_SKILL_NAME
from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from hitl_pmp.presets.preset import CliPreset
from hitl_pmp.step_protocol import StepProtocolValidation

TOSSING3D_RESETS = {"reset_cube_far", "reset_cube_and_bin_near"}
ARMS = [(method, always_ask) for method in ("pomdp", "ees") for always_ask in (False, True)]


def test_the_two_reset_prices_are_the_decided_ones() -> None:
    # Josh, 2026-10-23: every arm is charged 5 per human reset. The planner arms keep
    # their objective exactly by an exact rescale (robot step 0.01, lambda 3e-4);
    # EES's plan cost was already 5 (EXP-22c's ees-lamcost5).
    assert Tossing3DArmCosts.PLANNER_HUMAN_RESET_COST == 5
    assert Tossing3DArmCosts.EES_HUMAN_RESET_COST == 5


@pytest.mark.parametrize(("method", "always_ask"), ARMS)
def test_each_arm_prices_every_human_reset_in_its_methods_own_units(
    *, method: str, always_ask: bool
) -> None:
    costs = PracticeCosts.load(
        path=Tossing3DArmCosts.cost_config(method=method, always_ask=always_ask)
    )
    expected = TOSSING3D_RESETS | ({SAMPLED_START_RESET_SKILL_NAME} if always_ask else set())
    assert set(costs.human_skills) == expected
    price = (
        Tossing3DArmCosts.EES_HUMAN_RESET_COST
        if method == "ees"
        else Tossing3DArmCosts.PLANNER_HUMAN_RESET_COST
    )
    accounting = PracticeAccounting(costs=costs)
    for name, charge in costs.human_skills.items():
        # What EesMethod.practice_action_cost plans with under the step protocol.
        assert costs.human_weight * accounting.human_charge(skill=name).cost == price
        assert charge.cost.function is None
        # A reset is still one counted step of the practice budget.
        assert charge.duration.value == 1


@pytest.mark.parametrize(("method", "always_ask"), ARMS)
def test_every_arm_shares_the_canonical_robot_pricing(*, method: str, always_ask: bool) -> None:
    canonical = PracticeCosts.load(
        path=Tossing3DArmCosts.cost_config(method="pomdp", always_ask=False)
    )
    costs = PracticeCosts.load(
        path=Tossing3DArmCosts.cost_config(method=method, always_ask=always_ask)
    )
    assert costs.robot_step == canonical.robot_step
    assert costs.human_weight == canonical.human_weight
    assert costs.objective_lambda == canonical.objective_lambda


def test_the_planners_plain_arm_is_the_presets_own_cost_file() -> None:
    preset = CliPreset.load(name="tossing3d-canonical").values(env="tossing3d", method="pomdp")
    assert preset["practice_cost_config"] == Tossing3DArmCosts.cost_config(
        method="pomdp", always_ask=False
    )


def test_the_arm_cost_files_are_not_listed_as_presets() -> None:
    assert CliPreset.names() == [
        "pouring3d-balanced",
        "pouring3d-balanced-ees",
        "tossing3d-canonical",
    ]


def test_an_always_ask_arm_on_the_presets_cost_file_is_refused() -> None:
    # The bug this fixes: the preset's file prices no sampled-start reset.
    args = Cli.parse_args(
        argv=[
            "--env", "tossing3d", "--method", "pomdp", "--preset", "tossing3d-canonical",
            "--practice-step-budget", "1000", "--output-dir", "/nonexistent-out",
            "--always-ask-reset",
        ]
    )  # fmt: skip
    with pytest.raises(ValueError, match=SAMPLED_START_RESET_SKILL_NAME):
        StepProtocolValidation.check(args=args)


def test_an_always_ask_planner_arm_on_its_own_cost_file_is_accepted() -> None:
    path = Tossing3DArmCosts.cost_config(method="pomdp", always_ask=True)
    args = Cli.parse_args(
        argv=[
            "--env", "tossing3d", "--method", "pomdp", "--preset", "tossing3d-canonical",
            "--practice-step-budget", "1000", "--output-dir", "/nonexistent-out",
            "--always-ask-reset", "--practice-cost-config", str(path),
        ]
    )  # fmt: skip
    assert args.practice_cost_config == path
    StepProtocolValidation.check(args=args)


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="astar"):
        Tossing3DArmCosts.cost_config(method="astar", always_ask=False)
