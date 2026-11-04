"""A run that prices human resets is priced by a costs file, and nothing else.

Domains that offer a human reset run only under the step protocol, which always has a
`--practice-cost-config`; the paper domains keep the cycle-mode PracticeLoop and need
none. No skill carries a price of its own.
"""

import pytest
from pydantic import ValidationError

from hitl_pmp.cli import STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.core.method.types import Skill
from hitl_pmp.full_agentic.runner import FullAgenticRunner
from hitl_pmp.step_protocol import StepProtocolValidation
from tests.practice_cost_fixtures import CostFixtures

COSTS = CostFixtures.BY_ENV["tossing3d"]


def test_a_skill_has_no_price_of_its_own() -> None:
    assert "practice_cost" not in Skill.model_fields
    with pytest.raises(ValidationError):
        Skill(
            name="s",
            parameters=(),
            preconditions=frozenset(),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            param_dim=0,
            practice_cost=1.0,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("env", sorted(STEP_PROTOCOL_ENVIRONMENTS))
def test_a_domain_with_human_resets_refuses_the_cycle_mode_loop(*, env: str) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", env, "--method", "ees"])


@pytest.mark.parametrize("env", sorted(STEP_PROTOCOL_ENVIRONMENTS))
def test_the_step_protocol_requires_a_costs_file(
    *, env: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", env, "--method", "ees", "--practice-step-budget", "10"])
    assert "--practice-cost-config" in capsys.readouterr().err


def test_the_step_protocol_validation_refuses_a_run_without_a_costs_file() -> None:
    args = Cli.parse_args(
        argv=[
            "--env", "tossing3d", "--method", "ees", "--practice-step-budget", "10",
            "--practice-cost-config", str(COSTS),
        ]
    )  # fmt: skip
    args.practice_cost_config = None
    with pytest.raises(ValueError, match="--practice-cost-config"):
        StepProtocolValidation.check(args=args)


@pytest.mark.parametrize("env", ["lightswitch", "tossingroom", "ballring"])
def test_the_paper_domains_keep_the_cycle_mode_loop_without_a_costs_file(*, env: str) -> None:
    args = Cli.parse_args(argv=["--env", env, "--method", "ees"])
    assert args.practice_cost_config is None


@pytest.mark.parametrize("env", sorted(STEP_PROTOCOL_ENVIRONMENTS))
def test_no_domain_takes_a_human_reset_price_flag(*, env: str) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=[
                "--env", env, "--method", "ees", "--practice-step-budget", "10",
                "--practice-cost-config", str(COSTS), "--human-reset-practice-cost", "5",
            ]
        )  # fmt: skip


def test_the_full_agentic_runner_requires_a_costs_file() -> None:
    with pytest.raises(SystemExit):
        FullAgenticRunner.parse_args(
            argv=["--env", "tossing3d", "--output", "o", "--sandbox-settings", "s"]
        )


def test_no_skill_provider_carries_a_reset_price() -> None:
    from hitl_pmp.environments.kitchen_recycling3d.skill_provider import (
        KitchenRecyclingSkillProvider,
    )
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
    from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
    from hitl_pmp.environments.sweepoffshelf3d.skill_provider import (
        SweepOffShelf3DSkillProvider,
    )
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider

    for provider in (
        Tossing3DSkillProvider,
        KitchenRecyclingSkillProvider,
        KitchenSkillProvider,
        FrozenPourSkillProvider,
        SweepOffShelf3DSkillProvider,
    ):
        fields = set(provider.model_fields)
        assert not {f for f in fields if f.endswith("practice_cost")}, provider


def test_always_ask_takes_no_price_from_the_domain() -> None:
    """Its price is the costs file's `reset_to_sampled_start`, or one given explicitly;
    never a skill's own."""
    from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import AlwaysAskReset

    assert AlwaysAskReset().cost is None
    assert AlwaysAskReset(cost=7.0).cost == 7.0
    assert not hasattr(AlwaysAskReset, "for_provider")
