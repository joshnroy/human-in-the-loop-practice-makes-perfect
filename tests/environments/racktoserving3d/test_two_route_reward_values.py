"""Two-route rewards are configurable, opt-in, and reach every planner through the
provider: Sweep's ServedThree and GatherAndSweep's ServedFive default to 3 and 5."""

import argparse

import pytest
from pydantic import ValidationError

from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.methods.belief_space.rack_serving_domain import RACK_SERVING_DOMAIN

TWO_ROUTE = {"bulk_transfer_only": True, "two_route_rewards": True}


def _provider(**fields):
    return RackServingSkillProvider(
        env=RackServingEnvironment(config=RackServingConfig(**TWO_ROUTE, **fields))
    )


def test_the_default_rewards_are_three_and_five():
    assert _provider().goal_rewards() == {"ServedThree": 3.0, "ServedFive": 5.0}


def test_configured_rewards_reach_the_provider_and_the_rack_value_model():
    provider = _provider(served_three_reward=5.0, served_five_reward=10.0)
    assert provider.goal_rewards() == {"ServedThree": 5.0, "ServedFive": 10.0}
    domain = RACK_SERVING_DOMAIN.with_goal_rewards(provider=provider)
    assert dict(domain.bulk_rewards) == {"Sweep": 5.0, "GatherAndSweep": 10.0}


def test_without_two_route_rewards_nothing_is_priced_and_a_reward_is_refused():
    plain = RackServingSkillProvider(env=RackServingEnvironment())
    assert plain.goal_rewards() == {}
    with pytest.raises(ValidationError, match="two_route_rewards"):
        RackServingConfig(bulk_transfer_only=True, served_five_reward=10.0)


@pytest.mark.parametrize("field", ["served_three_reward", "served_five_reward"])
def test_a_reward_must_be_positive(*, field):
    with pytest.raises(ValidationError):
        RackServingConfig(**TWO_ROUTE, **{field: 0.0})


def test_the_cli_flags_set_the_rewards():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([
        "--bulk-transfer-only",
        "1",
        "--two-route-rewards",
        "1",
        "--served-three-reward",
        "5",
        "--served-five-reward",
        "10",
    ])
    problem = RackServingCli.build_problem(args=args, mode="training")
    try:
        assert RackServingSkillProvider(env=problem.env).goal_rewards() == {
            "ServedThree": 5.0,
            "ServedFive": 10.0,
        }
    finally:
        problem.env.close()
