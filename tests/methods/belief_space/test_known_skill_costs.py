"""--pomdp-known-skill-cost: ours can be told a skill's cost per execution, a human
skill's included, instead of learning it (Josh, 2026-11-17: in the known-curves arm,
costs are given too). A told cost holds before the skill has ever run."""

import re

from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod


def _estimates(*, known: tuple[tuple[str, float], ...]) -> dict[str, float]:
    env = TwoRoutesEnvironment(both_resets=True, chained_route_b=True)
    provider = TwoRoutesSkillProvider(env=env, both_resets=True, chained_route_b=True)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_solver="determinized_astar",
        pomdp_determinized_variant="curtis",
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
        pomdp_learn_skill_costs=True,
        pomdp_known_skill_costs=known,
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                robot_step=ChargeFunction(value=0.0),
                human_skills={"human_reset_b": HumanCharge(cost=ChargeFunction(value=0.3))},
            )
        )
    )
    costs = method._pomdp_state.expected_execution_costs
    return {re.search(r"name='(\w+)'", key).group(1): cost for key, cost in costs.items()}


def test_a_told_cost_is_used_before_the_skill_has_run() -> None:
    estimates = _estimates(known=(("RobotReset", 0.15), ("human_reset_b", 0.3)))
    assert estimates["RobotReset"] == 0.15
    assert estimates["human_reset_b"] == 0.3


def test_an_untold_skill_still_starts_from_the_prior() -> None:
    told = _estimates(known=(("RobotReset", 0.15),))
    untold = _estimates(known=())
    assert told["human_reset_b"] == untold["human_reset_b"]
    assert told["RouteA"] == untold["RouteA"]
