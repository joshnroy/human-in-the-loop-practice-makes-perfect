"""Ours' plans may only spend the practice steps that are left (Josh, 2026-11-12)."""

import json
from pathlib import Path

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
from tests.environments.tworoutes.run_fixtures import TwoRoutesRun

# Known prices: the better-paid route costs 50 control steps a try, the other 2.
PRICES = {"RouteA": 2.0, "RouteB": 50.0, "ResetA": 1.0}


def _method(*, decision_log: Path) -> Tossing3DPomdpMethod:
    env = TwoRoutesEnvironment()
    provider = TwoRoutesSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        pomdp_solver="determinized_astar",
        pomdp_determinized_variant="curtis",
        pomdp_observation_probability_weight=0.01,
        pomdp_inference_engine="grid",
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
        pomdp_skill_step_costs=PRICES,
        decision_log=decision_log,
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={"human_reset_b": HumanCharge(cost=ChargeFunction(value=1))},
                objective_lambda=0.0001,
            )
        )
    )
    return method


def _decide(*, tmp_path: Path, name: str, remaining_steps: int | None) -> dict:
    log = tmp_path / f"{name}.jsonl"
    method = _method(decision_log=log)
    method.env.hard_reset()
    atoms = method.abstract_state(state=method.env.get_current_state())
    if remaining_steps is None:
        method.observe_practice_action_budget(remaining_actions=10)
    else:
        method.observe_practice_action_budget(remaining_actions=10, remaining_steps=remaining_steps)
    chosen = method.select_skill_to_practice(true_atoms=atoms)
    decision = json.loads(log.read_text().splitlines()[-1])
    decision["chosen"] = chosen[0].skill.name
    return decision


def _plan(*, decision: dict) -> list[str]:
    return [
        [step["skill"]["name"] for step in e["best_action_path"]]
        for e in decision["search"]
        if e["event"] == "action_value" and e["action"] == decision["action"]
    ][0]


def test_without_a_step_budget_ours_practises_the_better_paid_route(*, tmp_path: Path) -> None:
    decision = _decide(tmp_path=tmp_path, name="free", remaining_steps=None)
    assert decision["chosen"] == "RouteB"
    assert decision["remaining_practice_steps"] is None


def test_a_step_budget_every_plan_fits_changes_nothing(*, tmp_path: Path) -> None:
    free = _decide(tmp_path=tmp_path, name="free", remaining_steps=None)
    roomy = _decide(tmp_path=tmp_path, name="roomy", remaining_steps=1_000_000)
    assert roomy["remaining_practice_steps"] == 1_000_000
    assert (roomy["chosen"], roomy["value"]) == (free["chosen"], free["value"])
    assert _plan(decision=roomy) == _plan(decision=free)


def test_a_budget_that_fits_cheap_tries_but_no_dear_one_practises_the_cheap_route(
    *, tmp_path: Path
) -> None:
    decision = _decide(tmp_path=tmp_path, name="tight", remaining_steps=20)
    assert decision["chosen"] == "RouteA"
    valued = {
        e["action"]["skill"]["name"] for e in decision["search"] if e["event"] == "action_value"
    }
    assert "RouteB" not in valued
    plan = _plan(decision=decision)
    assert sum(PRICES.get(name, 1.0) for name in plan) <= 20
    summary = next(e for e in decision["search"] if e["event"] == "search_summary")
    assert summary["selected_path_steps"] <= 20


def test_the_practice_loop_tells_ours_the_steps_left_before_every_action(*, tmp_path: Path) -> None:
    output = TwoRoutesRun.run(
        directory=tmp_path,
        method="pomdp",
        budget=20,
        extra=(
            "--pomdp-solver", "determinized_astar", "--pomdp-determinized-variant", "curtis",
            "--pomdp-observation-probability-weight", "0.01", "--pomdp-inference-engine", "grid",
            "--no-pomdp-compress-decision-log",
        ),
    )  # fmt: skip
    decisions = [
        json.loads(line) for line in (output / "pomdp_decisions.jsonl").read_text().splitlines()
    ]
    decisions = [d for d in decisions if d.get("event") == "decision"]
    assert decisions
    # Every Two Routes skill, the person's included, is one control step.
    seen = [d["remaining_practice_steps"] for d in decisions]
    assert seen[0] == 20
    assert all(isinstance(s, int) and 0 < s <= 20 for s in seen)
    assert seen == sorted(seen, reverse=True)
    assert len(set(seen)) == len(seen)
