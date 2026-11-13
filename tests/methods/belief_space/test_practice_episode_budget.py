"""Ours' plans may only hold as many practice tries as are left (Josh, 2026-11-13).

A try is an action whose skill has learnable parameters. Two Routes' skills have none,
so here both routes are given one; the reset stays a setup action."""

import json
from pathlib import Path

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, Skill
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

PRICES = {"RouteA": 2.0, "RouteB": 2.0, "ResetA": 1.0}
ROUTES = ("RouteA", "RouteB")


class ParameterisedRoutes(TwoRoutesSkillProvider):
    def skills(self) -> tuple[Skill, ...]:
        return tuple(
            skill.model_copy(update={"param_dim": 1}) if skill.name in ROUTES else skill
            for skill in super().skills()
        )

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        del rng
        return np.zeros(ground_skill.skill.param_dim)


def _method(*, decision_log: Path) -> Tossing3DPomdpMethod:
    env = TwoRoutesEnvironment()
    provider = ParameterisedRoutes(env=env)
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


def _decide(*, tmp_path: Path, name: str, remaining_episodes: int | None) -> dict:
    log = tmp_path / f"{name}.jsonl"
    method = _method(decision_log=log)
    method.env.hard_reset()
    atoms = method.abstract_state(state=method.env.get_current_state())
    budget = {} if remaining_episodes is None else {"remaining_episodes": remaining_episodes}
    method.observe_practice_action_budget(remaining_actions=10, remaining_steps=1_000_000, **budget)
    chosen = method.select_skill_to_practice(true_atoms=atoms)
    decision = json.loads(log.read_text().splitlines()[-1])
    decision["chosen"] = chosen[0].skill.name
    return decision


def _plans(*, decision: dict) -> list[list[str]]:
    return [
        [step["skill"]["name"] for step in e["best_action_path"]]
        for e in decision["search"]
        if e["event"] == "action_value"
    ]


def _summary(*, decision: dict) -> dict:
    return next(e for e in decision["search"] if e["event"] == "search_summary")


def test_without_an_episode_budget_the_decision_and_its_log_carry_no_episode_field(
    *, tmp_path: Path
) -> None:
    decision = _decide(tmp_path=tmp_path, name="free", remaining_episodes=None)
    assert "remaining_practice_episodes" not in decision
    assert "remaining_episodes" not in _summary(decision=decision)
    # Unbounded by tries, the best plan found practises more than once.
    assert max(sum(name in ROUTES for name in plan) for plan in _plans(decision=decision)) > 1


def test_the_tries_left_are_logged_with_every_decision(*, tmp_path: Path) -> None:
    decision = _decide(tmp_path=tmp_path, name="roomy", remaining_episodes=1_000)
    assert decision["remaining_practice_episodes"] == 1_000
    assert _summary(decision=decision)["remaining_episodes"] == 1_000


def test_an_episode_budget_every_plan_fits_changes_nothing(*, tmp_path: Path) -> None:
    free = _decide(tmp_path=tmp_path, name="free", remaining_episodes=None)
    roomy = _decide(tmp_path=tmp_path, name="roomy", remaining_episodes=1_000)
    assert (roomy["chosen"], roomy["value"]) == (free["chosen"], free["value"])
    assert _plans(decision=roomy) == _plans(decision=free)


def test_no_plan_holds_more_tries_than_are_left(*, tmp_path: Path) -> None:
    for remaining in (1, 2):
        decision = _decide(
            tmp_path=tmp_path, name=f"tight{remaining}", remaining_episodes=remaining
        )
        assert decision["chosen"] in ROUTES
        plans = _plans(decision=decision)
        assert plans
        for plan in plans:
            assert sum(name in ROUTES for name in plan) <= remaining
        assert _summary(decision=decision)["selected_path_episodes"] <= remaining
