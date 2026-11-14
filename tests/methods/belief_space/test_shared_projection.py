"""The open-list bound and J value the same node: its training forecast is computed once
for both, and every decision is the one computed with a forecast per call."""

import json
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.methods.belief_space import tossing3d_model
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from tests.methods.belief_space.test_practice_episode_budget import PRICES, ParameterisedRoutes


def _decide(*, tmp_path: Path, name: str) -> dict:
    log = tmp_path / f"{name}.jsonl"
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
        pomdp_value_bound_heuristic=True,
        pomdp_plan_beyond_session=True,
        pomdp_max_search_iterations=60,
        decision_log=log,
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={"human_reset_b": HumanCharge(cost=ChargeFunction(value=1))},
                objective_lambda=0.0001,
            )
        )
    )
    method.env.hard_reset()
    atoms = method.abstract_state(state=method.env.get_current_state())
    method.observe_practice_action_budget(remaining_actions=6, remaining_episodes=6)
    method.select_skill_to_practice(true_atoms=atoms)
    decision = json.loads(log.read_text().splitlines()[-1])
    for field in ("timestamp", "elapsed_seconds", "search_duration_seconds", "search"):
        decision.pop(field, None)
    return decision


def test_the_bound_and_j_share_one_forecast_per_node(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"refit": 0, "J": 0, "gain": 0}
    real_refit = tossing3d_model.refit_belief_state
    real_j, real_gain = Tossing3DPracticeModel.J, Tossing3DPracticeModel.optimistic_gain

    def refit(**kwargs):  # noqa: ANN003, ANN202
        counts["refit"] += 1
        return real_refit(**kwargs)

    def j(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counts["J"] += 1
        return real_j(*args, **kwargs)

    def gain(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counts["gain"] += 1
        return real_gain(*args, **kwargs)

    monkeypatch.setattr(tossing3d_model, "refit_belief_state", refit)
    monkeypatch.setattr(Tossing3DPracticeModel, "J", j)
    monkeypatch.setattr(Tossing3DPracticeModel, "optimistic_gain", gain)
    _decide(tmp_path=tmp_path, name="counted")
    assert counts["J"] > 10 and counts["gain"] > 10
    assert counts["refit"] <= max(counts["J"], counts["gain"]) + 2, counts


def test_a_shared_forecast_decides_as_a_fresh_one(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _decide(tmp_path=tmp_path, name="shared")
    monkeypatch.setattr(
        Tossing3DPracticeModel,
        "projected_belief_state",
        lambda self, *, belief_state: tossing3d_model.refit_belief_state(state=belief_state),
    )
    assert _decide(tmp_path=tmp_path, name="fresh") == shared
