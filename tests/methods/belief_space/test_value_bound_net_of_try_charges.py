"""The value-bound heuristic orders the open list net of what the tries must cost.

The Sweeping toy (Josh, 2026-11-14): Sweep (RouteA) is worth 5 at 0.5 + 0.25/try, Gather+
Sweep (RouteB) 10 at 0.1 + 0.05/try; every try and every reset costs 0.1. With 30 tries
the best plan is eighteen tries of Gather+Sweep (10 - 1.8 - 1.7 = 6.5). Gross of the
tries' charge, every node with eighteen tries left has the same bound, so a Sweep try
cost the open-list order no more than a Gather+Sweep try and the search spread over every
mix of the two: at 1,000 nodes its first plan stopped six or seven tries short, and once a
failure had shown that every retry needs a reset it settled for Sweep (4.5)."""

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
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount

CURVES = (("RouteA", 0.5, 0.25), ("RouteB", 0.1, 0.05), ("human_reset_b", 1.0, 0.0))


def _method(*, decision_log: Path) -> Tossing3DPomdpMethod:
    env = TwoRoutesEnvironment(both_resets=True, route_a_reward=5.0, route_b_reward=10.0)
    provider = TwoRoutesSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        practice_try_skills=frozenset({"RouteA", "RouteB"}),
        pomdp_solver="determinized_astar",
        pomdp_determinized_variant="curtis",
        pomdp_observation_probability_weight=0.01,
        pomdp_inference_engine="grid",
        pomdp_learning_rate_process_noise_std=0.0,
        pomdp_value_bound_heuristic=True,
        pomdp_plan_beyond_session=True,
        pomdp_max_search_iterations=1000,
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider).model_copy(
            update={"known_competence_curves": CURVES}
        ),
        decision_log=decision_log,
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                robot_step=ChargeFunction(value=0.1),
                human_skills={
                    "human_reset_b": HumanCharge(
                        cost=ChargeFunction(value=0.1), duration=ChargeFunction(value=1)
                    )
                },
                objective_lambda=1.0,
            )
        )
    )
    return method


def _first_decision(*, tmp_path: Path, tries: int) -> dict:
    log = tmp_path / f"tries{tries}.jsonl"
    method = _method(decision_log=log)
    method.env.hard_reset()
    atoms = method.abstract_state(state=method.env.get_current_state())
    method.observe_practice_action_budget(
        remaining_actions=10, remaining_steps=20 * tries + 20, remaining_episodes=tries
    )
    method.select_skill_to_practice(true_atoms=atoms)
    return json.loads(log.read_text().splitlines()[-1])


def _summary(*, decision: dict) -> dict:
    return next(e for e in decision["search"] if e["event"] == "search_summary")


@pytest.mark.parametrize("tries", [18, 30])
def test_ours_finds_the_eighteen_try_plan_whenever_eighteen_tries_are_left(
    *, tmp_path: Path, tries: int
) -> None:
    decision = _first_decision(tmp_path=tmp_path, tries=tries)
    assert decision["action"]["skill"]["name"] == "RouteB"
    assert _summary(decision=decision)["selected_path_episodes"] == 18
    assert decision["value"] == pytest.approx(8.1)


def _model_and_belief(*, tmp_path: Path):  # noqa: ANN202
    method = _method(decision_log=tmp_path / "unused.jsonl")
    method.env.hard_reset()
    atoms = method.abstract_state(state=method.env.get_current_state())
    method.observe_practice_action_budget(
        remaining_actions=10, remaining_steps=1000, remaining_episodes=30
    )
    # What `select_skill_to_practice` hands the search.
    method._refresh_execution_forecasts()
    return method, method._pomdp_model, method._pomdp_state, atoms


def test_a_try_is_charged_its_price_and_after_an_observed_failure_its_reset(
    *, tmp_path: Path
) -> None:
    method, model, state, atoms = _model_and_belief(tmp_path=tmp_path)
    consumes = lambda action: method.action_episodes(ground_skill=action)  # noqa: E731
    charges = model.route_try_charges(belief_state=state, consumes=consumes)
    # No failure seen: a failure is taken to change nothing, so a retry needs no reset.
    assert charges == {"RouteA": pytest.approx((0.1, 0.1)), "RouteB": pytest.approx((0.1, 0.1))}
    route_b = next(g for g in model.ground_skills if g.skill.name == "RouteB")
    failed = model.model_copy(
        update={
            "failure_effect_counts": (
                FailureEffectCount(
                    ground_skill=route_b,
                    before_atoms=frozenset(atoms),
                    was_random_exploration=False,
                    add_effects=frozenset(),
                    delete_effects=frozenset(atoms),
                ),
            )
        }
    )
    charges = failed.route_try_charges(belief_state=state, consumes=consumes)
    assert charges["RouteB"] == pytest.approx((0.1, 0.2))
    # The bound: eighteen Gather+Sweep tries, 10 - 0.1 - 17 * 0.2, over Sweep's 2.5 now.
    gain = failed.optimistic_gain(belief_state=state, tries_left=30, consumes=consumes)
    assert gain == pytest.approx(10 - 0.1 - 17 * 0.2 - 2.5)


def test_without_the_linear_objective_the_bound_is_gross_as_before(*, tmp_path: Path) -> None:
    method, model, state, _ = _model_and_belief(tmp_path=tmp_path)
    hard = model.model_copy(update={"linear_cost_lambda": None})
    consumes = lambda action: method.action_episodes(ground_skill=action)  # noqa: E731
    assert hard.route_try_charges(belief_state=state, consumes=consumes) == {}
    assert hard.optimistic_gain(
        belief_state=state, tries_left=30, consumes=consumes
    ) == pytest.approx(10 - 2.5)
