"""`--pomdp-plan-beyond-session`: ours' plan is bounded by the practice budget that is
left and by the search's expansion cap, not by the actions left in the session (Josh,
2026-11-13: "budget is more than practice session, right?")."""

import json
from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from tests.methods.belief_space.test_practice_episode_budget import (
    PRICES,
    ROUTES,
    ParameterisedRoutes,
)


def _decide(*, tmp_path: Path, name: str, beyond: bool | None, **budget: int) -> dict:
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
        decision_log=log,
        **({} if beyond is None else {"pomdp_plan_beyond_session": beyond}),
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
    method.observe_practice_action_budget(**budget)
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


def _search(*, decision: dict) -> list[dict]:
    volatile = ("timestamp", "elapsed_seconds", "search_elapsed_seconds")
    return [{k: v for k, v in e.items() if k not in volatile} for e in decision["search"]]


LAST_ACTION = {"remaining_actions": 1, "remaining_steps": 1_000_000}


def test_off_the_plan_ends_with_the_session_as_it_always_has(*, tmp_path: Path) -> None:
    default = _decide(tmp_path=tmp_path, name="default", beyond=None, **LAST_ACTION)
    off = _decide(tmp_path=tmp_path, name="off", beyond=False, **LAST_ACTION)
    assert _search(decision=off) == _search(decision=default)
    assert "plan_beyond_session" not in default
    assert max(len(plan) for plan in _plans(decision=off)) == 1
    assert _summary(decision=off)["effective_action_horizon"] == 1


def test_on_the_last_action_of_a_session_can_begin_a_longer_plan(*, tmp_path: Path) -> None:
    decision = _decide(tmp_path=tmp_path, name="on", beyond=True, **LAST_ACTION)
    assert decision["plan_beyond_session"] is True
    # The session's own count is still what the log says is left of it.
    assert decision["remaining_practice_actions"] == 1
    summary = _summary(decision=decision)
    assert summary["effective_action_horizon"] is None
    assert summary["selected_path_depth"] > 1
    assert max(len(plan) for plan in _plans(decision=decision)) > 1


def test_on_the_plan_is_what_it_is_with_no_session_limit_at_all(*, tmp_path: Path) -> None:
    # Bounded by the expansion cap and the budget alone: the same search as a session
    # with more actions left than the cap could ever reach.
    on = _decide(tmp_path=tmp_path, name="on", beyond=True, **LAST_ACTION)
    assert _summary(decision=on)["max_iterations"] == 100
    assert _summary(decision=on)["max_depth_reached"] <= 100


def test_on_the_plan_is_still_cut_by_the_steps_that_are_left(*, tmp_path: Path) -> None:
    decision = _decide(
        tmp_path=tmp_path, name="steps", beyond=True, remaining_actions=1, remaining_steps=5
    )
    plans = _plans(decision=decision)
    assert max(len(plan) for plan in plans) > 1
    for plan in plans:
        assert sum(PRICES.get(name, 1.0) for name in plan) <= 5
    assert _summary(decision=decision)["selected_path_steps"] <= 5


def test_on_the_plan_is_still_cut_by_the_tries_that_are_left(*, tmp_path: Path) -> None:
    decision = _decide(
        tmp_path=tmp_path,
        name="tries",
        beyond=True,
        remaining_actions=1,
        remaining_steps=1_000_000,
        remaining_episodes=2,
    )
    plans = _plans(decision=decision)
    assert max(len(plan) for plan in plans) > 1
    for plan in plans:
        assert sum(name in ROUTES for name in plan) <= 2


def test_the_flag_is_off_by_default_and_reaches_the_method() -> None:
    base = [
        "--env", "tworoutes", "--method", "pomdp", "--practice-step-budget", "10",
        "--practice-cost-config", "costs.json", "--pomdp-solver", "determinized_astar",
    ]  # fmt: skip
    assert Cli.parse_args(argv=base).pomdp_plan_beyond_session is False
    assert Cli.parse_args(argv=[*base, "--pomdp-plan-beyond-session"]).pomdp_plan_beyond_session


def test_the_flag_is_refused_for_expectimax(*, tmp_path: Path) -> None:
    import pytest

    from tests.environments.tworoutes.run_fixtures import TwoRoutesRun

    with pytest.raises(ValueError, match="determinized_astar"):
        TwoRoutesRun.run(
            directory=tmp_path,
            method="pomdp",
            extra=(
                "--pomdp-solver", "expectimax", "--pomdp-observation-probability-weight", "0.01",
                "--pomdp-plan-beyond-session",
            ),
        )  # fmt: skip


def test_a_run_recorded_before_the_flag_existed_still_resumes() -> None:
    from hitl_pmp.step_protocol import StepProtocolResume

    before = {"seed": "0", "env": "racktoserving3d"}
    assert StepProtocolResume.comparable(arguments=before) == StepProtocolResume.comparable(
        arguments={**before, "pomdp_plan_beyond_session": "False"}
    )
    assert StepProtocolResume.comparable(arguments=before) != StepProtocolResume.comparable(
        arguments={**before, "pomdp_plan_beyond_session": "True"}
    )
