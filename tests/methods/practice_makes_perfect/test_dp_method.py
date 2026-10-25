"""DpMethod: the Deliberate Practice baseline's practice-target choice on top of EES.

Built on SweepOffShelf3D's symbolic layer only (it offers human resets as well as
robot skills); nothing here touches the simulator or the planner."""

import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
)
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import (
    PARAMETERIZED_SKILL,
    AlwaysAskReset,
)
from hitl_pmp.methods.practice_makes_perfect.dp_allocator import DpAllocator
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import STOP_SKILL

STEP_PRICE = 0.01
START_RESET_PRICE = 5.0


def _build(*, budget: float = 100.0, always_ask: bool = False, accounting: bool = True):
    env = SweepOffShelf3DEnvironment()
    method = DpMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env, human_reset_practice_cost=2.0),
        dp_budget=budget,
        always_ask_reset=AlwaysAskReset(cost=START_RESET_PRICE) if always_ask else None,
    )
    if accounting:
        method.configure_practice_accounting(
            accounting=PracticeAccounting(
                costs=PracticeCosts(
                    robot_step=ChargeFunction(value=STEP_PRICE),
                    human_skills={
                        "reset_to_sampled_start": HumanCharge(
                            cost=ChargeFunction(value=START_RESET_PRICE)
                        )
                    },
                )
            )
        )
    return method


def _ground(*, method: DpMethod, name: str) -> GroundSkill:
    """The skill grounded on the first object of each parameter type, distinct objects
    for repeated types (SweepShelf's five cubes)."""
    skill = next(skill for skill in method.skills() if skill.name == name)
    used: list = []
    objects = []
    for parameter in skill.parameters:
        obj = next(o for o in method.objects() if o.type == parameter.type and o not in used)
        used.append(obj)
        objects.append(obj)
    return GroundSkill(skill=skill, objects=tuple(objects))


def _human(*, method: DpMethod) -> GroundSkill:
    return method.skill_provider.human_cube_bin_reset_skills()[0]


def _observe(*, method: DpMethod, ground: GroundSkill, outcomes: list[bool]) -> None:
    for outcome in outcomes:
        method.observe_outcome(ground_skill=ground, success=outcome)


def _execute(*, method: DpMethod, ground: GroundSkill, cost: float = STEP_PRICE) -> None:
    method.record_action_cost(ground_skill=ground)
    method.observe_execution_cost(cost=cost, steps=1, complete=True)


def _two_skill_setup(*, monkeypatch, budget: float = 100.0, always_ask: bool = False):
    """pick_wiper is weak (5 failures: p = 10/16) and return_wiper strong (20 successes:
    p = 30/31), both on the one seen plan, with Delta at its 0.1 prior. pick_cube has
    been executed but is on no plan."""
    method = _build(budget=budget, always_ask=always_ask)
    pick = _ground(method=method, name="PickWiper")
    back = _ground(method=method, name="ReturnWiper")
    cube = _ground(method=method, name="PickCubeFromShelf")
    _observe(method=method, ground=pick, outcomes=[False] * 5)
    _observe(method=method, ground=back, outcomes=[True] * 20)
    _observe(method=method, ground=cube, outcomes=[True])
    monkeypatch.setattr(DpMethod, "planning_progress_plans", lambda self: [[pick, back]])
    return method, pick, back, cube


def test_human_skills_are_scored_by_default_and_must_be():
    assert DpMethod.model_fields["score_human_skills"].default is True
    env = SweepOffShelf3DEnvironment()
    with pytest.raises(ValidationError, match="score_human_skills"):
        DpMethod(
            env=env,
            skill_provider=SweepOffShelf3DSkillProvider(env=env),
            dp_budget=1.0,
            score_human_skills=False,
        )


def test_ranks_by_remaining_allocation_and_zero_allocations_go_last(*, monkeypatch):
    method, pick, back, cube = _two_skill_setup(monkeypatch=monkeypatch)
    ranked = method.choose_practice_target()
    # pick saturates after ceil((1 - 10/16) / 0.1) = 4 attempts, return after 1.
    assert method.dp_allocation()[pick] == 4
    assert method.dp_allocation()[back] == 1
    assert ranked[:2] == [pick, back]
    zero = set(ranked[2:])
    assert cube in zero
    assert _human(method=method) in zero
    assert all(method.dp_allocation()[ground] == 0 for ground in zero)


def test_remaining_allocation_counts_down_within_a_cycle(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    assert method.choose_practice_target()[0] == pick
    for _ in range(4):
        _execute(method=method, ground=pick)
    assert method.dp_remaining_allocation()[pick] == 0
    assert method.choose_practice_target()[0] == back


def test_reallocates_once_per_practice_period(*, monkeypatch):
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    calls = []
    original = DpAllocator.allocate

    def counting(**kwargs):
        calls.append(kwargs["budget"])
        return original(**kwargs)

    monkeypatch.setattr(DpAllocator, "allocate", staticmethod(counting))
    method.choose_practice_target()
    method.choose_practice_target()
    assert len(calls) == 1
    method._practice_periods += 1  # noqa: SLF001 -- what get_practice_policy does
    method._practice_accounting.robot_cost = 30.0  # noqa: SLF001
    method.choose_practice_target()
    # Re-solved with the budget that remains.
    assert calls == [pytest.approx(100.0), pytest.approx(70.0)]


def test_expected_attempt_cost_is_a_running_mean_from_a_one_step_prior():
    method = _build()
    pick = _ground(method=method, name="PickWiper")
    human = _human(method=method)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(STEP_PRICE)
    assert method.dp_expected_attempt_cost(ground_skill=human) == pytest.approx(STEP_PRICE)
    _execute(method=method, ground=pick, cost=0.5)
    _execute(method=method, ground=pick, cost=1.5)
    _execute(method=method, ground=human, cost=2.0)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(1.0)
    assert method.dp_expected_attempt_cost(ground_skill=human) == pytest.approx(2.0)


def test_an_incomplete_execution_is_not_an_attempt_cost():
    method = _build()
    pick = _ground(method=method, name="PickWiper")
    method.record_action_cost(ground_skill=pick)
    method.observe_execution_cost(cost=9.0, steps=900, complete=False)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(STEP_PRICE)


def test_a_forced_reset_is_charged_to_the_skill_that_last_ran():
    method = _build(always_ask=True)
    sweep = _ground(method=method, name="SweepShelf")
    _execute(method=method, ground=sweep, cost=3.0)
    method.record_forced_reset(reason=PARAMETERIZED_SKILL, cost=5.0, consecutive_stranded=0)
    method.observe_execution_cost(cost=5.0, steps=1, complete=True)
    # One attempt costing 3 + 5, not two attempts averaging 4.
    assert method.dp_expected_attempt_cost(ground_skill=sweep) == pytest.approx(8.0)


def test_under_always_ask_a_triggering_skills_prior_includes_the_forced_reset():
    method = _build(always_ask=True)
    sweep = _ground(method=method, name="SweepShelf")
    pick = _ground(method=method, name="PickWiper")
    assert sweep.skill.param_dim > 0
    assert pick.skill.param_dim == 0
    assert method.dp_expected_attempt_cost(ground_skill=sweep) == pytest.approx(
        STEP_PRICE + START_RESET_PRICE
    )
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(STEP_PRICE)


def test_always_ask_allocates_over_robot_skills_only(*, monkeypatch):
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch, always_ask=True)
    ranked = method.choose_practice_target()
    humans = set(method.skill_provider.human_cube_bin_reset_skills())
    assert not humans & set(ranked)
    assert not humans & set(method.dp_allocation())


def test_by_default_a_human_skill_is_a_candidate_allocated_by_the_same_rule(*, monkeypatch):
    """It is on no evaluation plan (those never offer a human), so the rule gives it 0."""
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    ranked = method.choose_practice_target()
    for human in method.skill_provider.human_cube_bin_reset_skills():
        assert human in ranked
        assert method.dp_allocation()[human] == 0


def test_stops_practising_once_the_budget_is_spent(*, monkeypatch):
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch, budget=1.0)
    assert method.practice_complete() is False
    method._practice_accounting.robot_cost = 1.0  # noqa: SLF001
    assert method.practice_complete() is True
    assert method.choose_practice_target() == [STOP_SKILL]


def test_needs_practice_accounting():
    method = _build(accounting=False)
    with pytest.raises(RuntimeError, match="practice-cost-config"):
        method.choose_practice_target()
