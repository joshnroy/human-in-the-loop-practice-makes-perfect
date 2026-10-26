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
from hitl_pmp.methods.practice_makes_perfect.dp_gurobi import DpGurobiAllocator, DpOptimizationError
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpMdp, DpTaskModel, DpTransition
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import STOP_SKILL, _EesEpisode

STEP_PRICE = 0.01
START_RESET_PRICE = 5.0


def _build(*, budget: float = 100.0, always_ask: bool = False, accounting: bool = True):
    env = SweepOffShelf3DEnvironment()
    method = DpMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env, human_reset_practice_cost=2.0),
        dp_budget=budget,
        dp_solver="enumeration",
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
    method.begin_practice_attempt(ground_skill=ground)
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
    monkeypatch.setattr(
        DpTaskModel,
        "build",
        staticmethod(
            lambda **kwargs: DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={2: 1},
                transitions=[
                    DpTransition(state=0, skill=pick, successor=1),
                    DpTransition(state=1, skill=back, successor=2),
                ],
            )
        ),
    )
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


def test_curriculum_follows_dependencies_and_excludes_zero_allocations(*, monkeypatch):
    method, pick, back, cube = _two_skill_setup(monkeypatch=monkeypatch)
    ranked = method.choose_practice_target()
    # pick saturates after ceil((1 - 10/16) / 0.1) = 4 attempts, return after 1.
    assert method.dp_allocation()[pick] == 4
    assert method.dp_allocation()[back] == 1
    assert ranked == [pick, STOP_SKILL]
    assert method.dp_allocation()[cube] == 0
    assert method.dp_allocation()[_human(method=method)] == 0


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


def test_incomplete_execution_spending_is_retained_until_target_completes():
    method = _build()
    pick = _ground(method=method, name="PickWiper")
    method.begin_practice_attempt(ground_skill=pick)
    method.record_action_cost(ground_skill=pick)
    method.observe_execution_cost(cost=9.0, steps=900, complete=False)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(9.0 + STEP_PRICE)
    assert method.dp_remaining_allocation() == {}
    _execute(method=method, ground=pick, cost=1.0)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(10.0)


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
        assert human not in ranked
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


def test_setup_costs_and_incidental_skills_belong_to_target_not_allocation(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    method.choose_practice_target()
    human = _human(method=method)
    method.begin_practice_attempt(ground_skill=back)
    for ground, cost in [(human, 5), (pick, 2), (back, 3)]:
        method.record_action_cost(ground_skill=ground)
        method.observe_execution_cost(cost=cost, steps=1, complete=True)
    assert method.dp_expected_attempt_cost(ground_skill=back) == pytest.approx(10)
    assert method.dp_remaining_allocation()[pick] == 4
    assert method.dp_remaining_allocation()[back] == 0
    assert method.dp_expected_attempt_cost(ground_skill=pick) == STEP_PRICE


def test_failed_setup_costs_survive_reselection_and_are_amortized():
    method = _build()
    pick = _ground(method=method, name="PickWiper")
    back = _ground(method=method, name="ReturnWiper")
    method.begin_practice_attempt(ground_skill=back)
    method.record_action_cost(ground_skill=pick)
    method.observe_execution_cost(cost=2, steps=1, complete=True)
    _execute(method=method, ground=pick, cost=1)
    _execute(method=method, ground=back, cost=3)
    _execute(method=method, ground=back, cost=1)
    assert method.dp_expected_attempt_cost(ground_skill=back) == pytest.approx(3)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(1)


def test_forced_reset_during_setup_is_charged_once_to_the_target():
    method = _build(always_ask=True)
    sweep = _ground(method=method, name="SweepShelf")
    back = _ground(method=method, name="ReturnWiper")
    method.begin_practice_attempt(ground_skill=back)
    method.record_action_cost(ground_skill=sweep)
    method.observe_execution_cost(cost=3, steps=1, complete=True)
    method.record_forced_reset(reason=PARAMETERIZED_SKILL, cost=5, consecutive_stranded=0)
    method.observe_execution_cost(cost=5, steps=1, complete=True)
    method.record_action_cost(ground_skill=back)
    method.observe_execution_cost(cost=1, steps=1, complete=True)
    assert method.dp_expected_attempt_cost(ground_skill=back) == pytest.approx(9)
    assert method.dp_expected_attempt_cost(ground_skill=sweep) == pytest.approx(5.01)


def test_consumed_allocation_stops_period_without_zero_target_fallback(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    method.choose_practice_target()
    for _ in range(4):
        _execute(method=method, ground=pick)
    _execute(method=method, ground=back)
    assert method.choose_practice_target() == [STOP_SKILL]
    assert not method.practice_complete()


def test_dependency_order_wins_over_larger_downstream_allocation(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    _observe(method=method, ground=back, outcomes=[False] * 100)
    assert method.choose_practice_target() == [pick, STOP_SKILL]
    assert method.dp_allocation()[back] > method.dp_allocation()[pick]


def test_no_profitable_allocation_finishes_without_random_practice(*, monkeypatch):
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch, budget=0.001)
    assert method.choose_practice_target() == [STOP_SKILL]
    assert method.practice_complete()


def test_dp_skips_ees_goal_pursuit_even_when_enabled():
    method = _build()
    assert not method.pursues_goal_in_period(period=0)


def test_completed_predecessor_is_learned_before_downstream_practice(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    calls = []
    monkeypatch.setattr(DpMethod, "fit_samplers", lambda self: calls.append("fit"))
    assert method.choose_practice_target() == [pick, STOP_SKILL]
    for _ in range(4):
        _execute(method=method, ground=pick)
    assert calls == []
    assert method.choose_practice_target() == [back, STOP_SKILL]
    assert calls == ["fit"]
    method.choose_practice_target()
    assert calls == ["fit"]


def test_episode_notifies_target_before_executing_its_setup_prefix(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    monkeypatch.setattr(DpMethod, "plan_to", lambda self, **kwargs: [back])
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True, pursue_goal=False)
    assert episode._practice_plan(true_atoms=frozenset()) == [back, pick]
    for ground, cost in [(back, 2), (pick, 1)]:
        method.record_action_cost(ground_skill=ground)
        method.observe_execution_cost(cost=cost, steps=1, complete=True)
    assert method.dp_expected_attempt_cost(ground_skill=pick) == pytest.approx(3)
    assert method.dp_remaining_allocation()[back] == 1


def test_once_allocation_survives_periods_and_finishes_when_consumed(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    method.dp_budget_unit = "episodes"
    method.dp_allocation_schedule = "once"
    method.dp_budget = 20
    calls = []
    original = DpAllocator.allocate

    def counting(**kwargs):
        calls.append(kwargs["budget"])
        return original(**kwargs)

    monkeypatch.setattr(DpAllocator, "allocate", staticmethod(counting))
    method.choose_practice_target()
    _execute(method=method, ground=pick)
    method._practice_periods += 1
    method.choose_practice_target()
    assert calls == [20]
    assert method.dp_remaining_allocation()[pick] == 3
    for _ in range(3):
        _execute(method=method, ground=pick)
    _execute(method=method, ground=back)
    assert method.dp_episodes() == 5
    assert method.practice_complete()
    assert method.choose_practice_target() == [STOP_SKILL]


def test_episode_budget_counts_targets_not_prefix_or_human_cost(*, monkeypatch):
    method, pick, back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    method.dp_budget_unit = "episodes"
    method.dp_budget = 1
    method.choose_practice_target()
    method.begin_practice_attempt(ground_skill=pick)
    for ground in [_human(method=method), back]:
        method.record_action_cost(ground_skill=ground)
        method.observe_execution_cost(cost=50, steps=1, complete=True)
    method._practice_accounting.human_cost = 100
    assert method.dp_episodes() == 0
    assert not method.practice_complete()
    method.record_action_cost(ground_skill=pick)
    method.observe_execution_cost(cost=1, steps=1, complete=True)
    assert method.dp_episodes() == 1
    assert method.practice_complete()


def test_fractional_episode_budget_is_rejected():
    method = _build()
    with pytest.raises(ValidationError, match="integer"):
        DpMethod(
            env=method.env,
            skill_provider=method.skill_provider,
            dp_budget_unit="episodes",
            dp_budget=1.5,
        )


def test_gurobi_failure_is_logged_and_never_falls_back(*, monkeypatch, tmp_path):
    method, _pick, _back, _cube = _two_skill_setup(monkeypatch=monkeypatch)
    method.dp_solver = "gurobi"
    method.dp_solver_log = tmp_path / "allocations.jsonl"

    def fail(**kwargs):
        raise DpOptimizationError(diagnostics={"status": "TIME_LIMIT"})

    monkeypatch.setattr(DpGurobiAllocator, "allocate", staticmethod(fail))
    monkeypatch.setattr(DpAllocator, "allocate", lambda **kwargs: pytest.fail("silent fallback"))
    with pytest.raises(DpOptimizationError):
        method.choose_practice_target()
    assert "TIME_LIMIT" in method.dp_solver_log.read_text()


def test_once_with_optimistic_priors_can_leave_most_of_budget_unused(*, monkeypatch):
    method = _build(budget=2000)
    method.dp_budget_unit = "episodes"
    method.dp_allocation_schedule = "once"
    pick = _ground(method=method, name="PickWiper")
    back = _ground(method=method, name="ReturnWiper")
    monkeypatch.setattr(
        DpTaskModel,
        "build",
        staticmethod(
            lambda **kwargs: DpMdp(
                initial_probabilities={0: 1},
                terminal_rewards={2: 1},
                transitions=[
                    DpTransition(state=0, skill=pick, successor=1),
                    DpTransition(state=1, skill=back, successor=2),
                ],
            )
        ),
    )
    method.choose_practice_target()
    assert sum(method.dp_allocation().values()) == 2
    _execute(method=method, ground=pick)
    _execute(method=method, ground=back)
    assert method.practice_complete()
    assert method.dp_episodes() == 2
