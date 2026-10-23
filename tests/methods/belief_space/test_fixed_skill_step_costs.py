"""Fixed per-skill planner prices for the SweepOffShelf3D belief-space arms.

Josh, 2026-10-21: "fixed cost per skill". The planner prices a robot skill at a fixed
nominal step count measured once from the skill benchmark, never at a learned running
average with a 1-step prior for a never-run grounding. The practice budget is still
charged the steps a skill actually took.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.practice_costs import (
    ChargeFunction,
    HumanCharge,
    PracticeAccounting,
    PracticeCosts,
    SkillStepCosts,
)
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills
from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import estimated_action_cost
from hitl_pmp.step_protocol import PracticeClock

REPO = Path(__file__).resolve().parents[3]
TABLE = REPO / "configs" / "sweepoffshelf3d_skill_step_costs.json"
ROBOT_SKILLS = {skill.name for skill in SweepOffShelf3DSkills.robot_skills()}


def _steps() -> dict[str, float]:
    return {name: 100.0 * (index + 1) for index, name in enumerate(sorted(ROBOT_SKILLS))}


def _method(*, skill_step_costs: dict[str, float] | None) -> Tossing3DPomdpMethod:
    env = SweepOffShelf3DEnvironment()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env),
        pomdp_solver="determinized_astar",
        pomdp_num_particles=16,
        pomdp_domain=SWEEP_OFF_SHELF_DOMAIN,
        pomdp_skill_step_costs=skill_step_costs,
    )
    method.configure_practice_accounting(
        accounting=PracticeAccounting(
            costs=PracticeCosts(
                human_skills={
                    "reset_cubes": HumanCharge(cost=ChargeFunction(value=500)),
                    "reset_wiper": HumanCharge(cost=ChargeFunction(value=500)),
                }
            )
        )
    )
    return method


def _robot_groundings(*, method: Tossing3DPomdpMethod) -> list[GroundSkill]:
    names = {reset.skill.name for reset in method.skill_provider.human_cube_bin_reset_skills()}
    return [
        g
        for g in method._pomdp_model.ground_skills  # noqa: SLF001
        if g.skill.name not in names
    ]


def test_every_robot_grounding_is_priced_at_its_skills_fixed_steps_before_any_run() -> None:
    method = _method(skill_step_costs=_steps())
    groundings = _robot_groundings(method=method)
    assert {g.skill.name for g in groundings} == ROBOT_SKILLS
    for ground in groundings:
        price = estimated_action_cost(state=method.pomdp_state, action=ground)
        assert price == _steps()[ground.skill.name]


def test_an_execution_does_not_move_the_fixed_price_but_is_still_charged_in_full() -> None:
    method = _method(skill_step_costs=_steps())
    ground = next(
        g for g in _robot_groundings(method=method) if g.skill.name == "ReturnCubeToShelfFromBox"
    )
    fixed = _steps()["ReturnCubeToShelfFromBox"]
    method.record_action_cost(ground_skill=ground)
    method.observe_execution_cost(cost=1198, steps=1198, complete=True)
    assert estimated_action_cost(state=method.pomdp_state, action=ground) == fixed
    # What the run is charged is what the skill actually took.
    assert method.pomdp_state.accumulated_cost == 1198


@pytest.mark.parametrize("fixed", [True, False])
def test_every_grounding_of_a_human_reset_shares_one_estimate_from_observed_charges(
    *, fixed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model grounds `reset_cubes` over every cube tuple, repeats included, while
    the provider names one canonical grounding. Each is the same human reset, charged
    by name, so each carries one estimate: the shared one-step prior until a charge is
    observed, then the observed mean -- never the cost file's price, which only the
    environment knows. A fixed robot table does not price human skills."""
    method = _method(skill_step_costs=_steps() if fixed else None)
    monkeypatch.setattr(Tossing3DPomdpMethod, "abstract_state", lambda _self, *, state: frozenset())
    monkeypatch.setattr(SweepOffShelf3DEnvironment, "get_current_state", lambda _self: None)
    canonical = next(
        reset
        for reset in method.skill_provider.human_cube_bin_reset_skills()
        if reset.skill.name == "reset_cubes"
    )
    resets = [
        g
        for g in method._pomdp_model.ground_skills  # noqa: SLF001
        if g.skill.name == "reset_cubes"
    ]
    assert len(resets) > 1
    for reset in resets:
        assert estimated_action_cost(state=method.pomdp_state, action=reset) == 1
    method.record_action_cost(ground_skill=resets[-1])
    method.observe_execution_cost(cost=500, steps=1, complete=True)
    for reset in [*resets, canonical]:
        assert estimated_action_cost(state=method.pomdp_state, action=reset) == 500


def test_without_a_table_the_learned_forecast_and_its_prior_are_unchanged() -> None:
    method = _method(skill_step_costs=None)
    ground = _robot_groundings(method=method)[0]
    assert estimated_action_cost(state=method.pomdp_state, action=ground) == 1
    method.record_action_cost(ground_skill=ground)
    method.observe_execution_cost(cost=240, steps=240, complete=True)
    assert estimated_action_cost(state=method.pomdp_state, action=ground) == 240


def test_a_table_that_misses_a_robot_skill_is_refused() -> None:
    partial = _steps()
    del partial["SweepShelf"]
    with pytest.raises(ValueError, match="SweepShelf"):
        _method(skill_step_costs=partial)


@pytest.mark.parametrize("cost", [500.0, 5.0])
def test_the_budget_is_charged_one_counted_step_per_reset_whatever_its_cost(*, cost: float) -> None:
    accounting = PracticeAccounting(
        costs=PracticeCosts(
            human_skills={"reset_cubes": HumanCharge(cost=ChargeFunction(value=cost))}
        )
    )
    clock = PracticeClock(
        budget=10_000, interval=10_000, accounting=accounting, on_measure=lambda: None
    )
    clock.robot_active = True
    for _ in range(7):
        clock.before_robot_step()
        clock.after_robot_step()
    charge = clock.before_human(skill="reset_cubes")
    clock.human_invoked(charge=charge)
    assert clock.steps == 7 + 1
    assert accounting.practice_steps == 7 + 1
    assert accounting.human_cost == cost
    assert accounting.robot_cost == 7


def test_the_committed_table_covers_every_robot_skill_and_records_its_source() -> None:
    table = SkillStepCosts.load(path=TABLE)
    assert set(table.skill_steps) == ROBOT_SKILLS
    assert all(steps > 1 for steps in table.skill_steps.values())
    assert len(table.source.benchmark_commit) == 40
    assert len(table.source.report_sha256) == 64
    assert set(table.source.cases) == ROBOT_SKILLS


def test_a_table_with_a_nonpositive_price_is_refused() -> None:
    with pytest.raises(ValidationError):
        SkillStepCosts.model_validate({
            "source": SkillStepCosts.load(path=TABLE).source.model_dump(),
            "skill_steps": {"PickWiper": 0.0},
        })
