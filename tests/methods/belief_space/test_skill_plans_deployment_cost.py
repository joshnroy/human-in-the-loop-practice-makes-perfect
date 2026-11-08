"""Pouring's fixed skill plans charge expected evaluation cost, priced like practice."""

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.skill_plans_domain import (
    POURING_KITCHEN_SKILL_STEPS,
    SkillPlansPracticeDomain,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis

LAMBDA = 3e-4
PRICE = 0.01
SKILLS = ("PickPitcher", "Pour1", "Pour2", "Pour3", "PlacePitcherAtHome")
POUR_PLANS = tuple(("PickPitcher", f"Pour{k}", "PlacePitcherAtHome") for k in (1, 2, 3))


def _pouring(**update):
    fields = {
        "deployment_horizon": 3,
        "deployment_plans": POUR_PLANS,
        "belief_sources": tuple((name, PracticeExampleSource.OUTCOME) for name in SKILLS),
        "initial_competence_alpha": 4.0,
        "initial_competence_beta": 1.0,
        "skill_steps": POURING_KITCHEN_SKILL_STEPS,
    }
    return SkillPlansPracticeDomain(**(fields | update))


def _theta(**competences):
    return {
        name: SkillHypothesis(competence=competences.get(name, 0.8), learning_rate=0)
        for name in SKILLS
    }


def test_the_pouring_steps_are_the_measured_ones():
    assert POURING_KITCHEN_SKILL_STEPS["PickPitcher"] == (512.0, 276.0)
    assert POURING_KITCHEN_SKILL_STEPS["Pour2"] == (1061.0, 276.0)
    assert POURING_KITCHEN_SKILL_STEPS["PlacePitcherAtHome"] == (324.0, 79.0)
    assert set(POURING_KITCHEN_SKILL_STEPS) == {
        "DropShrimpIntoPitcher",
        "PickPitcher",
        "PickShrimpFromFloor",
        "PickShrimpFromSurface",
        "PickShrimpFromTray",
        "PlacePitcherAtHome",
        "Pour1",
        "Pour2",
        "Pour3",
    }


def test_unpriced_pouring_values_are_byte_identical_to_the_solve_probability():
    unpriced = _pouring()
    plain = _pouring(skill_steps={})
    theta = _theta(Pour2=0.3)
    assert unpriced.evaluate_theta(theta=theta) == plain.evaluate_theta(theta=theta)
    beliefs = make_default_tossing3d_belief(domain=unpriced, num_particles=16).skill_beliefs
    assert unpriced.expected_policy_value(beliefs=beliefs) == plain.expected_policy_value(
        beliefs=beliefs
    )
    np.testing.assert_array_equal(
        unpriced.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(3), count=64),
        plain.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(3), count=64),
    )
    assert plain.expected_deployment_cost(beliefs=beliefs) == 0.0
    priced_plain = plain.model_copy(update={"deployment_cost_lambda": LAMBDA})
    assert priced_plain.model_copy(update={"robot_step_price": PRICE}).expected_policy_value(
        beliefs=beliefs
    ) == plain.expected_policy_value(beliefs=beliefs)


def test_a_task_ends_at_its_first_failed_step():
    priced = _pouring(deployment_cost_lambda=LAMBDA, robot_step_price=PRICE)
    steps = POURING_KITCHEN_SKILL_STEPS
    pick, place = steps["PickPitcher"], steps["PlacePitcherAtHome"]
    theta = _theta(PickPitcher=0.5, Pour1=0.25, Pour2=0.25, Pour3=0.25, PlacePitcherAtHome=0.5)
    per_plan = []
    for k in (1, 2, 3):
        pour = steps[f"Pour{k}"]
        per_plan.append(
            0.5 * pick[0]
            + 0.5 * pick[1]
            + 0.5 * (0.25 * pour[0] + 0.75 * pour[1])
            + 0.5 * 0.25 * (0.5 * place[0] + 0.5 * place[1])
        )
    solve = 0.5 * 0.25 * 0.5
    assert priced.evaluate_theta(theta=theta) == pytest.approx(
        solve - LAMBDA * PRICE * float(np.mean(per_plan))
    )
    # A pitcher that is never picked up costs one failed pick per task, nothing else.
    assert priced.evaluate_theta(theta=_theta(PickPitcher=0.0)) == pytest.approx(
        -LAMBDA * PRICE * pick[1]
    )


def test_g_subtracts_lambda_times_expected_pouring_cost_on_both_paths():
    domain = _pouring()
    belief = make_default_tossing3d_belief(domain=domain, num_particles=16)
    plain = Tossing3DPracticeModel(domain=domain, linear_cost_lambda=LAMBDA, stop_cost=PRICE)
    priced = plain.model_copy(update={"robot_step_price": PRICE})
    weighted = priced.valued_domain()
    assert (weighted.deployment_cost_lambda, weighted.robot_step_price) == (LAMBDA, PRICE)
    beliefs = belief.skill_beliefs
    cost = weighted.expected_deployment_cost(beliefs=beliefs)
    # About two thousand control steps per task, priced at 0.01 each.
    assert 10.0 < cost < 30.0
    # Unpriced, the plain domain reports no cost.
    assert domain.expected_deployment_cost(beliefs=beliefs) == 0.0
    solve = domain.expected_policy_value(beliefs=beliefs)
    net = weighted.expected_policy_value(beliefs=beliefs)
    assert net == pytest.approx(solve - LAMBDA * cost)
    j = priced.J(belief_state=belief, summed_cost=1.0, num_samples=1)
    assert j == pytest.approx(solve - LAMBDA * cost - LAMBDA * (1.0 + PRICE))
    assert j < plain.J(belief_state=belief, summed_cost=1.0, num_samples=1)
    samples = priced.sample_policy_values_from_belief(belief_state=belief, num_samples=4096)
    assert float(np.mean(samples)) == pytest.approx(net, abs=0.02)


def test_pouring_steps_must_cover_every_planned_skill():
    with pytest.raises(ValueError, match="skill_steps"):
        _pouring(skill_steps={"PickPitcher": (1.0, 1.0)})
