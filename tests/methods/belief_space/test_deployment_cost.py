"""Expected evaluation (deployment) cost in the stopping value, priced like practice."""

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN, TOSSING3D_DOMAIN
from hitl_pmp.methods.belief_space.rack_serving_domain import (
    RACK_ROBOT_SKILLS,
    RACK_SERVING_DOMAIN,
    RACK_SKILL_STEPS,
    RackServingDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
    refit_belief_state,
)

LAMBDA = 3e-4
PRICE = 0.01


def _belief(*, domain):
    return make_default_tossing3d_belief(
        domain=domain,
        model="local_trend",
        engine="grid",
        num_particles=32,
        inference_config=InferenceConfig(competence_bins=9, learning_rate_bins=4),
    )


def _competences(**overrides):
    return {name: np.array([overrides.get(name, 0.9)]) for name in RACK_ROBOT_SKILLS}


@pytest.mark.parametrize("domain", [TOSSING3D_DOMAIN, SWEEP_OFF_SHELF_DOMAIN, RACK_SERVING_DOMAIN])
def test_the_default_is_zero_deployment_cost_and_changes_no_value(*, domain):
    assert domain.deployment_cost_lambda == 0.0
    assert domain.robot_step_price == 0.0
    belief = _belief(domain=domain)
    plain = Tossing3DPracticeModel(domain=domain, linear_cost_lambda=LAMBDA, stop_cost=PRICE)
    # No step price: the cost-weighted domain is never built.
    unpriced = Tossing3DPracticeModel(
        domain=domain, linear_cost_lambda=LAMBDA, stop_cost=PRICE, robot_step_price=0.0
    )
    for model in (plain, unpriced):
        assert model.J(belief_state=belief, summed_cost=1.0, num_samples=1) == plain.J(
            belief_state=belief, summed_cost=1.0, num_samples=1
        )
    assert domain.expected_deployment_cost(beliefs=belief.skill_beliefs) == 0.0


@pytest.mark.parametrize("domain", [TOSSING3D_DOMAIN, SWEEP_OFF_SHELF_DOMAIN])
def test_a_domain_that_does_not_model_deployment_cost_ignores_the_price(*, domain):
    belief = _belief(domain=domain)
    plain = Tossing3DPracticeModel(domain=domain, linear_cost_lambda=LAMBDA, stop_cost=PRICE)
    priced = plain.model_copy(update={"robot_step_price": PRICE})
    assert priced.J(belief_state=belief, summed_cost=2.0, num_samples=1) == plain.J(
        belief_state=belief, summed_cost=2.0, num_samples=1
    )


def test_the_hard_budget_objective_never_charges_deployment():
    belief = _belief(domain=RACK_SERVING_DOMAIN)
    plain = Tossing3DPracticeModel(domain=RACK_SERVING_DOMAIN)
    priced = plain.model_copy(update={"robot_step_price": PRICE})
    assert priced.J(belief_state=belief, summed_cost=1.0, num_samples=1) == plain.J(
        belief_state=belief, summed_cost=1.0, num_samples=1
    )


def test_zero_weight_rack_values_are_exactly_the_old_solve_probabilities():
    competences = _competences(GatherAndSweep=0.3, PickFood=0.6)
    values, steps = RackServingDeployment.solve(
        competences=competences, horizon=12, step_weight=0.0
    )
    np.testing.assert_array_equal(
        values, RackServingDeployment.values(competences=competences, horizon=12)
    )
    assert steps.shape == values.shape


def test_g_subtracts_lambda_times_expected_deployment_cost_in_practice_units():
    belief = _belief(domain=RACK_SERVING_DOMAIN)
    plain = Tossing3DPracticeModel(
        domain=RACK_SERVING_DOMAIN, linear_cost_lambda=LAMBDA, stop_cost=PRICE
    )
    priced = plain.model_copy(update={"robot_step_price": PRICE})
    weighted = RACK_SERVING_DOMAIN.model_copy(
        update={"deployment_cost_lambda": LAMBDA, "robot_step_price": PRICE}
    )
    beliefs = refit_belief_state(state=belief).skill_beliefs
    cost = weighted.expected_deployment_cost(beliefs=beliefs)
    # Thousands of control steps per task, priced at 0.01 each.
    assert 10.0 < cost < 200.0
    net = weighted.expected_policy_value(beliefs=beliefs)
    unpriced = RACK_SERVING_DOMAIN.expected_policy_value(beliefs=beliefs)
    # Pricing deployment lowers the value by about lambda * cost: the policy may also
    # trade a little solve probability for fewer steps.
    assert 0.0 < unpriced - net < 2 * LAMBDA * cost
    solve = net + LAMBDA * cost
    assert 0.0 < solve <= 1.0
    j = priced.J(belief_state=belief, summed_cost=1.0, num_samples=1)
    assert j == pytest.approx(solve - LAMBDA * cost - LAMBDA * (1.0 + PRICE))
    assert j == priced.G(policy_value=net, summed_cost=1.0)
    assert j < plain.J(belief_state=belief, summed_cost=1.0, num_samples=1)
    # The sampled path carries the same deployment term as model J.
    samples = priced.sample_policy_values_from_belief(belief_state=belief, num_samples=4096)
    assert float(np.mean(samples)) == pytest.approx(net, abs=0.02)


def test_rack_expected_cost_falls_as_sweep_competence_rises():
    weight = LAMBDA * PRICE
    costs = []
    for sweep in (0.1, 0.5, 0.9, 1.0):
        _, steps = RackServingDeployment.solve(
            competences=_competences(
                PickScraper=1.0,
                GatherAndSweep=sweep,
                Sweep=0.0,
                PickFood=1.0,
                PlaceFoodOnTray=1.0,
            ),
            horizon=12,
            step_weight=weight,
        )
        costs.append(float(steps[0]))
    assert costs == sorted(costs, reverse=True)
    assert costs[0] > costs[-1]
    # A sure sweep empties the rack in one pass: pick, sweep, nothing else.
    assert costs[-1] == pytest.approx(
        RACK_SKILL_STEPS["PickScraper"][0] + RACK_SKILL_STEPS["GatherAndSweep"][0]
    )


def test_rack_failure_costs_are_charged_on_a_failed_attempt():
    _, steps = RackServingDeployment.solve(
        competences=_competences(PickScraper=1.0, GatherAndSweep=0.5, Sweep=0.0, PickFood=0.0),
        horizon=12,
        step_weight=LAMBDA * PRICE,
    )
    success, failure = RACK_SKILL_STEPS["GatherAndSweep"]
    attempt = 0.5 * success + 0.5 * failure
    expected = RACK_SKILL_STEPS["PickScraper"][0] + sum(0.5**k * attempt for k in range(11))
    assert float(steps[0]) == pytest.approx(expected)
