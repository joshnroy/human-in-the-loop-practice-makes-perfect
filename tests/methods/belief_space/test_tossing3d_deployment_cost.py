"""Tossing3D's expected evaluation cost: the same pick/open/toss policy P(solve) assumes."""

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.practice_domain import (
    TOSSING3D_DEPLOYMENT_STEPS,
    TOSSING3D_DOMAIN,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_deployment_model import (
    DeploymentPolicyExpectation,
    evaluate_deployment_policy,
    evaluate_deployment_policy_costs,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta

LAMBDA = 3e-4
PRICE = 0.01


def _belief():
    return make_default_tossing3d_belief(
        domain=TOSSING3D_DOMAIN,
        model="local_trend",
        engine="grid",
        num_particles=32,
        inference_config=InferenceConfig(competence_bins=9, learning_rate_bins=4),
    )


def _observed(*, belief):
    beliefs = dict(belief.skill_beliefs)
    for name in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL):
        for success in (True, False, True):
            beliefs[name] = beliefs[name].condition_outcome(success=success)
    return belief.model_copy(update={"skill_beliefs": beliefs})


def _priced():
    return TOSSING3D_DOMAIN.model_copy(
        update={"deployment_cost_lambda": LAMBDA, "robot_step_price": PRICE}
    )


def _step_costs():
    steps = dict(TOSSING3D_DEPLOYMENT_STEPS)
    return {name: steps[name] * PRICE for name in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)}


def test_the_cost_recursion_walks_pick_open_toss_to_the_horizon():
    # Horizon 4 by hand: ready -> pick; success -> toss (ends); failure -> open, then
    # ready again with two steps left (pick, then toss on success / open on failure).
    p, q, cp, ct, co = 0.6, 0.7, 2.0, 1.0, 0.4
    closed_1 = co
    ready_2 = cp + p * ct + (1 - p) * closed_1
    closed_2 = co + q * cp + (1 - q) * closed_1
    closed_3 = co + q * ready_2 + (1 - q) * closed_2
    ready_4 = cp + p * ct + (1 - p) * closed_3
    cost = evaluate_deployment_policy_costs(
        pick_competences=np.array([p]),
        open_competences=np.array([q]),
        pick_costs=np.array([cp]),
        toss_costs=np.array([ct]),
        open_costs=np.array([co]),
        horizon=4,
    )
    assert cost[0] == pytest.approx(ready_4)


def test_the_exact_expectation_matches_the_policy_value_it_prices():
    beliefs = _observed(belief=_belief()).skill_beliefs
    costs = _step_costs()
    value, cost = DeploymentPolicyExpectation.evaluate_with_cost(
        pick=beliefs[PICK_SKILL],
        toss=beliefs[TOSS_SKILL],
        opened=beliefs[OPEN_GRIPPER_SKILL],
        horizon=4,
        pick_cost=costs[PICK_SKILL],
        toss_cost=costs[TOSS_SKILL],
        open_cost=costs[OPEN_GRIPPER_SKILL],
    )
    assert value == DeploymentPolicyExpectation.evaluate(
        pick=beliefs[PICK_SKILL],
        toss=beliefs[TOSS_SKILL],
        opened=beliefs[OPEN_GRIPPER_SKILL],
        horizon=4,
    )
    # Monte Carlo over the same beliefs agrees with the exact expected cost.
    rng = np.random.default_rng(0)
    count = 200_000
    draws = {name: beliefs[name].sample(rng=rng, count=count)[:, 0] for name in costs}
    sampled = evaluate_deployment_policy_costs(
        pick_competences=draws[PICK_SKILL],
        open_competences=draws[OPEN_GRIPPER_SKILL],
        pick_costs=np.full(count, costs[PICK_SKILL]),
        toss_costs=np.full(count, costs[TOSS_SKILL]),
        open_costs=np.full(count, costs[OPEN_GRIPPER_SKILL]),
        horizon=4,
    )
    assert cost == pytest.approx(float(sampled.mean()), rel=1e-2)
    assert _priced().expected_deployment_cost(beliefs=beliefs) == pytest.approx(cost)
    # Unpriced (the default), the domain reports none, like every domain that opts out.
    assert TOSSING3D_DOMAIN.expected_deployment_cost(beliefs=beliefs) == 0.0


def test_priced_values_subtract_lambda_times_expected_cost_on_every_path():
    beliefs = _observed(belief=_belief()).skill_beliefs
    priced = _priced()
    costs = _step_costs()
    cost = priced.expected_deployment_cost(beliefs=beliefs)
    # One task costs at least a pick and at most four of the dearest skill.
    assert costs[PICK_SKILL] <= cost <= 4 * costs[PICK_SKILL]
    assert priced.expected_policy_value(beliefs=beliefs) == pytest.approx(
        TOSSING3D_DOMAIN.expected_policy_value(beliefs=beliefs) - LAMBDA * cost
    )
    theta = Tossing3DTheta(
        pick=SkillHypothesis(competence=0.6, learning_rate=0.0),
        toss=SkillHypothesis(competence=0.5, learning_rate=0.0),
        open_gripper=SkillHypothesis(competence=0.7, learning_rate=0.0),
    )
    theta_cost = evaluate_deployment_policy_costs(
        pick_competences=np.array([0.6]),
        open_competences=np.array([0.7]),
        pick_costs=np.array([costs[PICK_SKILL]]),
        toss_costs=np.array([costs[TOSS_SKILL]]),
        open_costs=np.array([costs[OPEN_GRIPPER_SKILL]]),
        horizon=4,
    )[0]
    assert priced.evaluate_theta(theta=theta) == pytest.approx(
        evaluate_deployment_policy(
            toss_competence=0.5, pick_competence=0.6, open_competence=0.7, horizon=4
        )
        - LAMBDA * theta_cost
    )
    plain = TOSSING3D_DOMAIN.sample_policy_values(
        beliefs=beliefs, rng=np.random.default_rng(3), count=64
    )
    net = priced.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(3), count=64)
    assert np.all(net < plain)
    assert np.all(plain - net <= LAMBDA * 4 * costs[PICK_SKILL])


def test_a_priced_model_values_deployment_below_the_unpriced_one():
    belief = _observed(belief=_belief())
    model = Tossing3DPracticeModel(
        domain=TOSSING3D_DOMAIN, linear_cost_lambda=LAMBDA, stop_cost=PRICE
    )
    priced = model.model_copy(update={"robot_step_price": PRICE})
    assert priced.J(belief_state=belief, summed_cost=1.0, num_samples=1) < model.J(
        belief_state=belief, summed_cost=1.0, num_samples=1
    )


def test_the_real_robot_prices_deployment_at_its_measured_wall_clock_charges():
    """The real TidyBot charges wall-clock seconds (0.1 per second), not simulator steps:
    its deployment pick and toss cost what they were measured to cost on the robot, in
    steps of the run's 0.01 robot-step price. Nothing else differs from the sim domain."""
    from hitl_pmp.methods.belief_space.practice_domain import TOSSING3D_REAL_DOMAIN
    from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

    assert Tossing3DPomdpCli.DOMAINS["Tossing3DRealCli"] is TOSSING3D_REAL_DOMAIN
    assert Tossing3DPomdpCli.DOMAINS["Tossing3DCli"] is TOSSING3D_DOMAIN
    priced = TOSSING3D_REAL_DOMAIN.model_copy(
        update={"deployment_cost_lambda": 1.2e-4, "robot_step_price": PRICE}
    )
    cost = {name: priced._deployment_cost(skill_name=name) for name in (PICK_SKILL, TOSS_SKILL)}  # noqa: SLF001
    assert cost[PICK_SKILL] == pytest.approx(3.642, abs=1e-3)
    assert cost[TOSS_SKILL] == pytest.approx(2.829, abs=1e-3)
    assert (
        TOSSING3D_REAL_DOMAIN.model_copy(update={"deployment_steps": TOSSING3D_DEPLOYMENT_STEPS})
        == TOSSING3D_DOMAIN
    )
