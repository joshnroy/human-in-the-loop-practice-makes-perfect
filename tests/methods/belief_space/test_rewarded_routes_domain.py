"""Deployment value when each single-skill route to the goal has its own reward."""

import numpy as np
import pytest

from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis


def _domain(**kwargs):
    env = TwoRoutesEnvironment(**kwargs)
    return RewardedRoutesPracticeDomain.from_provider(provider=TwoRoutesSkillProvider(env=env))


def _theta(*, a: float, b: float):
    return {
        "RouteA": SkillHypothesis(competence=a, learning_rate=0.0, cost=1.0),
        "RouteB": SkillHypothesis(competence=b, learning_rate=0.0, cost=1.0),
    }


def test_the_domain_is_read_off_the_providers_rewards_and_skills():
    domain = _domain(route_b_reward=5.0)
    assert dict(domain.route_rewards) == {"RouteA": 1.0, "RouteB": 5.0}
    assert dict(domain.belief_sources) == dict.fromkeys(
        ("RouteA", "RouteB", "human_reset_b"), PracticeExampleSource.OUTCOME
    )
    assert domain.deterministic_skills == frozenset({"ResetA"})
    assert domain.human_skill_names == frozenset({"human_reset_b"})
    assert domain.deployment_horizon == 1
    assert (domain.initial_competence_alpha, domain.initial_competence_beta) == (1.0, 3.0)


def test_a_withheld_human_reset_is_not_modelled():
    env = TwoRoutesEnvironment()
    domain = RewardedRoutesPracticeDomain.from_provider(
        provider=TwoRoutesSkillProvider(env=env, offer_human_reset=False)
    )
    assert domain.human_skill_names == frozenset()
    assert "human_reset_b" not in dict(domain.belief_sources)


def test_a_provider_that_prices_no_goal_is_refused():
    from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
    from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider

    with pytest.raises(ValueError, match="prices no goal"):
        RewardedRoutesPracticeDomain.from_provider(
            provider=LightSwitchSkillProvider(env=LightSwitchEnvironment(grid_size=3))
        )


def test_the_value_is_the_better_of_reward_times_competence():
    domain = _domain()
    assert domain.evaluate_theta(theta=_theta(a=0.8, b=0.2)) == pytest.approx(0.8)
    assert domain.evaluate_theta(theta=_theta(a=0.8, b=0.3)) == pytest.approx(0.9)
    assert domain.evaluate_theta(theta=_theta(a=0.1, b=0.9)) == pytest.approx(2.7)


def test_sampled_and_expected_values_agree_and_exceed_the_value_at_the_means():
    domain = _domain()
    beliefs = make_default_tossing3d_belief(domain=domain, num_particles=64).skill_beliefs
    values = domain.sample_policy_values(beliefs=beliefs, rng=np.random.default_rng(0), count=20000)
    assert values.shape == (20000,)
    assert values.min() >= 0.0 and values.max() <= 3.0
    expected = domain.expected_policy_value(beliefs=beliefs)
    assert expected == pytest.approx(values.mean(), abs=0.03)
    assert expected == domain.expected_policy_value(beliefs=beliefs)
    at_means = max(
        1.0 * beliefs["RouteA"].mean_competence(), 3.0 * beliefs["RouteB"].mean_competence()
    )
    # E[max] >= max of E: being unsure which route is better is itself worth something.
    assert expected >= at_means - 1e-9


def test_sampled_thetas_cover_the_routes_only():
    domain = _domain()
    beliefs = make_default_tossing3d_belief(domain=domain, num_particles=16).skill_beliefs
    rng = np.random.default_rng(0)

    def sample(belief, count):  # noqa: PLR0917 -- the SampleSkills signature
        return [
            SkillHypothesis(competence=float(r[0]), learning_rate=float(r[1]), cost=float(r[2]))
            for r in belief.sample(rng=rng, count=count)
        ]

    thetas = domain.sample_thetas(beliefs=beliefs, sample_skills=sample, count=3)
    assert len(thetas) == 3 and all(set(t) == {"RouteA", "RouteB"} for t in thetas)
