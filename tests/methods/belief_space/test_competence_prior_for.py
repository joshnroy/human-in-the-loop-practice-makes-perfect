"""--pomdp-competence-prior-for SKILL=MEAN: one skill starts from its own Beta prior."""

import pytest

from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

DOMAIN = RewardedRoutesPracticeDomain(
    route_rewards=(("RouteA", 3.0), ("RouteB", 6.0)),
    belief_sources=(
        ("RouteA", PracticeExampleSource.OUTCOME),
        ("RouteB", PracticeExampleSource.OUTCOME),
    ),
)


def _means(*, domain: RewardedRoutesPracticeDomain) -> dict[str, float]:
    belief = make_default_tossing3d_belief(
        domain=domain,
        model="local_trend",
        engine="grid",
        num_particles=32,
        inference_config=InferenceConfig(competence_bins=41, learning_rate_bins=4),
    )
    return {name: b.mean_competence() for name, b in belief.skill_beliefs.items()}


def test_a_named_skill_starts_from_the_beta_of_its_mean_and_the_others_keep_the_shared_one():
    domain = Tossing3DPomdpCli.with_prior_means(domain=DOMAIN, means={"RouteA": 0.9})
    assert domain.competence_prior_overrides == (
        ("RouteA", pytest.approx(9.0), pytest.approx(1.0)),
    )
    means = _means(domain=domain)
    assert means["RouteA"] == pytest.approx(0.9, abs=0.03)
    assert means["RouteB"] == pytest.approx(_means(domain=DOMAIN)["RouteB"])


def test_no_means_is_the_domain_itself_and_an_unknown_skill_is_an_error():
    assert Tossing3DPomdpCli.with_prior_means(domain=DOMAIN, means={}) is DOMAIN
    with pytest.raises(ValueError, match="NoSuchSkill"):
        Tossing3DPomdpCli.with_prior_means(domain=DOMAIN, means={"NoSuchSkill": 0.5})
    with pytest.raises(ValueError, match="strictly between"):
        Tossing3DPomdpCli.with_prior_means(domain=DOMAIN, means={"RouteA": 1.0})
