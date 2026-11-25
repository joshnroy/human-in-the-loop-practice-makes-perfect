"""Under `--curve-controlled-sampler` every pick is an informed draw that counts, so
`--pomdp-clock-on-conditioned-only`'s forecast must not mark a curve-controlled skill
uncounted. Before this, the forecast read the classifier's state (no success yet ->
uninformative next pick), so ours forecast that practising gather or sweep5 would teach
it nothing and STOPped after 3 and 9 draws in the 2026-11-25 live check."""

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.practice_makes_perfect.curve_controlled_sampler import (
    CurveControlledSampler,
)


def _serving(*, curve: bool) -> Tossing3DPomdpMethod:
    env = RackServingToyEnvironment(config=RackServingConfig(gather_delta_high=0.2))
    provider = RackServingSkillProvider(env=env)
    return Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=2,
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
        pomdp_competence_model="local_trend",
        pomdp_inference_engine="grid",
        pomdp_competence_evidence=CompetenceEvidence.SAMPLER_SUCCEEDED,
        pomdp_clock_on_conditioned_only=True,
        curve_controlled_sampler=(
            CurveControlledSampler(regions=provider.known_outcome_regions()) if curve else None
        ),
    )


def test_a_curve_controlled_skill_is_never_forecast_uncounted() -> None:
    assert _serving(curve=False).uncounted_skills() == frozenset({"sweep3", "gather", "sweep5"})
    assert _serving(curve=True).uncounted_skills() == frozenset()
