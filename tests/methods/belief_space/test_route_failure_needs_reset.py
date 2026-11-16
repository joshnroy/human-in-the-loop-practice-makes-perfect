"""In Two Routes a try of either route, success or failure, leaves the robot in After,
waiting for a reset. The belief-space model must forecast the same: a failed try that
stayed in Ready would be a free extra try, and inflated the value of an unlikely route."""

from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.skill_provider import TwoRoutesSkillProvider
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    FrozenSkillEffects,
    binary_outcomes,
)


def _names(*, atoms):
    return {atom.predicate.name for atom in atoms}


def test_a_failed_route_try_ends_in_after_needing_a_reset():
    provider = TwoRoutesSkillProvider(env=TwoRoutesEnvironment(), both_resets=True)
    domain = RewardedRoutesPracticeDomain.from_provider(provider=provider)
    for route in ("RouteA", "RouteB"):
        skill = provider.ground(name=route)
        effects = FrozenSkillEffects(
            table={skill: (skill.add_effects, skill.delete_effects, frozenset())}
        )
        branches = binary_outcomes(
            state=make_default_tossing3d_belief(domain=domain, num_particles=8),
            true_atoms=skill.preconditions,
            ground_skill=skill,
            probability=0.3,
            cost=0.0,
            effects=effects,
            domain=domain,
        )
        failed = [atoms for _, _, atoms in branches if "Done" not in _names(atoms=atoms)]
        assert failed, route
        for atoms in failed:
            assert _names(atoms=atoms) == {"After"}, (route, _names(atoms=atoms))
