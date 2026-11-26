"""The belief-space model of the always-ask reset follows `--always-ask-reset-skills`
(Josh, 2026-11-26).

The wrapper forces the human's full reset only after the named skills -- on 3D
Sweeping the two sweeps, never gather, whose reset would undo it before sweep5. The
search model used to apply the reset after every skill with continuous parameters, so
ours planned as if gather were always undone and gather -> sweep5 could never be
reached. With no skills named, the model is unchanged: a reset after every skill with
continuous parameters, as Pouring and Tossing ran it.
"""

import pytest

from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    apply_success_effects,
    make_tossing3d_search_state,
)
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import AlwaysAskReset

PRICE = 0.3


def _serving(*, trigger_skills: frozenset[str] | None) -> Tossing3DPomdpMethod:
    env = RackServingToyEnvironment()
    provider = RackServingSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=0,
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
        pomdp_competence_model="local_trend",
        pomdp_inference_engine="grid",
        pomdp_competence_evidence=CompetenceEvidence.SAMPLER_SUCCEEDED,
        always_ask_reset=AlwaysAskReset(cost=PRICE, trigger_skills=trigger_skills),
    )
    env.hard_reset()
    method.model_forced_start_reset(start_atoms=_start(method=method))
    return method


def _start(*, method: Tossing3DPomdpMethod):
    return method.abstract_state(state=method.env.current_state)


def _ground(*, method: Tossing3DPomdpMethod, name: str, atoms):
    return next(
        skill
        for skill in method._pomdp_model.ground_skills  # noqa: SLF001
        if skill.skill.name == name and skill.preconditions <= atoms
    )


def _outcomes(*, method: Tossing3DPomdpMethod, name: str, atoms, reset_modelled: bool):
    model = method._pomdp_model  # noqa: SLF001
    if not reset_modelled:
        model = model.model_copy(update={"forced_start_reset": None})
    state = method.pomdp_state
    return model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=atoms),
        state=state,
        action=_ground(method=method, name=name, atoms=atoms),
    )


def _gathered(*, method: Tossing3DPomdpMethod):
    start = _start(method=method)
    return apply_success_effects(
        true_atoms=start,
        ground_skill=_ground(method=method, name="gather", atoms=start),
        effects=method._pomdp_model._effects,  # noqa: SLF001
    )


def test_gather_is_not_followed_by_a_reset_when_only_the_sweeps_trigger_one():
    method = _serving(trigger_skills=frozenset({"sweep3", "sweep5"}))
    start = _start(method=method)
    assert _ground(method=method, name="gather", atoms=start).skill.param_dim > 0
    modelled = _outcomes(method=method, name="gather", atoms=start, reset_modelled=True)
    naive = _outcomes(method=method, name="gather", atoms=start, reset_modelled=False)
    assert modelled == naive
    assert any(atoms != start for _, _, atoms in modelled)


@pytest.mark.parametrize("sweep", ["sweep3", "sweep5"])
def test_a_named_sweep_is_still_followed_by_the_priced_reset(*, sweep: str):
    method = _serving(trigger_skills=frozenset({"sweep3", "sweep5"}))
    start = _start(method=method)
    atoms = start if sweep == "sweep3" else _gathered(method=method)
    modelled = _outcomes(method=method, name=sweep, atoms=atoms, reset_modelled=True)
    naive = _outcomes(method=method, name=sweep, atoms=atoms, reset_modelled=False)
    assert len(modelled) == len(naive)
    for (p, belief, landed), (q, naive_belief, _) in zip(modelled, naive, strict=True):
        assert p == q
        assert landed == start
        assert belief.skill_beliefs == naive_belief.skill_beliefs
        assert belief.accumulated_cost == pytest.approx(naive_belief.accumulated_cost + PRICE)


def test_with_no_skills_named_every_parameterized_skill_is_reset_as_before():
    """Pouring and Tossing ran the always-ask arm without the skills flag: unchanged."""
    method = _serving(trigger_skills=None)
    start = _start(method=method)
    modelled = _outcomes(method=method, name="gather", atoms=start, reset_modelled=True)
    naive = _outcomes(method=method, name="gather", atoms=start, reset_modelled=False)
    for (p, belief, landed), (q, naive_belief, _) in zip(modelled, naive, strict=True):
        assert p == q
        assert landed == start
        assert belief.accumulated_cost == pytest.approx(naive_belief.accumulated_cost + PRICE)


def test_the_modelled_reset_carries_the_wrappers_trigger_skills():
    named = frozenset({"sweep3", "sweep5"})
    assert _serving(trigger_skills=named)._pomdp_model.forced_start_reset.trigger_skills == named  # noqa: SLF001
    assert _serving(trigger_skills=None)._pomdp_model.forced_start_reset.trigger_skills is None  # noqa: SLF001
