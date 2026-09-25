"""Which practice outcomes condition competence: all, non-epsilon, or informed only."""

import argparse

import numpy as np
import pytest

from hitl_pmp.core.method.types import SamplerConsultation
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

C = SamplerConsultation
E = CompetenceEvidence
ADMITS = {
    E.ALL: {C.INFORMED, C.UNINFORMATIVE, C.EPSILON_RANDOM, C.NO_SAMPLER},
    E.NON_EPSILON: {C.INFORMED, C.UNINFORMATIVE, C.NO_SAMPLER},
    E.INFORMED: {C.INFORMED, C.NO_SAMPLER},
}


@pytest.mark.parametrize("evidence", list(E))
@pytest.mark.parametrize("consultation", list(C))
def test_admission_table(
    *, evidence: CompetenceEvidence, consultation: SamplerConsultation
) -> None:
    assert evidence.admits(consultation=consultation) == (consultation in ADMITS[evidence])


def _method(*, evidence: CompetenceEvidence | None) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    extra = {} if evidence is None else {"pomdp_competence_evidence": evidence}
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=5,
        pomdp_competence_model="global_curve",
        pomdp_inference_engine="grid",
        pomdp_num_particles=64,
        sampler_max_train_iters=1,
        **extra,
    )
    env.current_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return method


def _belief(*, method: Tossing3DPomdpMethod, name: str) -> BayesianSkillBelief:
    belief = method.pomdp_state.skill_beliefs[name]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


def _attempt(
    *, method: Tossing3DPomdpMethod, name: str, consultation: SamplerConsultation, success: bool
) -> None:
    ground = next(g for g in method._pomdp_model.ground_skills if g.skill.name == name)  # noqa: SLF001
    method.record_practice_attempt(skill_name=name, success=success, consultation=consultation)
    method.observe_outcome(
        ground_skill=ground,
        success=success,
        was_random_exploration=consultation is C.EPSILON_RANDOM,
    )
    if consultation is not C.NO_SAMPLER:
        method.observe_sampler_outcome(
            skill_name=name, param_dim=4, sampler_input=[float(success)], success=success
        )


def test_default_is_todays_non_epsilon_behaviour() -> None:
    assert Tossing3DPomdpMethod.model_fields["pomdp_competence_evidence"].default == E.NON_EPSILON
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    assert parser.parse_args([]).pomdp_competence_evidence == "non_epsilon"
    args = parser.parse_args(["--pomdp-competence-evidence", "informed"])
    assert args.pomdp_competence_evidence == "informed"


@pytest.mark.parametrize("evidence", list(E))
def test_toss_competence_is_filtered_but_the_clock_counts_every_attempt(
    *, evidence: CompetenceEvidence
) -> None:
    method = _method(evidence=evidence)
    toss_kinds = (C.INFORMED, C.UNINFORMATIVE, C.EPSILON_RANDOM)
    for consultation in toss_kinds:
        before = _belief(method=method, name=TOSS_SKILL)
        _attempt(method=method, name=TOSS_SKILL, consultation=consultation, success=False)
        after = _belief(method=method, name=TOSS_SKILL)
        conditioned = after.cycle_failures == before.cycle_failures + 1
        assert conditioned == evidence.admits(consultation=consultation), consultation
        assert after.cost_belief != before.cost_belief
    assert method.pomdp_state.pending_examples[TOSS_SKILL] == len(toss_kinds)
    method.end_cycle()
    assert _belief(method=method, name=TOSS_SKILL).total_training_examples == len(toss_kinds)


@pytest.mark.parametrize("evidence", list(E))
def test_skills_without_a_sampler_are_always_conditioned(*, evidence: CompetenceEvidence) -> None:
    method = _method(evidence=evidence)
    before = _belief(method=method, name=PICK_SKILL)
    _attempt(method=method, name=PICK_SKILL, consultation=C.NO_SAMPLER, success=True)
    assert _belief(method=method, name=PICK_SKILL).cycle_successes == before.cycle_successes + 1


@pytest.mark.parametrize("evidence", list(E))
@pytest.mark.parametrize("mixed", [False, True])
def test_search_forecasts_under_the_same_evidence_rule(
    *, evidence: CompetenceEvidence, mixed: bool
) -> None:
    method = _method(evidence=evidence)
    training = (
        SamplerTrainingState(successes=1, failures=1, fitted_successes=1, fitted_failures=1)
        if mixed
        else SamplerTrainingState(failures=1, fitted_failures=1)
    )
    state = method.pomdp_state.model_copy(update={"sampler_training": {TOSS_SKILL: training}})
    model = method._pomdp_model  # noqa: SLF001
    toss = next(g for g in model.ground_skills if g.skill.name == TOSS_SKILL)
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=toss.preconditions),
        state=state,
        action=toss,
    )
    prior = state.skill_beliefs[TOSS_SKILL]
    assert isinstance(prior, BayesianSkillBelief)
    conditioned_mass = sum(
        probability
        for probability, branch, _ in outcomes
        if isinstance(belief := branch.skill_beliefs[TOSS_SKILL], BayesianSkillBelief)
        and belief.cycle_successes + belief.cycle_failures == 1
    )
    epsilon = model.exploration_epsilon if mixed else 0.0
    # A greedy draw from a one-class fit is uninformative; from a mixed fit, informed.
    greedy_kind = C.INFORMED if mixed else C.UNINFORMATIVE
    expected = (1 - epsilon) * evidence.admits(consultation=greedy_kind) + epsilon * (
        evidence.admits(consultation=C.EPSILON_RANDOM)
    )
    assert conditioned_mass == pytest.approx(expected)
