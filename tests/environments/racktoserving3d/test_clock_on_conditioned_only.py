"""`--pomdp-clock-on-conditioned-only`: a try the evidence rule keeps out of competence
also leaves the learning curve's training clock where it was (Josh, 2026-11-25).

Without it, Model B's predicted competence min(1, c + rate * n) climbed with every
excluded try: on 3D Sweeping seed 2 ours made 7 sweep5 tries, all failed, none
conditioned, and still believed sweep5 at 0.943. Opt-in, so every existing run, and
Tossing under `informed`, is unchanged.
"""

import argparse

import numpy as np
import pytest

from hitl_pmp.core.method.types import SamplerConsultation
from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingSkillProvider
from hitl_pmp.environments.racktoserving3d.toy_environment import RackServingToyEnvironment
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.rewarded_routes_domain import RewardedRoutesPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

C = SamplerConsultation
E = CompetenceEvidence
SWEEP5 = "sweep5"


def _serving(*, clock_on_conditioned_only: bool) -> Tossing3DPomdpMethod:
    env = RackServingToyEnvironment()
    provider = RackServingSkillProvider(env=env)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=provider,
        seed=2,
        pomdp_domain=RewardedRoutesPracticeDomain.from_provider(provider=provider),
        pomdp_competence_model="local_trend",
        pomdp_inference_engine="grid",
        pomdp_competence_evidence=E.SAMPLER_SUCCEEDED,
        competence_after_first_success=True,
        pomdp_clock_on_conditioned_only=clock_on_conditioned_only,
    )
    env.hard_reset()
    return method


def _tossing(*, evidence: CompetenceEvidence, clock_on_conditioned_only: bool):
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=5,
        pomdp_competence_model="global_curve",
        pomdp_inference_engine="grid",
        pomdp_num_particles=64,
        sampler_max_train_iters=1,
        pomdp_competence_evidence=evidence,
        pomdp_clock_on_conditioned_only=clock_on_conditioned_only,
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


def _try(
    *,
    method: Tossing3DPomdpMethod,
    name: str,
    consultation: SamplerConsultation,
    success: bool,
    training_row: bool = False,
) -> None:
    ground = next(g for g in method._pomdp_model.ground_skills if g.skill.name == name)  # noqa: SLF001
    method.record_practice_attempt(skill_name=name, success=success, consultation=consultation)
    method.observe_outcome(
        ground_skill=ground,
        success=success,
        was_random_exploration=consultation is C.EPSILON_RANDOM,
    )
    if training_row:
        method.observe_sampler_outcome(
            skill_name=name, param_dim=4, sampler_input=[float(success)], success=success
        )


def test_the_flag_is_off_by_default() -> None:
    assert Tossing3DPomdpMethod.model_fields["pomdp_clock_on_conditioned_only"].default is False
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    assert parser.parse_args([]).pomdp_clock_on_conditioned_only is False
    args = parser.parse_args(["--pomdp-clock-on-conditioned-only"])
    assert args.pomdp_clock_on_conditioned_only is True


def test_an_excluded_try_moves_neither_the_belief_nor_the_clock() -> None:
    method = _serving(clock_on_conditioned_only=True)
    before = method.pomdp_state
    # sweep5's first try: no success yet, so a uniform pick, kept out of competence.
    for _ in range(3):
        _try(method=method, name=SWEEP5, consultation=C.UNINFORMATIVE, success=False)
    assert method.pomdp_state.pending_examples.get(SWEEP5, 0) == 0
    assert method.pomdp_state.skill_beliefs[SWEEP5] == before.skill_beliefs[SWEEP5]
    method.end_cycle()
    assert _belief(method=method, name=SWEEP5).total_training_examples == 0


def test_a_conditioned_try_advances_both_as_before() -> None:
    method = _serving(clock_on_conditioned_only=True)
    _try(method=method, name=SWEEP5, consultation=C.INFORMED, success=True)
    assert method.pomdp_state.pending_examples[SWEEP5] == 1
    assert _belief(method=method, name=SWEEP5).cycle_successes == 1
    method.end_cycle()
    assert _belief(method=method, name=SWEEP5).total_training_examples == 1


def test_without_the_flag_an_excluded_try_still_advances_the_clock() -> None:
    method = _serving(clock_on_conditioned_only=False)
    _try(method=method, name=SWEEP5, consultation=C.UNINFORMATIVE, success=False)
    assert method.pomdp_state.pending_examples[SWEEP5] == 1
    assert _belief(method=method, name=SWEEP5).cycle_failures == 0


@pytest.mark.parametrize("clock_on_conditioned_only", [False, True])
def test_a_sampler_skills_clock_counts_only_admitted_tries_under_the_flag(
    *, clock_on_conditioned_only: bool
) -> None:
    evidence = E.INFORMED
    method = _tossing(evidence=evidence, clock_on_conditioned_only=clock_on_conditioned_only)
    kinds = (C.INFORMED, C.UNINFORMATIVE, C.EPSILON_RANDOM)
    for consultation in kinds:
        _try(
            method=method,
            name=TOSS_SKILL,
            consultation=consultation,
            success=False,
            training_row=True,
        )
    admitted = sum(evidence.admits(consultation=kind) for kind in kinds)
    expected = admitted if clock_on_conditioned_only else len(kinds)
    assert method.pomdp_state.pending_examples[TOSS_SKILL] == expected
    # The sampler's own training data keeps every row either way.
    assert method.pomdp_state.sampler_training[TOSS_SKILL].failures == len(kinds)
    assert _belief(method=method, name=TOSS_SKILL).cycle_failures == admitted
