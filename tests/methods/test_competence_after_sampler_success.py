"""Informed-only competence excludes only picks made while the sampler has no success yet.

`LearnedSkillSampler.sample` reports a pick UNINFORMATIVE whenever its scores tie over
most candidates. That covers an unfitted sampler and a one-class fit of failures (no
success yet, so a uniform draw), but also a one-class fit of successes: a skill that has
never failed. Excluding the last kind left such a skill, sweep3 on 3D Sweeping, at its
prior for good (coordinator for Josh, 2026-11-24). #884's stated intent, "no success yet,
so a uniform draw", excludes only the first two.
"""

import pytest

from hitl_pmp.core.method.types import SamplerConsultation
from hitl_pmp.methods.belief_space.types.belief_state import SamplerTrainingState
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

C = SamplerConsultation


def _sampler(*, labels: list[bool], fit: bool = True) -> LearnedSkillSampler:
    sampler = LearnedSkillSampler(skill_name="sweep3", param_dim=1, max_train_iters=5)
    for index, success in enumerate(labels):
        sampler.observe(sampler_input=[1.0, float(index)], success=success)
    if fit:
        sampler.fit()
    return sampler


@pytest.mark.parametrize(
    ("labels", "fit", "has_success"),
    [
        ([], True, False),
        ([False, False], True, False),
        ([True, True], True, True),
        ([False, True], True, True),
        ([True], False, False),
    ],
)
def test_a_sampler_knows_whether_its_fit_saw_a_success(*, labels, fit, has_success):
    assert _sampler(labels=labels, fit=fit).fitted_with_success is has_success


@pytest.mark.parametrize(
    ("consultation", "labels", "excluded"),
    [
        (C.UNINFORMATIVE, [], True),
        (C.UNINFORMATIVE, [False, False], True),
        (C.UNINFORMATIVE, [True, True], False),
        (C.INFORMED, [True, False], False),
        (C.EPSILON_RANDOM, [], False),
        (C.NO_SAMPLER, [], False),
    ],
)
def test_ees_excludes_only_a_pick_made_before_the_samplers_first_success(
    *, consultation, labels, excluded
):
    method = EesMethod.model_construct()
    method._samplers = {"sweep3": _sampler(labels=labels)}  # noqa: SLF001
    assert (
        method.picked_before_sampler_success(skill_name="sweep3", consultation=consultation)
        is excluded
    )


def test_an_all_success_skill_updates_competence_and_a_no_success_skill_does_not():
    from hitl_pmp.methods.practice_makes_perfect.competence_models import (
        OptimisticSkillCompetenceModel,
    )

    method = EesMethod.model_construct(
        competence_from_informed_only=True, reproduce_predicators_double_observe=False
    )
    method._samplers = {  # noqa: SLF001
        "sweep3": _sampler(labels=[True, True]),
        "gather": _sampler(labels=[False, False]),
    }
    for name, observed in (("sweep3", 1), ("gather", 0)):
        model = OptimisticSkillCompetenceModel()
        EesMethod.record_competence_outcome(
            method,
            model=model,
            success=True,
            was_random_exploration=False,
            was_uninformative=method.picked_before_sampler_success(
                skill_name=name, consultation=C.UNINFORMATIVE
            ),
        )
        assert model.num_observations == observed, name
        assert model.num_tries == 1, name


@pytest.mark.parametrize(
    ("consultation", "has_success", "admitted"),
    [
        (C.UNINFORMATIVE, True, True),
        (C.UNINFORMATIVE, False, False),
        (C.INFORMED, False, True),
        (C.EPSILON_RANDOM, True, False),
        (C.NO_SAMPLER, False, True),
    ],
)
def test_ours_sampler_succeeded_evidence_admits_after_the_first_success(
    *, consultation, has_success, admitted
):
    assert (
        CompetenceEvidence.SAMPLER_SUCCEEDED.admits(
            consultation=consultation, sampler_has_success=has_success
        )
        is admitted
    )


def test_the_other_evidence_rules_ignore_whether_the_sampler_has_succeeded():
    for evidence in (
        CompetenceEvidence.ALL,
        CompetenceEvidence.NON_EPSILON,
        CompetenceEvidence.INFORMED,
    ):
        for consultation in C:
            assert evidence.admits(
                consultation=consultation, sampler_has_success=True
            ) == evidence.admits(consultation=consultation)


def test_ours_forecast_conditions_a_greedy_draw_from_an_all_success_fit():
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        greedy_draw_admitted,
    )

    all_success = SamplerTrainingState(successes=2, fitted_successes=2)
    all_failure = SamplerTrainingState(failures=2, fitted_failures=2)
    evidence = CompetenceEvidence.SAMPLER_SUCCEEDED
    assert greedy_draw_admitted(evidence=evidence, training=all_success)
    assert not greedy_draw_admitted(evidence=evidence, training=all_failure)
    assert not greedy_draw_admitted(evidence=CompetenceEvidence.INFORMED, training=all_success)
    assert greedy_draw_admitted(evidence=evidence, training=None)
