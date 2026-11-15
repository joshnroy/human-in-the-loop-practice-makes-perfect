"""`competence_from_informed_only` (Josh, 2026-11-15): a skill's competence is learned
only from the sampler's informed picks. An uninformative pick (the sampler still has no
success to learn from, so it draws uniformly) is not observed but still counts as a try,
so the deploy-time curve conditions each informed outcome at the try it happened."""

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution

from hitl_pmp.methods.deployment import CurrentCompetence
from hitl_pmp.methods.practice_makes_perfect.competence_models import (
    OptimisticSkillCompetenceModel,
)

PRIOR = (0.12, 10.0, 0.03, 0.015)


def by_hand(*, outcomes_at: list[tuple[int, bool]], total: int) -> float:
    starts = (np.arange(CurrentCompetence.START_GRID) + 0.5) / CurrentCompetence.START_GRID
    rates = np.linspace(0.0, 0.03 + 4 * 0.015, CurrentCompetence.RATE_GRID)
    c, r = np.meshgrid(starts, rates, indexing="ij")
    log_w = beta_distribution.logpdf(c, 1.2, 8.8) - 0.5 * ((r - 0.03) / 0.015) ** 2
    for i, success in outcomes_at:
        p = np.clip(c + r * i, 1e-9, 1 - 1e-9)
        log_w = log_w + (np.log(p) if success else np.log(1 - p))
    w = np.exp(log_w - log_w.max())
    return float((w * np.minimum(1.0, c + r * total)).sum() / w.sum())


def test_a_skipped_try_is_not_observed_but_moves_the_try_index() -> None:
    model = OptimisticSkillCompetenceModel()
    model.skip_try()
    model.skip_try()
    model.observe(success=True)
    model.skip_try()
    model.observe(success=False)
    assert model.cycle_observations == [[True, False]]
    assert model.observation_try_indices() == [2, 4]
    assert model.num_tries == 5


def test_without_skips_the_try_index_is_the_observation_index() -> None:
    model = OptimisticSkillCompetenceModel()
    for outcome in (False, True, True):
        model.observe(success=outcome)
    assert model.observation_try_indices() == [0, 1, 2]
    assert model.num_tries == 3


def test_current_competence_conditions_each_outcome_at_its_own_try() -> None:
    got = CurrentCompetence.estimate(
        prior=PRIOR, outcomes=[True, True], try_indices=[10, 12], num_tries=14
    )
    assert got == pytest.approx(by_hand(outcomes_at=[(10, True), (12, True)], total=14))
    unindexed = CurrentCompetence.estimate(prior=PRIOR, outcomes=[True, True])
    assert unindexed == pytest.approx(by_hand(outcomes_at=[(0, True), (1, True)], total=2))
    assert got != pytest.approx(unindexed)


def test_ees_drops_an_uninformative_pick_only_when_the_flag_is_on() -> None:
    from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

    for flag, expected in ((False, [[False, True]]), (True, [[True]])):
        method = EesMethod.model_construct(competence_from_informed_only=flag)
        model = OptimisticSkillCompetenceModel()
        EesMethod.record_competence_outcome(
            method,
            model=model,
            success=False,
            was_random_exploration=False,
            was_uninformative=True,
        )
        EesMethod.record_competence_outcome(
            method,
            model=model,
            success=True,
            was_random_exploration=False,
            was_uninformative=False,
        )
        assert model.cycle_observations == expected
        assert model.num_tries == 2
