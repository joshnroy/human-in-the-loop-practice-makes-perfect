"""Tests for OptimisticSkillCompetenceModel.

Every expected number here is worked out by hand from the Beta-Bernoulli
posterior mean formula (alpha + s) / (alpha + beta + n) and from plain
per-cycle means -- never copied back out of the implementation."""

import pytest

from hitl_pmp.methods.practice_makes_perfect.competence_models import (
    LinearSkillCompetenceModel,
    OptimisticSkillCompetenceModel,
)


def _observe_all(*, model, outcomes):
    for outcome in outcomes:
        model.observe(success=outcome)


def test_fresh_model_returns_prior_mean():
    # No data at all: (alpha + 0) / (alpha + beta + 0) = 10 / 11.
    model = OptimisticSkillCompetenceModel()
    assert model.get_current_competence() == pytest.approx(10.0 / 11.0)
    assert model.num_observations == 0


def test_single_cycle_posterior_mean():
    # 3 successes, 1 failure: alpha_n = 10 + 3 = 13, beta_n = (4 - 3) + 1 = 2.
    model = OptimisticSkillCompetenceModel()
    _observe_all(model=model, outcomes=[True, True, True, False])
    assert model.get_current_competence() == pytest.approx(13.0 / 15.0)
    assert model.num_observations == 4


def test_all_failures_posterior_mean():
    # 0 successes, 5 failures: alpha_n = 10, beta_n = 5 + 1 = 6 -> 10 / 16.
    model = OptimisticSkillCompetenceModel()
    _observe_all(model=model, outcomes=[False] * 5)
    assert model.get_current_competence() == pytest.approx(10.0 / 16.0)


def test_advance_cycle_partitions_data():
    model = OptimisticSkillCompetenceModel()
    _observe_all(model=model, outcomes=[True, False])
    model.advance_cycle()
    _observe_all(model=model, outcomes=[True])
    assert model.cycle_observations == [[True, False], [True]]
    # Pooling both cycles (window 5 >= 2 cycles): s = 2, n = 3.
    assert model.get_current_competence() == pytest.approx(12.0 / 14.0)
    assert model.num_observations == 3


def test_empty_cycles_are_skipped():
    """Empty cycles must not count toward the nonempty-window logic, and must
    not push real data out of the window."""
    model = OptimisticSkillCompetenceModel()
    _observe_all(model=model, outcomes=[True, True, False])
    for _ in range(4):
        model.advance_cycle()
    # Still exactly one nonempty cycle: s = 2, n = 3 -> 12 / 14.
    assert model.get_current_competence() == pytest.approx(12.0 / 14.0)
    # And fewer than 2 nonempty cycles, so prediction is the bonus branch.
    assert model.predict_competence(num_additional_data=1) == pytest.approx(
        min(1.0, 12.0 / 14.0 + 0.5)
    )


def test_window_size_truncation():
    """With 6 nonempty cycles only the last 5 are pooled."""
    model = OptimisticSkillCompetenceModel()
    for outcome in [True, True, True, True, True, False]:
        _observe_all(model=model, outcomes=[outcome])
        model.advance_cycle()
    # Last 5 cycles -> [True, True, True, True, False]: s = 4, n = 5.
    # alpha_n = 14, beta_n = (5 - 4) + 1 = 2 -> 14 / 16 = 0.875.
    assert model.get_current_competence() == pytest.approx(0.875)
    # All 6 would have been 15 / 17, which is a genuinely different number.
    assert model.get_current_competence() != pytest.approx(15.0 / 17.0)


def test_custom_window_size():
    model = OptimisticSkillCompetenceModel(window_size=2)
    for outcome in [True, True, False]:
        _observe_all(model=model, outcomes=[outcome])
        model.advance_cycle()
    # Last 2 cycles -> [True, False]: alpha_n = 11, beta_n = 1 + 1 = 2 -> 11 / 13.
    assert model.get_current_competence() == pytest.approx(11.0 / 13.0)


def test_predict_with_fewer_than_two_nonempty_cycles_adds_bonus():
    # Flat prior so the bonus branch is visible below the 1.0 ceiling.
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    _observe_all(model=model, outcomes=[False] * 4)
    # current = (1 + 0) / (1 + 1 + 4) = 1 / 6.
    assert model.get_current_competence() == pytest.approx(1.0 / 6.0)
    assert model.predict_competence(num_additional_data=1) == pytest.approx(1.0 / 6.0 + 0.5)
    # num_additional_data is ignored on this branch (matches predicators).
    assert model.predict_competence(num_additional_data=7) == pytest.approx(1.0 / 6.0 + 0.5)


def test_predict_bonus_branch_clips_at_one():
    # Default prior mean is 10/11 ~= 0.909, so 0.909 + 0.5 must clip to 1.0.
    model = OptimisticSkillCompetenceModel()
    assert model.predict_competence(num_additional_data=1) == pytest.approx(1.0)


def test_predict_monotone_improvement_exceeds_current():
    """Per-cycle raw means 0.2, 0.4, 0.6 -> best_change = 0.4."""
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    for num_successes in (1, 2, 3):
        _observe_all(model=model, outcomes=[True] * num_successes + [False] * (5 - num_successes))
        model.advance_cycle()
    # current: pooled 15 outcomes, s = 6 -> (1 + 6) / (1 + 1 + 15) = 7 / 17.
    current = model.get_current_competence()
    assert current == pytest.approx(7.0 / 17.0)
    # gain = (0.6 - 0.2) * 1 = 0.4.
    assert model.predict_competence(num_additional_data=1) == pytest.approx(7.0 / 17.0 + 0.4)
    assert model.predict_competence(num_additional_data=1) > current
    # 7/17 + 0.4 * 2 = 1.2118..., above the ceiling, so it clips to 1.0.
    assert model.predict_competence(num_additional_data=2) == pytest.approx(1.0)


def test_predict_plateau_equals_current():
    """Identical per-cycle means -> best_change == 0 -> no predicted gain.

    This is the property that makes EES stop practicing a maxed-out skill."""
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    for _ in range(3):
        _observe_all(model=model, outcomes=[True, False])
        model.advance_cycle()
    # current: pooled 6 outcomes, s = 3 -> (1 + 3) / (1 + 1 + 6) = 4 / 8 = 0.5.
    assert model.get_current_competence() == pytest.approx(0.5)
    assert model.predict_competence(num_additional_data=1) == pytest.approx(0.5)
    assert model.predict_competence(num_additional_data=10) == pytest.approx(0.5)


def test_predict_clips_at_upper_bound():
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    _observe_all(model=model, outcomes=[False, False])
    model.advance_cycle()
    _observe_all(model=model, outcomes=[True, True])
    # means 0.0 and 1.0 -> best_change = 1.0; gain = 5.0, way over 1.
    assert model.predict_competence(num_additional_data=5) == pytest.approx(1.0)


def test_predict_clips_at_lower_bound():
    """current == 0 and best_change == 0 would give 0.0; clipped up to 1e-6."""
    model = OptimisticSkillCompetenceModel(alpha=0.0, beta=1.0)
    for _ in range(2):
        _observe_all(model=model, outcomes=[False, False])
        model.advance_cycle()
    assert model.get_current_competence() == pytest.approx(0.0)
    assert model.predict_competence(num_additional_data=1) == pytest.approx(1e-6)


def test_recency_size_truncation():
    """Only the last recency_size nonempty cycles feed max/min of the means."""
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    _observe_all(model=model, outcomes=[True, True])  # mean 1.0, will fall out of recency window
    model.advance_cycle()
    for _ in range(5):
        _observe_all(model=model, outcomes=[False, False])  # mean 0.0 each
        model.advance_cycle()
    # Last 5 means are all 0.0 -> best_change = 0.0, so prediction == current.
    # current pools the last window_size=5 cycles: 10 failures, s = 0 ->
    # (1 + 0) / (1 + 1 + 10) = 1 / 12.
    assert model.get_current_competence() == pytest.approx(1.0 / 12.0)
    assert model.predict_competence(num_additional_data=1) == pytest.approx(1.0 / 12.0)


def test_custom_recency_size():
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0, recency_size=2, window_size=2)
    for num_successes in (0, 1, 2):
        _observe_all(model=model, outcomes=[True] * num_successes + [False] * (4 - num_successes))
        model.advance_cycle()
    # Per-cycle means are 0.0, 0.25, 0.5; the last 2 give best_change = 0.25
    # (all 3 would have given 0.5, so the recency truncation is load-bearing).
    # current pools last 2 cycles: 8 outcomes, s = 3 -> (1 + 3) / (1 + 1 + 8) = 4 / 10.
    assert model.get_current_competence() == pytest.approx(0.4)
    assert model.predict_competence(num_additional_data=1) == pytest.approx(0.65)
    # gain scales linearly with num_additional_data.
    assert model.predict_competence(num_additional_data=2) == pytest.approx(0.9)


def test_two_instances_do_not_share_state():
    first = OptimisticSkillCompetenceModel()
    second = OptimisticSkillCompetenceModel()
    _observe_all(model=first, outcomes=[True, True, True])
    first.advance_cycle()
    assert second.cycle_observations == [[]]
    assert second.num_observations == 0
    assert second.get_current_competence() == pytest.approx(10.0 / 11.0)
    assert first.cycle_observations == [[True, True, True], []]


def test_num_observations_counts_all_cycles_including_out_of_window():
    model = OptimisticSkillCompetenceModel()
    for _ in range(7):
        _observe_all(model=model, outcomes=[True, False])
        model.advance_cycle()
    # 14 total attempts, even though only the last 5 cycles (10) are in-window.
    assert model.num_observations == 14


def test_defaults_match_predicators():
    model = OptimisticSkillCompetenceModel()
    assert model.window_size == 5
    assert model.recency_size == 5
    assert model.alpha == 10.0
    assert model.beta == 1.0
    assert model.initial_prediction_bonus == 0.5


def test_returns_are_plain_floats():
    model = OptimisticSkillCompetenceModel(alpha=1.0, beta=1.0)
    _observe_all(model=model, outcomes=[True, False])
    model.advance_cycle()
    _observe_all(model=model, outcomes=[True, True])
    assert type(model.get_current_competence()) is float
    assert type(model.predict_competence(num_additional_data=1)) is float


# --------------------------------------------------- LinearSkillCompetenceModel
#
# Expected numbers are worked by hand from the Beta posterior mean
# (alpha + s) / (alpha + beta + n), pooled over every outcome so far, and from the


def test_linear_fresh_model_reports_its_priors():
    model = LinearSkillCompetenceModel()
    assert model.get_current_competence() == pytest.approx(10.0 / 11.0)
    assert model.learning_rate() == pytest.approx(0.1)


def test_linear_learning_rate_is_the_papers_running_average_per_attempt():
    model = LinearSkillCompetenceModel()
    _observe_all(model=model, outcomes=[False, False])
    model.advance_cycle()
    prior, first, second = 10.0 / 11.0, 10.0 / 12.0, 10.0 / 13.0
    delta1 = 0.5 * (0.5 * 0.1 + 0.5 * (first - prior)) + 0.5 * (second - first)
    assert model.learning_rate() == pytest.approx(delta1)

    _observe_all(model=model, outcomes=[True, True, True])
    model.advance_cycle()
    third, fourth, fifth = 11.0 / 14.0, 12.0 / 15.0, 13.0 / 16.0
    delta2 = 0.5 * (
        0.5 * (0.5 * delta1 + 0.5 * (third - second)) + 0.5 * (fourth - third)
    ) + 0.5 * (fifth - fourth)
    assert model.learning_rate() == pytest.approx(delta2)
    assert model.get_current_competence() == pytest.approx(fifth)


@pytest.mark.parametrize("smoothing", [0.0, 0.5, 0.9, 1.0])
def test_linear_rate_does_not_depend_on_closed_period_partition(*, smoothing):
    outcomes = [False, False, True, False, True, True, True, False]
    batched = LinearSkillCompetenceModel(smoothing=smoothing)
    episodic = LinearSkillCompetenceModel(smoothing=smoothing)
    _observe_all(model=batched, outcomes=outcomes)
    batched.advance_cycle()
    for outcome in outcomes:
        episodic.observe(success=outcome)
        episodic.advance_cycle()
    assert batched.get_current_competence() == episodic.get_current_competence()
    assert batched.learning_rate() == pytest.approx(episodic.learning_rate())


def test_linear_recorded_seed_zero_failures_keep_signed_episode_updates():
    probability = 0.1326530612244898
    model = LinearSkillCompetenceModel(
        alpha=2 * probability,
        beta=2 * (1 - probability),
        initial_delta=0.00620110544217687,
        cycle_observations=[[False] * 8, []],
    )
    expected = model.initial_delta
    previous = probability
    for count in range(1, 9):
        current = 2 * probability / (2 + count)
        expected = 0.5 * expected + 0.5 * (current - previous)
        previous = current
    assert model.learning_rate() == pytest.approx(expected)
    assert model.learning_rate() < 0


def test_linear_open_cycle_moves_competence_but_not_the_learning_rate():
    model = LinearSkillCompetenceModel()
    _observe_all(model=model, outcomes=[False])
    assert model.get_current_competence() == pytest.approx(10.0 / 12.0)
    assert model.learning_rate() == pytest.approx(0.1)


def test_linear_idle_cycle_leaves_the_learning_rate_alone():
    # A cycle in which the skill was never practised carries no evidence about how
    # fast practice improves it, so it must not decay Delta toward zero.
    model = LinearSkillCompetenceModel()
    model.advance_cycle()
    model.advance_cycle()
    assert model.learning_rate() == pytest.approx(0.1)


def test_linear_smoothing_and_priors_are_configurable():
    model = LinearSkillCompetenceModel(alpha=1.0, beta=1.0, initial_delta=0.2, smoothing=0.0)
    assert model.get_current_competence() == pytest.approx(0.5)
    _observe_all(model=model, outcomes=[True])
    model.advance_cycle()
    # eps = 0: Delta is exactly the last per-attempt change, (2/3 - 1/2) / 1.
    assert model.learning_rate() == pytest.approx(2.0 / 3.0 - 0.5)


def test_linear_prediction_is_clipped_to_the_unit_interval():
    model = LinearSkillCompetenceModel(alpha=1.0, beta=1.0, initial_delta=0.1)
    assert model.predict_competence(num_attempts=0) == pytest.approx(0.5)
    assert model.predict_competence(num_attempts=3) == pytest.approx(0.8)
    assert model.predict_competence(num_attempts=50) == 1.0
    falling = LinearSkillCompetenceModel(alpha=1.0, beta=1.0, initial_delta=-0.1)
    assert falling.predict_competence(num_attempts=50) == 0.0
