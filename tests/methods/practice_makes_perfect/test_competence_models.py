from hitl_pmp.methods.practice_makes_perfect.competence_models import SkillCompetenceModel


def test_initial_competence_uses_the_beta_prior_mean_when_no_observations() -> None:
    model = SkillCompetenceModel(alpha=10.0, beta=1.0)
    assert model.get_current_competence() == 10.0 / 11.0


def test_get_current_competence_reflects_observations_via_the_posterior() -> None:
    model = SkillCompetenceModel(alpha=10.0, beta=1.0)
    for _ in range(5):
        model.observe(outcome=True)
    for _ in range(5):
        model.observe(outcome=False)
    # alpha_n = 10 + 5, beta_n = 1 + 5 -> mean = 15/21
    assert model.get_current_competence() == 15.0 / 21.0


def test_get_current_competence_uses_a_sliding_window_over_recent_nonempty_cycles() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, window_size=2)
    for outcome in [True, True, True]:  # cycle 0: all successes
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [False, False, False]:  # cycle 1: all failures
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [False]:  # cycle 2: one failure
        model.observe(outcome=outcome)

    # window_size=2 -> only cycles 1 and 2 count: 0 successes, 4 observations.
    assert model.get_current_competence() == 1.0 / 6.0


def test_get_current_competence_ignores_empty_cycles_in_the_window() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, window_size=1)
    model.observe(outcome=True)
    model.advance_cycle()
    model.advance_cycle()  # an empty cycle in between -- no observations
    # Current (empty) cycle has no observations either -- the window should
    # fall back to the most recent NONEMPTY cycle, not an empty one.
    assert model.get_current_competence() == 2.0 / 3.0  # alpha=1+1, beta=1 -> 2/3


def test_advance_cycle_starts_a_fresh_observation_list() -> None:
    model = SkillCompetenceModel()
    model.observe(outcome=True)
    model.advance_cycle()
    assert model.cycle_observations[-1] == []
    assert model.cycle_observations[0] == [True]


def test_predict_competence_uses_the_optimistic_bonus_before_two_cycles_of_data() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, initial_prediction_bonus=0.5)
    model.observe(outcome=True)
    current = model.get_current_competence()
    assert model.predict_competence(num_additional_data=10) == min(1.0, current + 0.5)


def test_predict_competence_bonus_is_capped_at_one() -> None:
    model = SkillCompetenceModel(alpha=100.0, beta=1.0, initial_prediction_bonus=0.5)
    model.observe(outcome=True)
    assert model.predict_competence(num_additional_data=10) == 1.0


def test_predict_competence_extrapolates_the_largest_recent_per_cycle_change() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, recency_size=5)
    for outcome in [False, False]:  # cycle 0: rate 0.0
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [True, True]:  # cycle 1: rate 1.0
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [True]:  # cycle 2: rate 1.0 (current cycle, still counts)
        model.observe(outcome=outcome)

    current = model.get_current_competence()
    # best_change = max(0.0, 1.0, 1.0) - min(0.0, 1.0, 1.0) = 1.0
    predicted = model.predict_competence(num_additional_data=1)
    assert predicted == min(1.0, current + 1.0 * 1)


def test_predict_competence_only_considers_the_last_recency_size_cycles() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, recency_size=1)
    for outcome in [False, False]:  # cycle 0: rate 0.0 -- outside the window
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [False]:  # cycle 1: rate 0.0 -- only this one counts
        model.observe(outcome=outcome)

    # Only one cycle is in the recency window -> best_change = 0.0 - 0.0 = 0.0.
    current = model.get_current_competence()
    assert model.predict_competence(num_additional_data=5) == current


def test_predict_competence_extrapolated_gain_is_capped_at_one() -> None:
    model = SkillCompetenceModel(alpha=1.0, beta=1.0, recency_size=5)
    for outcome in [False]:  # cycle 0: rate 0.0
        model.observe(outcome=outcome)
    model.advance_cycle()
    for outcome in [True]:  # cycle 1: rate 1.0
        model.observe(outcome=outcome)

    # best_change = 1.0 - 0.0 = 1.0; a huge num_additional_data would send
    # current + gain far past 1.0 without the clip.
    predicted = model.predict_competence(num_additional_data=1000)
    assert predicted == 1.0
