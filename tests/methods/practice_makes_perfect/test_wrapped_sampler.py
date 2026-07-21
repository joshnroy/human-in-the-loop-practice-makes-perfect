import numpy as np

from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler

# Every sampler in this file is trained with a deliberately tiny iteration cap: the
# whole point of the tests is behavior (layout, argmax, epsilon, determinism), and
# predicators' own 100000-iteration budget would make the file minutes long. The
# separable task below is linearly separable in one feature, so a few hundred
# full-batch Adam steps are plenty.
TEST_MAX_TRAIN_ITERS = 300


def _make_sampler(**kwargs):
    defaults = {
        "skill_name": "TurnOnLight",
        "param_dim": 1,
        "max_train_iters": TEST_MAX_TRAIN_ITERS,
        "num_candidates": 10,
    }
    defaults.update(kwargs)
    return LearnedSkillSampler(**defaults)


def _fit_separable_sampler(*, seed=0, num_observations=80, **kwargs):
    """Train a sampler on the separable synthetic task `label = params[0] > 0.5`,
    with a constant state feature so the only informative input dimension is the
    parameter itself."""
    sampler = _make_sampler(seed=seed, **kwargs)
    rng = np.random.default_rng(123)
    for _ in range(num_observations):
        params = rng.uniform(0.0, 1.0, size=(1,))
        sampler.observe(features=[0.0], params=params, success=bool(params[0] > 0.5))
    sampler.fit()
    return sampler


def _candidate_grid():
    """Ten candidates, five clearly on each side of the 0.5 decision boundary."""
    return [np.array([v]) for v in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]]


def test_build_sampler_input_layout_is_bias_then_state_then_params():
    """Pins predicators' utils.construct_active_sampler_input layout exactly."""
    x = LearnedSkillSampler.build_sampler_input(
        state_features=[2.0, 3.0, 4.0], params=np.array([0.25, 0.75])
    )
    assert x == [1.0, 2.0, 3.0, 4.0, 0.25, 0.75]


def test_build_sampler_input_with_no_state_features():
    x = LearnedSkillSampler.build_sampler_input(state_features=[], params=np.array([0.5]))
    assert x == [1.0, 0.5]


def test_sample_without_training_returns_a_given_candidate():
    sampler = _make_sampler()
    candidates = _candidate_grid()
    chosen, was_random = sampler.sample(features=[0.0], candidates=candidates, explore=False)
    assert any(np.array_equal(chosen, c) for c in candidates)
    assert not np.isnan(chosen).any()
    assert was_random is False


def test_fit_with_no_data_is_a_noop_and_sample_still_works():
    sampler = _make_sampler()
    sampler.fit()
    assert not sampler.is_fitted
    chosen, _ = sampler.sample(features=[0.0], candidates=_candidate_grid(), explore=False)
    assert chosen.shape == (1,)


def test_single_class_data_does_not_crash_and_still_returns_a_candidate():
    """predicators' _NormalizingBinaryClassifier refuses to train on one class and
    falls back to predicting that class constantly; we do the same."""
    sampler = _make_sampler()
    for value in [0.6, 0.7, 0.8, 0.9]:
        sampler.observe(features=[0.0], params=np.array([value]), success=True)
    sampler.fit()
    candidates = _candidate_grid()
    chosen, _ = sampler.sample(features=[0.0], candidates=candidates, explore=False)
    assert any(np.array_equal(chosen, c) for c in candidates)
    scores = sampler.score_candidates(features=[0.0], candidates=candidates)
    assert all(s == 1.0 for s in scores)


def test_learns_a_separable_task_and_greedily_picks_the_good_side():
    """The real learning test: after training on `label = params[0] > 0.5`, the
    greedy (explore=False) choice must land above 0.5 far more often than the 50%
    a uniformly random pick over the symmetric candidate grid would give."""
    sampler = _fit_separable_sampler()
    rng = np.random.default_rng(7)
    num_trials = 20
    num_good = 0
    for _ in range(num_trials):
        candidates = [np.array([v]) for v in rng.uniform(0.0, 1.0, size=10)]
        chosen, _ = sampler.sample(features=[0.0], candidates=candidates, explore=False)
        if chosen[0] > 0.5:
            num_good += 1
    assert num_good >= 18, f"only {num_good}/{num_trials} greedy picks were on the good side"


def test_scores_are_monotone_enough_to_separate_the_two_sides():
    sampler = _fit_separable_sampler()
    candidates = _candidate_grid()
    scores = sampler.score_candidates(features=[0.0], candidates=candidates)
    assert min(scores[5:]) > max(scores[:5])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_exploration_epsilon_one_always_reports_random():
    sampler = _fit_separable_sampler(exploration_epsilon=1.0)
    candidates = _candidate_grid()
    for _ in range(10):
        chosen, was_random = sampler.sample(features=[0.0], candidates=candidates, explore=True)
        assert was_random is True
        assert any(np.array_equal(chosen, c) for c in candidates)


def test_exploration_epsilon_zero_matches_greedy_and_never_reports_random():
    sampler = _fit_separable_sampler(exploration_epsilon=0.0)
    candidates = _candidate_grid()
    greedy, _ = sampler.sample(features=[0.0], candidates=candidates, explore=False)
    for _ in range(10):
        chosen, was_random = sampler.sample(features=[0.0], candidates=candidates, explore=True)
        assert was_random is False
        assert np.array_equal(chosen, greedy)


def test_explore_false_never_reports_random_even_with_epsilon_one():
    """explore=False is the test-time sampler: epsilon must not apply at all
    (predicators wires _wrap_sampler_test with no epsilon branch)."""
    sampler = _fit_separable_sampler(exploration_epsilon=1.0)
    candidates = _candidate_grid()
    for _ in range(5):
        _, was_random = sampler.sample(features=[0.0], candidates=candidates, explore=False)
        assert was_random is False


def test_same_seed_and_same_data_give_identical_choices():
    candidates = _candidate_grid()
    a = _fit_separable_sampler(seed=3, exploration_epsilon=0.5)
    b = _fit_separable_sampler(seed=3, exploration_epsilon=0.5)
    for _ in range(10):
        chosen_a, random_a = a.sample(features=[0.0], candidates=candidates, explore=True)
        chosen_b, random_b = b.sample(features=[0.0], candidates=candidates, explore=True)
        assert np.array_equal(chosen_a, chosen_b)
        assert random_a == random_b


def test_two_instances_do_not_share_training_data():
    """Guards the pydantic mutable-default trap: observed data must be per-instance."""
    a = _make_sampler()
    b = _make_sampler()
    a.observe(features=[0.0], params=np.array([0.9]), success=True)
    assert a.num_observations == 1
    assert b.num_observations == 0


def test_observe_rejects_wrong_param_dim():
    sampler = _make_sampler(param_dim=2)
    try:
        sampler.observe(features=[0.0], params=np.array([0.5]), success=True)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a params vector of the wrong length")


def test_refit_from_scratch_forgets_nothing_and_tracks_all_data():
    """Each learning cycle refits on *all* data (predicators rebuilds the classifier
    from the full dataset every cycle) -- observing more and refitting must keep the
    earlier examples."""
    sampler = _fit_separable_sampler(num_observations=40)
    assert sampler.num_observations == 40
    sampler.observe(features=[0.0], params=np.array([0.99]), success=True)
    sampler.fit()
    assert sampler.num_observations == 41
