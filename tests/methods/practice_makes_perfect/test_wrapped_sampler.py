import numpy as np
import pytest
import torch
from pydantic import Field, ValidationError

from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import (
    LearnedSkillSampler,
    MlpBinaryClassifier,
    SamplerChoice,
)

# Every sampler in this file is trained with a deliberately tiny iteration cap: the
# whole point of the tests is behavior (layout, argmax, epsilon, determinism), and
# predicators' own 100000-iteration budget would make the file minutes long. The
# separable task below is linearly separable in one feature, so a few hundred
# full-batch Adam steps are plenty.
TEST_MAX_TRAIN_ITERS = 300


def _thread_probe_data() -> tuple[np.ndarray, np.ndarray]:
    """Deliberately wide (64 features, 400 rows): a reduction only has something to
    reassociate across threads when the dot products are long enough to be split."""
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(400, 64))
    y_data = (x_data[:, 0] + 0.5 * x_data[:, 1] > 0).astype(float)
    return x_data, y_data


def _fit_probe_classifier(*, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    classifier = MlpBinaryClassifier(seed=0, max_train_iters=TEST_MAX_TRAIN_ITERS)
    classifier.fit(x_data=x_data, y_data=y_data)
    return classifier.predict_proba(x_data=x_data)


def _row(*, params: np.ndarray, features=(0.0,)) -> list[float]:
    """The default ("all") classifier input row for a candidate -- the sampler now
    consumes prebuilt rows, so the tests build them the same way EesMethod does."""
    return LearnedSkillSampler.build_sampler_input(state_features=list(features), params=params)


def _rows(*, candidates: list[np.ndarray], features=(0.0,)) -> list[list[float]]:
    return [_row(params=c, features=features) for c in candidates]


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
        sampler.observe(sampler_input=_row(params=params), success=bool(params[0] > 0.5))
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
    choice = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    assert any(np.array_equal(choice.params, c) for c in candidates)
    assert not np.isnan(choice.params).any()
    assert choice.was_random is False


def test_fit_with_no_data_is_a_noop_and_sample_still_works():
    sampler = _make_sampler()
    sampler.fit()
    assert not sampler.is_fitted
    candidates = _candidate_grid()
    choice = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    assert choice.params.shape == (1,)


def test_single_class_data_does_not_crash_and_still_returns_a_candidate():
    """predicators' _NormalizingBinaryClassifier refuses to train on one class and
    falls back to predicting that class constantly; we do the same."""
    sampler = _make_sampler()
    for value in [0.6, 0.7, 0.8, 0.9]:
        sampler.observe(sampler_input=_row(params=np.array([value])), success=True)
    sampler.fit()
    candidates = _candidate_grid()
    choice = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    assert any(np.array_equal(choice.params, c) for c in candidates)
    scores = sampler.score_inputs(sampler_inputs=_rows(candidates=candidates))
    assert all(s == 1.0 for s in scores)


def test_all_negative_data_does_not_pin_the_choice_to_the_first_candidate():
    """Deviation 6's contract: when the classifier cannot discriminate, the pick must
    be a uniform draw, not "whichever candidate the caller happened to draw first".

    All-negative data trips `MlpBinaryClassifier`'s single-class shortcut, which sets
    `_single_class_prediction = 0.0` without building a net. `score_inputs` then
    returns one identical value per candidate, so an argmax over them is decided
    entirely by draw order."""
    sampler = _make_sampler()
    for value in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
        sampler.observe(sampler_input=_row(params=np.array([value])), success=False)
    sampler.fit()
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    chosen = [
        sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False).params
        for _ in range(50)
    ]
    num_first = sum(1 for c in chosen if np.array_equal(c, candidates[0]))
    assert num_first < 50, (
        f"{num_first}/50 draws returned candidates[0]; an all-negative classifier "
        "carries no information, so the pick must be uniform over the candidates"
    )
    # `< 50` alone would still pass on a sampler pinned to candidates[1] instead, so
    # also require the draw to actually range over the candidate set. Under a uniform
    # draw over 10 candidates, 50 draws miss at least one with probability ~7e-3.
    distinct = {next(i for i, c in enumerate(candidates) if np.array_equal(p, c)) for p in chosen}
    assert distinct == set(range(len(candidates))), (
        f"50 draws only ever returned candidates {sorted(distinct)}"
    )


class _FixedScoreSampler(LearnedSkillSampler):
    """A sampler whose classifier scores are dictated by the test.

    The tie-breaking and can-it-discriminate branches are properties of the *score
    vector*, and an MLP cannot be asked for an exact tie on demand -- so the one
    input those branches read is supplied directly instead of trained for."""

    scores: list[float] = Field(default_factory=list)

    def score_inputs(self, *, sampler_inputs: list[list[float]]) -> list[float]:
        assert len(sampler_inputs) == len(self.scores)
        return list(self.scores)


def _fixed(*, scores, **kwargs):
    return _FixedScoreSampler(
        skill_name="TurnOnLight", param_dim=1, num_candidates=len(scores), scores=scores, **kwargs
    )


def test_a_tie_for_the_best_score_is_broken_randomly_not_at_the_lowest_index():
    """`np.argmax` returns the first maximal index, so a tie was previously decided by
    the order the caller drew its candidates in -- and reported as a deliberate
    choice. Three of ten candidates tie here, which is under the
    `uninformative_tie_fraction`, so the pick must stay inside the tied set *and*
    move around within it."""
    scores = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    tied = {1, 3, 6}
    sampler = _fixed(scores=scores)
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    picked = set()
    for _ in range(50):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False)
        index = next(i for i, c in enumerate(candidates) if np.array_equal(choice.params, c))
        assert index in tied, f"picked index {index}, which does not attain the maximum score"
        assert choice.was_informed is True
        picked.add(index)
    assert picked == tied, f"only ever picked {sorted(picked)} of the tied {sorted(tied)}"


def test_a_maximum_shared_by_most_of_the_candidates_counts_as_no_discrimination():
    """Six of ten candidates tie at the top, above the 0.5 default, so the score
    vector is treated as carrying no information: the draw is uniform over *all*
    candidates, including ones that did not attain the maximum."""
    sampler = _fixed(scores=[1.0] * 6 + [0.0] * 4)
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    picked = set()
    for _ in range(200):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False)
        assert choice.was_informed is False
        assert choice.was_random is False
        picked.add(next(i for i, c in enumerate(candidates) if np.array_equal(choice.params, c)))
    assert picked == set(range(10)), f"never drew {sorted(set(range(10)) - picked)}"


def test_exactly_the_tie_fraction_still_counts_as_discrimination():
    """The boundary is strict: `uninformative_tie_fraction` of the candidates tying
    is still informative, and one more is not. This is the semantics of the whole
    change, so it is pinned rather than left to the two tests either side of it."""
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    at_the_line = _fixed(scores=[1.0] * 5 + [0.0] * 5)
    over_the_line = _fixed(scores=[1.0] * 6 + [0.0] * 4)
    assert at_the_line.sample(
        sampler_inputs=rows, candidates=candidates, explore=False
    ).was_informed
    assert not over_the_line.sample(
        sampler_inputs=rows, candidates=candidates, explore=False
    ).was_informed


def test_a_non_default_tie_fraction_moves_the_line():
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    strict = _fixed(scores=[1.0] * 3 + [0.0] * 7, uninformative_tie_fraction=0.2)
    assert not strict.sample(sampler_inputs=rows, candidates=candidates, explore=False).was_informed


def test_epsilon_can_still_escape_an_informative_tie():
    """The tie-break confines the *greedy* pick to the tied set; the epsilon branch
    must remain free to draw outside it, or exploration would be silently narrowed
    to whatever the classifier already likes."""
    scores = [0.0] * 7 + [1.0] * 3
    sampler = _fixed(scores=scores, exploration_epsilon=1.0)
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    outside = 0
    for _ in range(50):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True)
        assert choice.was_random is True
        index = next(i for i, c in enumerate(candidates) if np.array_equal(choice.params, c))
        outside += index < 7
    assert outside > 0, "every epsilon-random draw stayed inside the tied set"


def test_a_choice_cannot_be_both_random_and_informed():
    with pytest.raises(ValidationError):
        SamplerChoice(params=np.array([0.5]), was_random=True, was_informed=True)


def test_an_uninformative_draw_is_not_reported_as_epsilon_random():
    """`was_random` means exactly "the epsilon-greedy branch fired", and
    `EesMethod.observe_outcome` suppresses the competence update when it is set.
    Reporting the deviation-6 fallback as random would stop a skill that has never
    succeeded from ever receiving competence evidence, pinning it at its prior --
    so the fallback must keep `was_random` False while still saying, through
    `was_informed`, that nothing was learned."""
    sampler = _fixed(scores=[0.0] * 10, exploration_epsilon=1.0)
    candidates = _candidate_grid()
    rows = _rows(candidates=candidates)
    for _ in range(20):
        choice = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=True)
        assert choice.was_random is False
        assert choice.was_informed is False


def test_an_unfitted_sampler_reports_that_its_draw_was_not_informed():
    sampler = _make_sampler()
    candidates = _candidate_grid()
    choice = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    assert choice.was_informed is False


def test_a_discriminating_classifier_reports_an_informed_draw():
    sampler = _fit_separable_sampler()
    candidates = _candidate_grid()
    choice = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    assert choice.was_informed is True
    assert choice.params[0] > 0.5


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
        choice = sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
        )
        if choice.params[0] > 0.5:
            num_good += 1
    assert num_good >= 18, f"only {num_good}/{num_trials} greedy picks were on the good side"


def test_scores_are_monotone_enough_to_separate_the_two_sides():
    sampler = _fit_separable_sampler()
    candidates = _candidate_grid()
    scores = sampler.score_inputs(sampler_inputs=_rows(candidates=candidates))
    assert min(scores[5:]) > max(scores[:5])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_exploration_epsilon_one_always_reports_random():
    sampler = _fit_separable_sampler(exploration_epsilon=1.0)
    candidates = _candidate_grid()
    for _ in range(10):
        choice = sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=True
        )
        assert choice.was_random is True
        assert choice.was_informed is False
        assert any(np.array_equal(choice.params, c) for c in candidates)


def test_exploration_epsilon_zero_matches_greedy_and_never_reports_random():
    sampler = _fit_separable_sampler(exploration_epsilon=0.0)
    candidates = _candidate_grid()
    greedy = sampler.sample(
        sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
    )
    for _ in range(10):
        choice = sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=True
        )
        assert choice.was_random is False
        assert np.array_equal(choice.params, greedy.params)


def test_explore_false_never_reports_random_even_with_epsilon_one():
    """explore=False is the test-time sampler: epsilon must not apply at all
    (predicators wires _wrap_sampler_test with no epsilon branch)."""
    sampler = _fit_separable_sampler(exploration_epsilon=1.0)
    candidates = _candidate_grid()
    for _ in range(5):
        choice = sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
        )
        assert choice.was_random is False


def test_same_seed_and_same_data_give_identical_choices():
    candidates = _candidate_grid()
    a = _fit_separable_sampler(seed=3, exploration_epsilon=0.5)
    b = _fit_separable_sampler(seed=3, exploration_epsilon=0.5)
    for _ in range(10):
        choice_a = a.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=True
        )
        choice_b = b.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=True
        )
        assert np.array_equal(choice_a.params, choice_b.params)
        assert choice_a.was_random == choice_b.was_random
        assert choice_a.was_informed == choice_b.was_informed


def test_two_instances_do_not_share_training_data():
    """Guards the pydantic mutable-default trap: observed data must be per-instance."""
    a = _make_sampler()
    b = _make_sampler()
    a.observe(sampler_input=_row(params=np.array([0.9])), success=True)
    assert a.num_observations == 1
    assert b.num_observations == 0


def test_sample_rejects_wrong_candidate_dim():
    """The param_dim guard moved onto `sample` (which still sees raw candidates)
    now that `observe` consumes an already-built row: a candidate whose shape
    disagrees with the classifier's parameter dimensionality is a bug worth
    catching loudly."""
    sampler = _make_sampler(param_dim=2)
    candidates = [np.array([0.5])]  # shape (1,), not (2,)
    try:
        sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
        )
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a candidate vector of the wrong length")


def test_sample_rejects_mismatched_input_and_candidate_counts():
    sampler = _make_sampler()
    candidates = _candidate_grid()
    try:
        sampler.sample(
            sampler_inputs=_rows(candidates=candidates[:-1]), candidates=candidates, explore=False
        )
    except ValueError:
        return
    raise AssertionError("expected a ValueError when input rows and candidates disagree in count")


def test_refit_from_scratch_forgets_nothing_and_tracks_all_data():
    """Each learning cycle refits on *all* data (predicators rebuilds the classifier
    from the full dataset every cycle) -- observing more and refitting must keep the
    earlier examples."""
    sampler = _fit_separable_sampler(num_observations=40)
    assert sampler.num_observations == 40
    sampler.observe(sampler_input=_row(params=np.array([0.99])), success=True)
    sampler.fit()
    assert sampler.num_observations == 41


def test_training_does_not_depend_on_the_ambient_torch_thread_count():
    """`--seed` must fully determine a run, and it did not: `torch.manual_seed` pins
    the initial weights but not the *reduction order* of the matmuls that follow, so
    the same seed and the same data trained to different weights depending on how
    many intra-op threads torch happened to be using.

    That made the thread count a second, unrecorded input to every result. It bit
    this project once already: `scripts/run_sweep.py` pins `OMP_NUM_THREADS=1` on
    every child while a bare `hitl_pmp.cli` run inherits the machine's default, so a
    sweep and a CLI re-run of the same seed were two different experiments (see
    docs/tossing3d-integration-status.md section 5.9).
    """
    x_data, y_data = _thread_probe_data()

    torch.set_num_threads(1)
    single = _fit_probe_classifier(x_data=x_data, y_data=y_data)
    torch.set_num_threads(4)
    multi = _fit_probe_classifier(x_data=x_data, y_data=y_data)

    assert np.array_equal(single, multi)


# ------------------------------------------------------------- linear ablation
#
# EesMethod's sampler_classifier="linear" ablation (see ees_method.py) works by
# passing hid_sizes=() into MlpBinaryClassifier, which _build_net turns into
# logistic regression -- see that method's docstring. These tests exercise that
# claim directly: the hid_sizes=() net really is affine in logit space, and the
# default hid_sizes=(32, 32) net really is not.


def _logit(*, p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)) -- the inverse of the sigmoid MlpBinaryClassifier's output
    layer applies. `predict_proba` is sigmoid(affine(x)), not affine(x) itself, so
    testing the affine identity has to happen in logit space, not probability
    space."""
    return np.log(p / (1.0 - p))


def _nonlinear_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """A 2D XOR-like pattern: not linearly separable, so a fitted logistic-regression
    classifier cannot interpolate it (giving the linear test real, non-degenerate
    weights to check) and a fitted MLP has to lean on its hidden layer's nonlinearity
    to do better than chance (giving the negative test something to detect). 10%
    label noise keeps probabilities away from the 0.0/1.0 float32 saturation trap."""
    rng = np.random.default_rng(0)
    x_data = rng.uniform(-1.0, 1.0, size=(200, 2))
    clean_label = (x_data[:, 0] * x_data[:, 1] > 0).astype(float)
    flip = rng.uniform(size=200) < 0.1
    y_data = np.where(flip, 1.0 - clean_label, clean_label)
    return x_data, y_data


def _affine_probe_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Several (x0, x1) pairs spanning a wide range of the input space. A single
    random pair risks landing inside one ReLU activation region, where a piecewise-
    affine net is locally exactly affine -- probing several pairs is what keeps the
    negative test (MLP is NOT affine) from passing by accident."""
    return [
        (np.array([-0.9, -0.9]), np.array([0.9, 0.9])),
        (np.array([-0.9, 0.9]), np.array([0.9, -0.9])),
        (np.array([-0.5, 0.2]), np.array([0.6, -0.3])),
        (np.array([0.1, 0.1]), np.array([-0.7, 0.8])),
        (np.array([-0.2, -0.6]), np.array([0.4, 0.5])),
        (np.array([0.8, -0.1]), np.array([-0.6, -0.8])),
    ]


def test_hid_sizes_empty_is_affine_in_logit_space():
    """`hid_sizes=()` (the linear ablation) must be exactly logistic regression:
    predict_proba is sigmoid(affine(x)), which is affine when read back through
    logit. Few training iterations plus mildly-noisy, non-separable-to-perfection
    data keep probabilities interior -- a probability saturated to exactly 0.0/1.0 in
    float32 would make logit +/-inf and the identity vacuous."""
    x_data, y_data = _nonlinear_classification_data()
    classifier = MlpBinaryClassifier(seed=0, hid_sizes=(), max_train_iters=20)
    classifier.fit(x_data=x_data, y_data=y_data)
    for x0, x1 in _affine_probe_pairs():
        xm = (x0 + x1) / 2.0
        p0, p1, pm = classifier.predict_proba(x_data=np.stack([x0, x1, xm]))
        assert 0.0 < p0 < 1.0
        assert 0.0 < p1 < 1.0
        assert 0.0 < pm < 1.0, "probabilities saturated to 0/1; logit is +/-inf and vacuous"
        logit0, logit1, logitm = _logit(p=np.array([p0, p1, pm]))
        assert logitm == pytest.approx((logit0 + logit1) / 2.0, abs=1e-3)


def test_hid_sizes_32_32_is_not_affine_in_logit_space():
    """The default MLP (hid_sizes=(32, 32)) must measurably fail the same identity
    on at least one of several probe pairs -- a same-shaped test that never fails
    would not be evidence the ablation changes anything. Multiple pairs guard against
    the single-region trap described in `_affine_probe_pairs`."""
    x_data, y_data = _nonlinear_classification_data()
    classifier = MlpBinaryClassifier(seed=0, hid_sizes=(32, 32), max_train_iters=300)
    classifier.fit(x_data=x_data, y_data=y_data)
    deviations = []
    for x0, x1 in _affine_probe_pairs():
        xm = (x0 + x1) / 2.0
        p0, p1, pm = classifier.predict_proba(x_data=np.stack([x0, x1, xm]))
        if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0 and 0.0 < pm < 1.0):
            continue  # saturated at this probe pair; skip rather than divide by zero
        logit0, logit1, logitm = _logit(p=np.array([p0, p1, pm]))
        deviations.append(abs(logitm - (logit0 + logit1) / 2.0))
    assert deviations, "every probe pair saturated; cannot test nonlinearity here"
    assert max(deviations) > 1e-2, (
        f"max deviation {max(deviations):.6f} across {len(deviations)} probes is too "
        "small to trust as evidence of nonlinearity, not noise"
    )


def test_linear_sampler_plugs_into_sample_unchanged():
    """A LearnedSkillSampler built with hid_sizes=() -- the linear ablation -- goes
    through the exact same sample() call as the default MLP configuration and
    returns a SamplerChoice of the same shape, with the same was_random/was_informed
    semantics; no other code path is taken. Mirrors
    test_learns_a_separable_task_and_greedily_picks_the_good_side, the MLP version of
    this same check."""
    # A purely linear model converges more slowly than the 32x32 MLP does on this
    # task at TEST_MAX_TRAIN_ITERS=300 (measured: still the wrong sign at 300-2000
    # full-batch Adam steps, correct by 5000), so this arm gets its own, larger
    # budget rather than sharing the MLP tests' cheap default.
    sampler = _fit_separable_sampler(hid_sizes=(), max_train_iters=5000)
    rng = np.random.default_rng(7)
    num_trials = 20
    num_good = 0
    for _ in range(num_trials):
        candidates = [np.array([v]) for v in rng.uniform(0.0, 1.0, size=10)]
        choice = sampler.sample(
            sampler_inputs=_rows(candidates=candidates), candidates=candidates, explore=False
        )
        assert isinstance(choice, SamplerChoice)
        assert isinstance(choice.was_random, bool)
        assert isinstance(choice.was_informed, bool)
        if choice.params[0] > 0.5:
            num_good += 1
    assert num_good >= 18, f"only {num_good}/{num_trials} greedy picks were on the good side"
