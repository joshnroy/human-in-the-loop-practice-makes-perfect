"""The margin-seeking eval-time choices pick the interior of a success region, not its
edge, from success/failure labels alone."""

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect.robust_choice import (
    LaplaceGpClassifier,
    NoiseSmoothedScore,
)


def _plateau(rows: np.ndarray) -> np.ndarray:  # noqa: PLR0917 -- a score_rows callable
    """A saturated classifier: exactly 1.0 on [0.2, 0.8], exactly 0.0 elsewhere."""
    x = rows[:, 1]
    return ((x >= 0.2) & (x <= 0.8)).astype(np.float64)


def test_a_saturated_plateau_ties_every_candidate_under_plain_scores() -> None:
    rows = np.array([[1.0, x] for x in np.linspace(0.0, 1.0, 101)])
    scores = _plateau(rows)
    assert np.sum(scores == scores.max()) == 61


def test_noise_smoothing_puts_the_best_score_at_the_plateau_centre() -> None:
    rows = np.array([[1.0, x] for x in np.linspace(0.0, 1.0, 101)])
    smoother = NoiseSmoothedScore(noise_std=(0.0, 0.1), num_draws=4000, seed=0)
    smoothed = smoother.smoothed_scores(rows=rows, score_rows=_plateau)
    best = rows[np.flatnonzero(smoothed == smoothed.max()), 1]
    assert np.all(np.abs(best - 0.5) <= 0.05)
    # The edge now scores strictly below the centre: about 1/2 at a hard boundary.
    assert smoothed[20] < 0.6 < smoothed[50]


def test_noise_smoothing_leaves_columns_with_zero_noise_untouched() -> None:
    seen: list[np.ndarray] = []

    def record(rows: np.ndarray) -> np.ndarray:  # noqa: PLR0917 -- a score_rows callable
        seen.append(rows.copy())
        return np.zeros(len(rows))

    rows = np.array([[1.0, 0.3], [1.0, 0.7]])
    NoiseSmoothedScore(noise_std=(0.0, 0.1), num_draws=8, seed=0).smoothed_scores(
        rows=rows, score_rows=record
    )
    stacked = np.concatenate(seen)
    assert np.all(stacked[:, 0] == 1.0)
    assert np.any(stacked[:, 1] != 0.3)


def test_noise_std_must_match_the_row_width() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        NoiseSmoothedScore(noise_std=(0.1,), num_draws=8, seed=0).smoothed_scores(
            rows=np.zeros((3, 2)), score_rows=_plateau
        )


def test_gp_classifier_peaks_inside_the_positive_cluster_and_does_not_saturate() -> None:
    # Failures on both sides, successes in the middle, and two mixed labels at the
    # near edge: the shape of today's repeated-command data.
    x = np.array([[0.05], [0.15], [0.3], [0.3], [0.45], [0.5], [0.55], [0.85], [0.95]])
    y = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0])
    gp = LaplaceGpClassifier(lower=(0.0,), upper=(1.0,))
    gp.fit(x_data=x, y_data=y)
    grid = np.linspace(0.0, 1.0, 101)[:, None]
    mean = gp.predict_proba(x_data=grid)
    assert 0.4 <= grid[int(np.argmax(mean)), 0] <= 0.65
    assert mean.max() < 0.99
    assert mean[30] < mean[50]


def test_gp_lower_bound_is_below_its_mean_and_prefers_data_over_empty_space() -> None:
    x = np.array([[0.4], [0.45], [0.5], [0.55], [0.9]])
    y = np.array([1, 1, 1, 1, 0])
    gp = LaplaceGpClassifier(lower=(0.0,), upper=(1.0,))
    gp.fit(x_data=x, y_data=y)
    grid = np.linspace(0.0, 1.0, 101)[:, None]
    lcb = gp.lower_confidence_bound(x_data=grid, kappa=1.0)
    assert np.all(lcb <= gp.predict_proba(x_data=grid) + 1e-12)
    assert 0.3 <= grid[int(np.argmax(lcb)), 0] <= 0.65


def test_gp_classifier_with_one_class_is_flat_at_the_prior() -> None:
    gp = LaplaceGpClassifier(lower=(0.0,), upper=(1.0,))
    gp.fit(x_data=np.array([[0.2], [0.4]]), y_data=np.array([0, 0]))
    mean = gp.predict_proba(x_data=np.linspace(0, 1, 11)[:, None])
    assert np.all(mean < 0.5)
