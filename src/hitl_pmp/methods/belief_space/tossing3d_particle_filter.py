"""Vectorized Liu-West inference for particle-filter beliefs."""

from math import lgamma, log, pi

import numpy as np

from .types.particle_filter_belief import ParticleFilterBelief
from .types.skill_belief import COMPETENCE_MAX, COMPETENCE_MIN, LEARNING_RATE_MAX, LEARNING_RATE_MIN

_OBSERVATION_SCALE = 0.02
_OBSERVATION_DOF = 4.0
_ESS_FRACTION = 0.5
_SHRINKAGE = 0.98
_STREAM_TAG = 0x524553414D504C45


def condition_outcome(*, belief: ParticleFilterBelief, success: bool) -> ParticleFilterBelief:
    parameters, weights = belief.arrays()
    likelihoods = parameters[:, 0] if success else 1.0 - parameters[:, 0]
    masses = weights * np.maximum(likelihoods, np.finfo(np.float64).tiny)
    return _condition(belief=belief, parameters=parameters, masses=masses)


def condition_learning_rate(
    *, belief: ParticleFilterBelief, observed_learning_rate: float
) -> ParticleFilterBelief:
    assert LEARNING_RATE_MIN <= observed_learning_rate <= LEARNING_RATE_MAX
    dof, scale = _OBSERVATION_DOF, _OBSERVATION_SCALE
    normalizer = lgamma((dof + 1.0) / 2.0) - lgamma(dof / 2.0) - 0.5 * log(dof * pi) - log(scale)
    parameters, weights = belief.arrays()
    residuals = (observed_learning_rate - parameters[:, 1]) / scale
    log_masses = np.log(weights) + normalizer - (dof + 1.0) / 2.0 * np.log1p(residuals**2 / dof)
    masses = np.exp(log_masses - float(np.max(log_masses)))
    return _condition(belief=belief, parameters=parameters, masses=masses)


def _condition(
    *, belief: ParticleFilterBelief, parameters: np.ndarray, masses: np.ndarray
) -> ParticleFilterBelief:
    normalizer = float(np.sum(masses))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("observation has zero probability")
    weights = masses / normalizer
    if 1.0 / float(np.dot(weights, weights)) >= len(weights) * _ESS_FRACTION:
        return belief.from_arrays(parameters=parameters, weights=weights)
    return _resample(belief=belief, parameters=parameters, weights=weights)


def _resample(
    *, belief: ParticleFilterBelief, parameters: np.ndarray, weights: np.ndarray
) -> ParticleFilterBelief:
    count = len(weights)
    rng = np.random.default_rng(
        np.random.SeedSequence([belief.resampling_seed, belief.resampling_count, _STREAM_TAG])
    )
    positions = (rng.random() + np.arange(count)) / count
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    mean = np.average(parameters, axis=0, weights=weights)
    centered = parameters - mean
    covariance = (centered * weights[:, np.newaxis]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    selected = parameters[indices]
    noise = rng.normal(size=selected.shape) @ root.T
    rejuvenated = (
        _SHRINKAGE * selected + (1.0 - _SHRINKAGE) * mean + np.sqrt(1.0 - _SHRINKAGE**2) * noise
    )
    rejuvenated[:, 0] = np.clip(rejuvenated[:, 0], COMPETENCE_MIN, COMPETENCE_MAX)
    rejuvenated[:, 1] = np.clip(rejuvenated[:, 1], LEARNING_RATE_MIN, LEARNING_RATE_MAX)
    result = belief.from_arrays(parameters=rejuvenated, weights=np.full(count, 1.0 / count))
    return result.model_copy(update={"resampling_count": belief.resampling_count + 1})
