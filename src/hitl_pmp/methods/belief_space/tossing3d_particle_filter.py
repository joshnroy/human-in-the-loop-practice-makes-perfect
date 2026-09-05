"""Vectorized Liu-West inference for particle-filter beliefs."""

from __future__ import annotations

from math import lgamma, log, pi
from typing import Protocol, TypeVar

import numpy as np
from typing_extensions import Self

from .types.skill_belief import (
    COMPETENCE_MAX,
    COMPETENCE_MIN,
    COST_MAX,
    COST_MIN,
    COST_OBSERVATION_SCALE,
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
)


class ParticleBelief(Protocol):
    resampling_seed: int
    resampling_count: int

    def arrays(self) -> tuple[np.ndarray, np.ndarray]: ...
    def from_arrays(self, *, parameters: np.ndarray, weights: np.ndarray) -> Self: ...
    def model_copy(self, *, update: dict[str, object]) -> Self: ...


BeliefT = TypeVar("BeliefT", bound=ParticleBelief)

_OBSERVATION_SCALE = 0.02
_OBSERVATION_DOF = 4.0
_ESS_FRACTION = 0.5
_SHRINKAGE = 0.98
_STREAM_TAG = 0x524553414D504C45


def condition_outcome(*, belief: BeliefT, success: bool) -> BeliefT:
    parameters, weights = belief.arrays()
    likelihoods = parameters[:, 0] if success else 1.0 - parameters[:, 0]
    masses = weights * np.maximum(likelihoods, np.finfo(np.float64).tiny)
    return _condition(belief=belief, parameters=parameters, masses=masses)


def condition_execution(*, belief: BeliefT, success: bool, observed_cost: float) -> BeliefT:
    """Condition competence and cost on the same completed execution."""
    assert observed_cost >= 0.0
    parameters, weights = belief.arrays()
    outcome_likelihoods = parameters[:, 0] if success else 1.0 - parameters[:, 0]
    cost_log_likelihoods = _student_t_log_likelihoods(
        observation=observed_cost,
        hypotheses=parameters[:, 2],
        scale=COST_OBSERVATION_SCALE,
    )
    log_masses = (
        np.log(weights)
        + np.log(np.maximum(outcome_likelihoods, np.finfo(np.float64).tiny))
        + cost_log_likelihoods
    )
    masses = np.exp(log_masses - float(np.max(log_masses)))
    return _condition(belief=belief, parameters=parameters, masses=masses)


def condition_cost(*, belief: BeliefT, observed_cost: float) -> BeliefT:
    assert observed_cost >= 0.0
    parameters, weights = belief.arrays()
    log_masses = np.log(weights) + _student_t_log_likelihoods(
        observation=observed_cost,
        hypotheses=parameters[:, 2],
        scale=COST_OBSERVATION_SCALE,
    )
    masses = np.exp(log_masses - float(np.max(log_masses)))
    return _condition(belief=belief, parameters=parameters, masses=masses)


def condition_learning_rate(*, belief: BeliefT, observed_learning_rate: float) -> BeliefT:
    assert LEARNING_RATE_MIN <= observed_learning_rate <= LEARNING_RATE_MAX
    parameters, weights = belief.arrays()
    log_masses = np.log(weights) + _student_t_log_likelihoods(
        observation=observed_learning_rate,
        hypotheses=parameters[:, 1],
        scale=_OBSERVATION_SCALE,
    )
    masses = np.exp(log_masses - float(np.max(log_masses)))
    return _condition(belief=belief, parameters=parameters, masses=masses)


def _condition(*, belief: BeliefT, parameters: np.ndarray, masses: np.ndarray) -> BeliefT:
    normalizer = float(np.sum(masses))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("observation has zero probability")
    weights = masses / normalizer
    if 1.0 / float(np.dot(weights, weights)) >= len(weights) * _ESS_FRACTION:
        return belief.from_arrays(parameters=parameters, weights=weights)
    return _resample(belief=belief, parameters=parameters, weights=weights)


def _resample(*, belief: BeliefT, parameters: np.ndarray, weights: np.ndarray) -> BeliefT:
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
    rejuvenated[:, 2] = np.clip(rejuvenated[:, 2], COST_MIN, COST_MAX)
    result = belief.from_arrays(parameters=rejuvenated, weights=np.full(count, 1.0 / count))
    return result.model_copy(update={"resampling_count": belief.resampling_count + 1})


def _student_t_log_likelihoods(
    *, observation: float, hypotheses: np.ndarray, scale: float
) -> np.ndarray:
    dof = _OBSERVATION_DOF
    normalizer = lgamma((dof + 1.0) / 2.0) - lgamma(dof / 2.0) - 0.5 * log(dof * pi) - log(scale)
    residuals = (observation - hypotheses) / scale
    return normalizer - (dof + 1.0) / 2.0 * np.log1p(residuals**2 / dof)
