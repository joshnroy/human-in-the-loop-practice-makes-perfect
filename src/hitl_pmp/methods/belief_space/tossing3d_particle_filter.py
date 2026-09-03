"""Vectorized Liu-West particle inference for Tossing3D skill parameters."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .types.skill_belief import PARTICLE_DTYPE, PARTICLE_STORAGE_VERSION, SkillBelief

_RESAMPLING_ESS_FRACTION = 0.5
_LIU_WEST_SHRINKAGE = 0.98
_RESAMPLING_STREAM_TAG = 0x524553414D504C45


class ParticleFilterDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_particles: int = Field(ge=1)
    effective_sample_size: float = Field(gt=0.0)
    resampling_count: int = Field(ge=0)


def make_particle_belief_prior(*, num_particles: int, seed: int) -> SkillBelief:
    """Sample a continuous uniform prior over competence and learning rate."""
    assert num_particles >= 1
    rng = np.random.default_rng(seed)
    parameters = rng.uniform((0.0, 0.0), (1.0, 0.1), size=(num_particles, 2)).astype(
        PARTICLE_DTYPE, copy=False
    )
    weights = np.full(num_particles, 1.0 / num_particles, dtype=PARTICLE_DTYPE)
    return SkillBelief(
        estimator="particle_filter",
        resampling_seed=seed,
        particle_parameters=parameters.tobytes(),
        particle_weights=weights.tobytes(),
        num_particles=num_particles,
        particle_storage_version=PARTICLE_STORAGE_VERSION,
    )


def effective_sample_size(*, belief: SkillBelief) -> float:
    _, weights = belief_arrays(belief=belief)
    return float(1.0 / np.dot(weights, weights))


def particle_filter_diagnostics(*, belief: SkillBelief) -> ParticleFilterDiagnostics:
    return ParticleFilterDiagnostics(
        num_particles=belief.num_particles,
        effective_sample_size=effective_sample_size(belief=belief),
        resampling_count=belief.resampling_count,
    )


def condition_particle_belief(*, belief: SkillBelief, success: bool) -> SkillBelief:
    """Apply the Bernoulli likelihood and ESS-triggered systematic resampling."""
    assert belief.estimator == "particle_filter"
    parameters, weights = belief_arrays(belief=belief)
    likelihoods = parameters[:, 0] if success else 1.0 - parameters[:, 0]
    likelihoods = np.maximum(likelihoods, np.finfo(np.float64).tiny)
    return condition_and_maybe_resample(
        belief=belief, parameters=parameters, masses=weights * likelihoods
    )


def belief_arrays(*, belief: SkillBelief) -> tuple[np.ndarray, np.ndarray]:
    assert belief.estimator == "particle_filter"
    parameters = np.frombuffer(belief.particle_parameters, dtype=PARTICLE_DTYPE).reshape(-1, 2)
    weights = np.frombuffer(belief.particle_weights, dtype=PARTICLE_DTYPE)
    return parameters, weights


def condition_and_maybe_resample(
    *, belief: SkillBelief, parameters: np.ndarray, masses: np.ndarray
) -> SkillBelief:
    normalizer = float(np.sum(masses))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("observation has zero probability")
    weights = masses / normalizer
    if 1.0 / float(np.dot(weights, weights)) >= len(weights) * _RESAMPLING_ESS_FRACTION:
        return belief_from_arrays(belief=belief, parameters=parameters, weights=weights)
    return resample(belief=belief, parameters=parameters, weights=weights)


def belief_from_arrays(
    *, belief: SkillBelief, parameters: np.ndarray, weights: np.ndarray
) -> SkillBelief:
    return belief.model_copy(
        update={
            "particle_parameters": np.ascontiguousarray(parameters, dtype=PARTICLE_DTYPE).tobytes(),
            "particle_weights": np.ascontiguousarray(weights, dtype=PARTICLE_DTYPE).tobytes(),
            "num_particles": len(weights),
            "particle_storage_version": PARTICLE_STORAGE_VERSION,
        }
    )


def resample(*, belief: SkillBelief, parameters: np.ndarray, weights: np.ndarray) -> SkillBelief:
    """Systematically resample and Liu-West rejuvenate the continuous parameters."""
    num_particles = len(weights)
    rng = np.random.default_rng(
        np.random.SeedSequence([
            belief.resampling_seed,
            belief.resampling_count,
            _RESAMPLING_STREAM_TAG,
        ])
    )
    positions = (rng.random() + np.arange(num_particles)) / num_particles
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")

    mean = np.average(parameters, axis=0, weights=weights)
    centered = parameters - mean
    covariance = (centered * weights[:, np.newaxis]).T @ centered
    shrinkage = _LIU_WEST_SHRINKAGE
    bandwidth = np.sqrt(1.0 - shrinkage**2)
    selected = parameters[indices]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    noise = rng.normal(size=selected.shape) @ covariance_root.T
    rejuvenated = shrinkage * selected + (1.0 - shrinkage) * mean + bandwidth * noise
    rejuvenated[:, 0] = np.clip(rejuvenated[:, 0], 0.0, 1.0)
    rejuvenated[:, 1] = np.clip(rejuvenated[:, 1], 0.0, 1.0)
    resampled = belief_from_arrays(
        belief=belief,
        parameters=rejuvenated,
        weights=np.full(num_particles, 1.0 / num_particles),
    )
    return resampled.model_copy(
        update={
            "resampling_count": belief.resampling_count + 1,
        }
    )
