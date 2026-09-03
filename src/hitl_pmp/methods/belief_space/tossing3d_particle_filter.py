"""Sequential Monte Carlo inference for Tossing3D skill parameters."""

from threading import Lock

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from stonesoup.resampler.particle import ESSResampler, SystematicResampler
from stonesoup.types.array import StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState

from .types.skill_belief import SkillBelief, SkillHypothesis, WeightedHypothesis

_STONE_SOUP_RANDOM_LOCK = Lock()


class ParticleFilterDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_particles: int = Field(ge=1)
    effective_sample_size: float = Field(gt=0.0)
    resampling_count: int = Field(ge=0)


def make_particle_belief_prior(*, num_particles: int, seed: int) -> SkillBelief:
    """Sample a continuous uniform prior over competence and learning rate."""
    assert num_particles >= 1
    rng = np.random.default_rng(seed)
    parameters = rng.uniform((0.0, 0.0), (1.0, 0.1), size=(num_particles, 2))
    probability = 1.0 / num_particles
    return SkillBelief(
        estimator="particle_filter",
        resampling_seed=seed,
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(
                    competence=float(competence), learning_rate=float(learning_rate)
                ),
                probability=probability,
            )
            for competence, learning_rate in parameters
        ),
    )


def effective_sample_size(*, belief: SkillBelief) -> float:
    return 1.0 / sum(item.probability**2 for item in belief.hypotheses)


def particle_filter_diagnostics(*, belief: SkillBelief) -> ParticleFilterDiagnostics:
    return ParticleFilterDiagnostics(
        num_particles=len(belief.hypotheses),
        effective_sample_size=effective_sample_size(belief=belief),
        resampling_count=belief.resampling_count,
    )


def condition_particle_belief(*, belief: SkillBelief, success: bool) -> SkillBelief:
    """Apply the Bernoulli likelihood and ESS-triggered systematic resampling."""
    assert belief.estimator == "particle_filter"
    masses = np.asarray(
        [
            item.probability
            * (item.hypothesis.competence if success else 1.0 - item.hypothesis.competence)
            for item in belief.hypotheses
        ],
        dtype=np.float64,
    )
    normalizer = float(np.sum(masses))
    if normalizer <= 0.0:
        raise ValueError(f"observation success={success} has zero probability")
    weights = masses / normalizer
    weighted = belief.model_copy(
        update={
            "hypotheses": tuple(
                item.model_copy(update={"probability": float(probability)})
                for item, probability in zip(belief.hypotheses, weights, strict=True)
            )
        }
    )
    if effective_sample_size(belief=weighted) >= len(weighted.hypotheses) / 2:
        return weighted
    return _resample(belief=weighted)


def _resample(*, belief: SkillBelief) -> SkillBelief:
    vectors = StateVectors(
        np.asarray(
            [
                [item.hypothesis.competence for item in belief.hypotheses],
                [item.hypothesis.learning_rate for item in belief.hypotheses],
            ],
            dtype=np.float64,
        )
    )
    particles = ParticleState(
        state_vector=vectors,
        weight=[Probability(item.probability) for item in belief.hypotheses],
    )
    resampler = ESSResampler(
        threshold=len(belief.hypotheses) / 2,
        resampler=SystematicResampler(),
    )
    # Stone Soup's systematic resampler currently uses NumPy's legacy global RNG.
    # Preserve application state while deriving a deterministic draw from the belief.
    with _STONE_SOUP_RANDOM_LOCK:
        random_state = np.random.get_state()
        np.random.seed((belief.resampling_seed + belief.resampling_count) % 2**32)
        try:
            resampled = resampler.resample(particles)
        finally:
            np.random.set_state(random_state)
    probability = 1.0 / len(resampled)
    return belief.model_copy(
        update={
            "hypotheses": tuple(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(
                        competence=float(resampled.state_vector[0, index]),
                        learning_rate=float(resampled.state_vector[1, index]),
                    ),
                    probability=probability,
                )
                for index in range(len(resampled))
            ),
            "resampling_count": belief.resampling_count + 1,
        }
    )
