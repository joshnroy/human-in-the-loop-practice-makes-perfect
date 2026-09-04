"""Packed, continuous particle-filter skill belief."""

from typing import Final

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from .skill_belief import (
    COMPETENCE_MAX,
    COMPETENCE_MIN,
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
    SkillBelief,
)

PARTICLE_DTYPE: Final = np.dtype("<f8")


class ParticleFilterBelief(SkillBelief):
    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")
    resampling_count: int = Field(default=0, ge=0)
    resampling_seed: int = Field(default=0, ge=0)
    particle_parameters: bytes
    particle_weights: bytes
    num_particles: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_particles(self) -> "ParticleFilterBelief":
        parameters, weights = self.arrays()
        if parameters.shape != (self.num_particles, 2) or weights.shape != (self.num_particles,):
            raise ValueError("particle buffers do not match num_particles")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("particle parameters must be finite")
        if not np.all((parameters[:, 0] >= COMPETENCE_MIN) & (parameters[:, 0] <= COMPETENCE_MAX)):
            raise ValueError("particle competences are outside their bounds")
        if not np.all(
            (parameters[:, 1] >= LEARNING_RATE_MIN) & (parameters[:, 1] <= LEARNING_RATE_MAX)
        ):
            raise ValueError("particle learning rates are outside their bounds")
        if (
            not np.all(np.isfinite(weights))
            or not np.all(weights > 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("particle weights must be positive, finite, and sum to one")
        return self

    @classmethod
    def broad_prior(cls, *, num_particles: int, seed: int) -> "ParticleFilterBelief":
        assert num_particles >= 1
        rng = np.random.default_rng(seed)
        quantiles = (np.arange(num_particles) + 0.5) / num_particles
        competences = rng.permutation(quantiles)
        learning_rates = rng.permutation(quantiles)
        parameters = np.column_stack((
            COMPETENCE_MIN + competences * (COMPETENCE_MAX - COMPETENCE_MIN),
            LEARNING_RATE_MIN + learning_rates * (LEARNING_RATE_MAX - LEARNING_RATE_MIN),
        ))
        weights = np.full(num_particles, 1.0 / num_particles)
        return cls(
            resampling_seed=seed,
            particle_parameters=np.asarray(parameters, dtype=PARTICLE_DTYPE).tobytes(),
            particle_weights=np.asarray(weights, dtype=PARTICLE_DTYPE).tobytes(),
            num_particles=num_particles,
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.frombuffer(self.particle_parameters, dtype=PARTICLE_DTYPE).reshape(
            -1, 2
        ), np.frombuffer(self.particle_weights, dtype=PARTICLE_DTYPE)

    def mean_competence(self) -> float:
        parameters, weights = self.arrays()
        return float(weights @ parameters[:, 0])

    def mean_learning_rate(self) -> float:
        parameters, weights = self.arrays()
        return float(weights @ parameters[:, 1])

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        parameters, weights = self.arrays()
        return parameters[np.atleast_1d(rng.choice(self.num_particles, size=count, p=weights))]

    def condition_outcome(self, *, success: bool) -> "ParticleFilterBelief":
        from hitl_pmp.methods.belief_space.tossing3d_particle_filter import condition_outcome

        return condition_outcome(belief=self, success=success)

    def condition_learning_rate(self, *, observed_learning_rate: float) -> "ParticleFilterBelief":
        from hitl_pmp.methods.belief_space.tossing3d_particle_filter import (
            condition_learning_rate,
        )

        return condition_learning_rate(belief=self, observed_learning_rate=observed_learning_rate)

    def refit(self, *, training_examples: int) -> "ParticleFilterBelief":
        assert training_examples >= 0
        if training_examples == 0:
            return self
        parameters, weights = self.arrays()
        projected = parameters.copy()
        projected[:, 0] = np.clip(
            projected[:, 0] + projected[:, 1] * training_examples, COMPETENCE_MIN, COMPETENCE_MAX
        )
        return self.from_arrays(parameters=projected, weights=weights)

    def diagnostics(self) -> dict[str, object]:
        _, weights = self.arrays()
        return {
            "representation": "particle_filter",
            "num_particles": self.num_particles,
            "effective_sample_size": float(1.0 / np.dot(weights, weights)),
            "resampling_count": self.resampling_count,
        }

    def signature(self) -> tuple[object, ...]:
        return (
            self.resampling_count,
            self.resampling_seed,
            self.particle_parameters,
            self.particle_weights,
        )

    def from_arrays(self, *, parameters: np.ndarray, weights: np.ndarray) -> "ParticleFilterBelief":
        return self.model_copy(
            update={
                "particle_parameters": np.ascontiguousarray(
                    parameters, dtype=PARTICLE_DTYPE
                ).tobytes(),
                "particle_weights": np.ascontiguousarray(weights, dtype=PARTICLE_DTYPE).tobytes(),
                "num_particles": len(weights),
            }
        )
