"""Packed, continuous particle-filter skill belief."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from hitl_pmp.methods.belief_space.tossing3d_particle_filter import (
    ExecutionObservation,
    condition_cost,
    condition_cycle_evidence,
    condition_execution,
    condition_executions,
    condition_learning_rate,
    condition_outcome,
    cycle_learning_rate_observation,
    make_rng,
    reflect_into_interval,
)

from .skill_belief import (
    COMPETENCE_MAX,
    COMPETENCE_MIN,
    COST_MAX,
    COST_MIN,
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
    SkillBelief,
)

PARTICLE_DTYPE: Final = np.dtype("<f8")


class ParticleFilterBelief(SkillBelief):
    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")
    resampling_count: int = Field(default=0, ge=0)
    process_transition_count: int = Field(default=0, ge=0)
    resampling_seed: int = Field(default=0, ge=0)
    particle_parameters: bytes
    particle_weights: bytes
    num_particles: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_particles(self) -> Self:
        parameters, weights = self.arrays()
        if parameters.shape != (self.num_particles, 3) or weights.shape != (self.num_particles,):
            raise ValueError("particle buffers do not match num_particles")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("particle parameters must be finite")
        if not np.all((parameters[:, 0] >= COMPETENCE_MIN) & (parameters[:, 0] <= COMPETENCE_MAX)):
            raise ValueError("particle competences are outside their bounds")
        if not np.all(
            (parameters[:, 1] >= LEARNING_RATE_MIN) & (parameters[:, 1] <= LEARNING_RATE_MAX)
        ):
            raise ValueError("particle learning rates are outside their bounds")
        if not np.all((parameters[:, 2] >= COST_MIN) & (parameters[:, 2] <= COST_MAX)):
            raise ValueError("particle costs are outside their bounds")
        if (
            not np.all(np.isfinite(weights))
            or not np.all(weights > 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("particle weights must be positive, finite, and sum to one")
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.frombuffer(self.particle_parameters, dtype=PARTICLE_DTYPE).reshape(
            -1, 3
        ), np.frombuffer(self.particle_weights, dtype=PARTICLE_DTYPE)

    def mean_competence(self) -> float:
        parameters, weights = self.arrays()
        return float(weights @ parameters[:, 0])

    def mean_learning_rate(self) -> float:
        parameters, weights = self.arrays()
        return float(weights @ parameters[:, 1])

    def mean_cost(self) -> float:
        parameters, weights = self.arrays()
        return float(weights @ parameters[:, 2])

    def sample(self, *, rng: np.random.Generator, count: int) -> np.ndarray:
        parameters, weights = self.arrays()
        return parameters[np.atleast_1d(rng.choice(self.num_particles, size=count, p=weights))]

    def condition_outcome(self, *, success: bool) -> Self:
        return condition_outcome(belief=self, success=success)

    def condition_execution(self, *, success: bool, observed_cost: float) -> Self:
        return condition_execution(belief=self, success=success, observed_cost=observed_cost)

    def condition_executions(self, *, observations: Sequence[ExecutionObservation]) -> Self:
        return condition_executions(belief=self, observations=observations)

    def condition_cycle_evidence(
        self,
        *,
        observations: Sequence[ExecutionObservation],
        competence_before: float,
        training_examples: int,
    ) -> tuple[Self, float | None]:
        return condition_cycle_evidence(
            belief=self,
            observations=observations,
            competence_before=competence_before,
            training_examples=training_examples,
        )

    def cycle_learning_rate_observation(
        self,
        *,
        observations: Sequence[ExecutionObservation],
        competence_before: float,
        training_examples: int,
    ) -> float | None:
        return cycle_learning_rate_observation(
            belief=self,
            observations=observations,
            competence_before=competence_before,
            training_examples=training_examples,
        )

    def condition_cost(self, *, observed_cost: float) -> Self:
        return condition_cost(belief=self, observed_cost=observed_cost)

    def condition_learning_rate(self, *, observed_learning_rate: float) -> Self:
        return condition_learning_rate(belief=self, observed_learning_rate=observed_learning_rate)

    def refit(self, *, training_examples: int) -> Self:
        assert training_examples >= 0
        if training_examples == 0:
            return self
        parameters, weights = self.arrays()
        projected = parameters.copy()
        projected[:, 0] = np.clip(
            projected[:, 0] + projected[:, 1] * training_examples, COMPETENCE_MIN, COMPETENCE_MAX
        )
        return self.from_arrays(parameters=projected, weights=weights)

    def advance_learning_rate(self, *, process_noise_std: float) -> Self:
        """Apply one Gaussian random-walk transition to the latent learning rate."""
        assert process_noise_std >= 0.0
        if process_noise_std == 0.0:
            return self
        parameters, weights = self.arrays()
        rng = make_rng(
            seed=self.resampling_seed,
            stream=self.process_transition_count,
            tag=0x455441,
        )
        transitioned = parameters.copy()
        proposals = transitioned[:, 1] + rng.normal(0.0, process_noise_std, size=self.num_particles)
        transitioned[:, 1] = reflect_into_interval(
            values=proposals,
            lower=LEARNING_RATE_MIN,
            upper=LEARNING_RATE_MAX,
        )
        return self.from_arrays(parameters=transitioned, weights=weights).model_copy(
            update={"process_transition_count": self.process_transition_count + 1}
        )

    def diagnostics(self) -> dict[str, object]:
        _, weights = self.arrays()
        return {
            "representation": "particle_filter",
            "num_particles": self.num_particles,
            "effective_sample_size": float(1.0 / np.dot(weights, weights)),
            "resampling_count": self.resampling_count,
            "process_transition_count": self.process_transition_count,
        }

    def signature(self) -> tuple[object, ...]:
        return (
            self.resampling_count,
            self.process_transition_count,
            self.resampling_seed,
            self.particle_parameters,
            self.particle_weights,
        )

    def from_arrays(self, *, parameters: np.ndarray, weights: np.ndarray) -> Self:
        return self.model_copy(
            update={
                "particle_parameters": np.ascontiguousarray(
                    parameters, dtype=PARTICLE_DTYPE
                ).tobytes(),
                "particle_weights": np.ascontiguousarray(weights, dtype=PARTICLE_DTYPE).tobytes(),
                "num_particles": len(weights),
            }
        )


def create_broad_particle_prior(*, num_particles: int, seed: int) -> ParticleFilterBelief:
    assert num_particles >= 1
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(num_particles) + 0.5) / num_particles
    parameters = np.column_stack((
        COMPETENCE_MIN + rng.permutation(quantiles) * (COMPETENCE_MAX - COMPETENCE_MIN),
        LEARNING_RATE_MIN + rng.permutation(quantiles) * (LEARNING_RATE_MAX - LEARNING_RATE_MIN),
        COST_MIN + rng.permutation(quantiles) * (COST_MAX - COST_MIN),
    ))
    return ParticleFilterBelief(
        resampling_seed=seed,
        particle_parameters=np.asarray(parameters, dtype=PARTICLE_DTYPE).tobytes(),
        particle_weights=np.full(
            num_particles, 1.0 / num_particles, dtype=PARTICLE_DTYPE
        ).tobytes(),
        num_particles=num_particles,
    )


def create_fixed_performance_cost_prior(
    *,
    num_particles: int,
    seed: int,
    competence: float,
    learning_rate: float,
) -> ParticleFilterBelief:
    """Keep known performance fixed while inferring an execution cost."""
    assert COMPETENCE_MIN <= competence <= COMPETENCE_MAX
    assert LEARNING_RATE_MIN <= learning_rate <= LEARNING_RATE_MAX
    assert num_particles >= 1
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(num_particles) + 0.5) / num_particles
    parameters = np.column_stack((
        np.full(num_particles, competence),
        np.full(num_particles, learning_rate),
        COST_MIN + rng.permutation(quantiles) * (COST_MAX - COST_MIN),
    ))
    return ParticleFilterBelief(
        resampling_seed=seed,
        particle_parameters=np.asarray(parameters, dtype=PARTICLE_DTYPE).tobytes(),
        particle_weights=np.full(
            num_particles, 1.0 / num_particles, dtype=PARTICLE_DTYPE
        ).tobytes(),
        num_particles=num_particles,
    )
