"""Skill-belief data used by the Tossing3D practice model."""

from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

BeliefEstimator = Literal["finite_grid", "particle_filter"]
PARTICLE_STORAGE_VERSION: Final[Literal[1]] = 1
PARTICLE_DTYPE: Final = np.dtype("<f8")


class SkillBelief(BaseModel):
    """Posterior over a skill's competence and its derivative per example."""

    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    hypotheses: tuple["WeightedHypothesis", ...] = ()
    estimator: BeliefEstimator = "finite_grid"
    resampling_count: int = Field(default=0, ge=0)
    resampling_seed: int = Field(default=0, ge=0)
    particle_parameters: bytes = b""
    particle_weights: bytes = b""
    num_particles: int = Field(default=0, ge=0)
    particle_storage_version: Literal[1] | None = None

    @model_validator(mode="after")
    def validate_representation(self) -> "SkillBelief":
        if self.estimator == "finite_grid":
            if not self.hypotheses:
                raise ValueError("finite-grid beliefs require hypotheses")
            if self.num_particles or self.particle_parameters or self.particle_weights:
                raise ValueError("finite-grid beliefs cannot contain packed particles")
            return self
        if self.hypotheses:
            raise ValueError("particle beliefs cannot contain object hypotheses")
        if self.particle_storage_version != PARTICLE_STORAGE_VERSION:
            raise ValueError("unsupported particle storage version")
        parameters = np.frombuffer(self.particle_parameters, dtype=PARTICLE_DTYPE)
        weights = np.frombuffer(self.particle_weights, dtype=PARTICLE_DTYPE)
        if parameters.size != 2 * self.num_particles or weights.size != self.num_particles:
            raise ValueError("particle buffers do not match num_particles")
        parameters = parameters.reshape(-1, 2)
        if not np.all(np.isfinite(parameters)) or not np.all((parameters >= 0) & (parameters <= 1)):
            raise ValueError("particle parameters must be finite probabilities")
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("particle weights must be finite and positive")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("particle weights must sum to one")
        return self


class SkillHypothesis(BaseModel):
    """One hypothesis for ``(competence, d competence / d example)``."""

    model_config = ConfigDict(frozen=True)

    competence: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)
