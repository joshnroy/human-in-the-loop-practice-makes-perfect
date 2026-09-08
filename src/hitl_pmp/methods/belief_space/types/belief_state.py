"""Tossing3D belief-state data."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .particle_filter_belief import ParticleFilterBelief
from .weighted_hypothesis_belief import WeightedHypothesisBelief

ConcreteSkillBelief = ParticleFilterBelief | WeightedHypothesisBelief


class SkillExecutionObservation(BaseModel):
    """Evidence from one completed execution, deferred to the cycle boundary."""

    model_config = ConfigDict(frozen=True)

    success: bool | None = None
    observed_cost: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_evidence(self) -> "SkillExecutionObservation":
        if self.success is None and self.observed_cost is None:
            raise ValueError("an execution observation must contain outcome or cost evidence")
        return self


class Tossing3DBeliefState(BaseModel):
    """Latent-controller posterior, pending examples, and paid practice cost."""

    model_config = ConfigDict(frozen=True)

    skill_beliefs: dict[str, ConcreteSkillBelief]
    pending_examples: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    pending_execution_observations: dict[str, tuple[SkillExecutionObservation, ...]] = Field(
        default_factory=dict
    )
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
