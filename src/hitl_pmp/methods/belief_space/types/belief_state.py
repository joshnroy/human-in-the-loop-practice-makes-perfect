"""Tossing3D belief-state data."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief

from .particle_filter_belief import ParticleFilterBelief
from .weighted_hypothesis_belief import WeightedHypothesisBelief

ConcreteSkillBelief = BayesianSkillBelief | ParticleFilterBelief | WeightedHypothesisBelief


class SamplerTrainingState(BaseModel):
    """Label support collected so far and used by the currently fitted sampler.

    A one-class fit cannot rank candidates. Its examples remain available for
    the first mixed-class fit, rather than being spent on fictitious learning.
    """

    model_config = ConfigDict(frozen=True)

    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    fitted_successes: int = Field(default=0, ge=0)
    fitted_failures: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_fitted_counts(self) -> Self:
        if self.fitted_successes > self.successes or self.fitted_failures > self.failures:
            raise ValueError("fitted label counts cannot exceed observed label counts")
        return self

    @property
    def fitted_mixed_classes(self) -> bool:
        return self.fitted_successes > 0 and self.fitted_failures > 0

    @property
    def refit_examples(self) -> int:
        if not self.successes or not self.failures:
            return 0
        credited = self.fitted_successes + self.fitted_failures if self.fitted_mixed_classes else 0
        return self.successes + self.failures - credited

    def observe(self, *, success: bool) -> "SamplerTrainingState":
        field = "successes" if success else "failures"
        return self.model_copy(update={field: getattr(self, field) + 1})

    def refitted(self) -> "SamplerTrainingState":
        return self.model_copy(
            update={"fitted_successes": self.successes, "fitted_failures": self.failures}
        )


class Tossing3DBeliefState(BaseModel):
    """Latent-controller posterior, sampler lifecycle, and paid practice cost.

    Production initialization includes sampler metadata. Its empty default keeps
    historical serialized states and standalone abstract-model states readable;
    absent metadata means the abstract learning model has no runtime constraint.
    """

    model_config = ConfigDict(frozen=True)

    skill_beliefs: dict[str, ConcreteSkillBelief]
    pending_examples: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    sampler_training: dict[str, SamplerTrainingState] = Field(default_factory=dict)
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
