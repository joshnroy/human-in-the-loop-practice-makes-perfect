"""Skill-belief data used by the Tossing3D practice model."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BeliefEstimator = Literal["finite_grid", "particle_filter"]


class SkillBelief(BaseModel):
    """Posterior over a skill's competence and its derivative per example."""

    model_config = ConfigDict(frozen=True)

    hypotheses: tuple["WeightedHypothesis", ...]
    estimator: BeliefEstimator = "finite_grid"
    resampling_count: int = Field(default=0, ge=0)
    resampling_seed: int = Field(default=0, ge=0)


class SkillHypothesis(BaseModel):
    """One hypothesis for ``(competence, d competence / d example)``."""

    model_config = ConfigDict(frozen=True)

    competence: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)
