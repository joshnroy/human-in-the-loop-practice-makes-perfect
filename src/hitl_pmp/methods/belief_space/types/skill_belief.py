"""Skill-belief data used by the Tossing3D practice model."""

from pydantic import BaseModel, ConfigDict, Field


class SkillBelief(BaseModel):
    """Posterior over a skill's competence and its derivative per example."""

    model_config = ConfigDict(frozen=True)

    hypotheses: tuple["WeightedHypothesis", ...]


class SkillHypothesis(BaseModel):
    """One hypothesis for ``(competence, d competence / d example)``."""

    model_config = ConfigDict(frozen=True)

    competence: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)
