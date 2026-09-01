"""Skill-belief data used by the Tossing3D practice model."""

from pydantic import BaseModel, ConfigDict, Field

from .core import Theta


class SkillBelief(BaseModel):
    """Posterior over a skill's competence and learning rate."""

    model_config = ConfigDict(frozen=True)

    hypotheses: tuple["WeightedHypothesis", ...]


class SkillHypothesis(Theta):
    model_config = ConfigDict(frozen=True)

    competence: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)


class WeightedHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis: SkillHypothesis
    probability: float = Field(gt=0.0, le=1.0)
