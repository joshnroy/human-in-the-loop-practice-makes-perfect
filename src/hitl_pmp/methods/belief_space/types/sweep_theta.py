"""A joint draw of independent Sweep controller hypotheses."""

from pydantic import BaseModel, ConfigDict

from .skill_belief import SkillHypothesis


class SweepTheta(BaseModel):
    model_config = ConfigDict(frozen=True)
    skills: dict[str, SkillHypothesis]
