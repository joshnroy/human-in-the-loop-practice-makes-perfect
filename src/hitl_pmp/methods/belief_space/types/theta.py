"""Sampled Tossing3D policy-parameter data."""

from pydantic import BaseModel, ConfigDict

from .skill_belief import SkillHypothesis


class Tossing3DTheta(BaseModel):
    model_config = ConfigDict(frozen=True)

    pick: SkillHypothesis
    toss: SkillHypothesis
    open_gripper: SkillHypothesis
