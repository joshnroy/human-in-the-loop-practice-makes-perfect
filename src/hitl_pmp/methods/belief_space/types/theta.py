"""Sampled Tossing3D policy-parameter data."""

from .core import Theta
from .skill_belief import SkillHypothesis


class Tossing3DTheta(Theta):
    pick: SkillHypothesis
    toss: SkillHypothesis
    open_gripper: SkillHypothesis
