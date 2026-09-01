"""Tossing3D belief-state data."""

from pydantic import ConfigDict, Field

from .core import BeliefState
from .skill_belief import SkillBelief


class Tossing3DBeliefState(BeliefState):
    """Latent-controller posterior, pending examples, and paid practice cost."""

    model_config = ConfigDict(frozen=True)

    toss_belief: SkillBelief
    pick_belief: SkillBelief
    open_gripper_belief: SkillBelief
    pending_pick_examples: int = Field(default=0, ge=0)
    pending_open_gripper_examples: int = Field(default=0, ge=0)
    pending_training_examples: int = Field(default=0, ge=0)
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
