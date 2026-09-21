"""Immutable observations supporting empirical Tossing3D failure effects."""

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom


class FailureEffectCount(BaseModel):
    """A failure delta counted within one action, symbolic context and sampler mode."""

    model_config = ConfigDict(frozen=True)

    ground_skill: GroundSkill
    before_atoms: frozenset[GroundAtom]
    was_random_exploration: bool
    add_effects: frozenset[GroundAtom]
    delete_effects: frozenset[GroundAtom]
    count: int = Field(default=1, ge=1)
