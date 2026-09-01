"""Tossing3D belief-state data."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .skill_belief import SkillBelief


class Tossing3DBeliefState(BaseModel):
    """Latent-controller posterior, pending examples, and paid practice cost."""

    model_config = ConfigDict(frozen=True)

    skill_beliefs: dict[str, SkillBelief]
    pending_examples: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    accumulated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
