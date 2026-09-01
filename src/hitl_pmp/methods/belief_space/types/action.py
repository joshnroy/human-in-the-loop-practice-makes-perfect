"""Tossing3D POMDP action data."""

from pydantic import Field

from hitl_pmp.core.method.types import GroundSkill

from .core import POMDPAction


class Tossing3DAction(POMDPAction):
    name: str
    ground_skill: GroundSkill = Field(exclude=True)
