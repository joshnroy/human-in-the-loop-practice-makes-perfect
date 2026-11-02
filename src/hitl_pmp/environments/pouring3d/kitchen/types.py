"""Observable kitchen state uses the same symbolic lookup interface as pouring."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.environments.pouring3d.practice.types import PourObjects


class KitchenObjects(PourObjects):
    hob_type: ClassVar[Type] = Type(name="kitchen_hob", feature_names=("on", "cold"))
    hob: ClassVar[Object] = Object(name="hob", type=hob_type)

    def all(self) -> tuple[Object, ...]:
        return (*super().all(), self.hob)

    @classmethod
    def types(cls) -> tuple[Type, ...]:
        return (*super().types(), cls.hob_type)


class Pose(BaseModel):
    """A body's world pose at one instant; quaternion in scipy (x, y, z, w) order."""

    model_config = ConfigDict(frozen=True)

    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]


class PickRefusal(BaseModel):
    """A dispatched dumpling pick refused to plan from these poses (`pick_plannable.py`)."""

    model_config = ConfigDict(frozen=True)

    dumpling: Pose
    pan: Pose
    error: str | None = None
