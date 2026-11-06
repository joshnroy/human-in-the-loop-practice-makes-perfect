"""Observable kitchen state uses the same symbolic lookup interface as pouring."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.environments.pouring3d.practice.types import PourObjects


class ClosedLoopPourParameters(BaseModel):
    """The kitchen Pour's one learned parameter: its tilt ramp rate past the approach."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 0.25 deg/s keeps the 80-95 deg ramp within 60 s; 5 deg/s is the widest approved
    # sampling bound (Josh, 2026-11-04).
    LIMITS: ClassVar[tuple[float, float]] = (0.25, 5.0)
    ramp_rate_deg_s: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_limits(self) -> "ClosedLoopPourParameters":
        low, high = self.LIMITS
        if not low <= self.ramp_rate_deg_s <= high:
            raise ValueError(f"ramp_rate_deg_s must lie in [{low}, {high}] deg/s")
        return self


class KitchenObjects(PourObjects):
    hob_type: ClassVar[Type] = Type(name="kitchen_hob", feature_names=("on", "cold"))
    hob: ClassVar[Object] = Object(name="hob", type=hob_type)

    def requests(self) -> tuple[Object, ...]:
        """None: a kitchen task's goal is its `Split{n}` atom, not a request object."""
        return ()

    def all(self) -> tuple[Object, ...]:
        return (*super().all(), self.hob)

    @classmethod
    def types(cls) -> tuple[Type, ...]:
        return (*(t for t in super().types() if t != cls.request_type), cls.hob_type)


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
