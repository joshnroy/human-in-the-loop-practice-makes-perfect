"""Typed symbolic sides of Tossing3D's one-way barrier."""

from enum import Enum
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.types import Object, Type


class Tossing3DSide(str, Enum):
    """A side relative to the robot, rather than to world-frame x."""

    ROBOT = "robot_side"
    OPPOSITE = "opposite_side"


# Continuous placement geometry belongs to the environment implementation, not the
# static scene JSON. The symbolic side object selects one of these two regions.
class Tossing3DResetRegion(BaseModel):
    """A code-defined continuous placement region for one symbolic side."""

    model_config = ConfigDict(frozen=True)

    target: str = "ground"
    ranges: tuple[tuple[float, float, float, float], ...]
    yaw_ranges: tuple[tuple[float, float], ...]


BIN_RESET_REGION_BY_SIDE: Final[dict[Tossing3DSide, Tossing3DResetRegion]] = {
    Tossing3DSide.ROBOT: Tossing3DResetRegion(
        ranges=((-2.3, -2.3, -1.48, 2.3),), yaw_ranges=((180, 180),)
    ),
    Tossing3DSide.OPPOSITE: Tossing3DResetRegion(
        ranges=((1.48, -2.3, 2.3, 2.3),), yaw_ranges=((0, 0),)
    ),
}


class Tossing3DSides:
    """Featureless PDDL objects used as reset destinations."""

    type: ClassVar[Type] = Type(name="tossing3d_side", feature_names=())
    robot: ClassVar[Object] = Object(name=Tossing3DSide.ROBOT.value, type=type)
    opposite: ClassVar[Object] = Object(name=Tossing3DSide.OPPOSITE.value, type=type)

    @classmethod
    def objects(cls) -> tuple[Object, Object]:
        return (cls.robot, cls.opposite)

    @classmethod
    def parse(cls, *, name: str) -> Object:
        by_name = {side.name: side for side in cls.objects()}
        try:
            return by_name[name]
        except KeyError as error:
            raise ValueError(
                f"unknown Tossing3D side {name!r}; expected {sorted(by_name)}"
            ) from error
