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
    # Measured 2026-09-22 at the live pick controller with the trap-2 stuck
    # arrangement (cube 0.073 m off bin center, face gap 0.052 m): every probed
    # placement with bin x <= -1.0 REFUSED the grasp (20/20, including mid-room y,
    # 0.65-1.0 m from any wall plane), while the former range x in [-2.3, -1.48]
    # sat entirely inside that refusing zone -- which is what let a practiced toss
    # strand marginal cubes there. This is the MAXIMAL grasp-safe rectangle the
    # eastward-extent mapping supports: every probed point inside
    # x in [-0.9, 0.2], y in [-1.0, 1.5] accepted and lifted the marginal cube
    # (14/14 across the two probe rounds), while y = -1.5 refused at x = 0.6 and
    # y = +-1.9 refused at x = -0.5 (0/2), so the rectangle stops there. The
    # y = 0 corridor stays graspable east to at least x = 1.1, but the region is
    # north-south asymmetric beyond the rectangle, so only the rectangle is
    # encoded. Its x <= 0.2 also keeps clear of the cube spawn strip
    # (x in [0.54, 0.71]). A larger receiver region keeps more toss standoff
    # geometry feasible west of the barrier at x = 1.25.
    Tossing3DSide.ROBOT: Tossing3DResetRegion(
        ranges=((-0.9, -1.0, 0.2, 1.5),), yaw_ranges=((180, 180),)
    ),
    Tossing3DSide.OPPOSITE: Tossing3DResetRegion(
        ranges=((2.60, -2.3, 3.42, 2.3),), yaw_ranges=((0, 0),)
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
