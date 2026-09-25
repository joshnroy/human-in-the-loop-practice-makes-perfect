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
    # A block of the south-wall band, measured 2026-09-24 with the real base motion
    # planner and live picks. Across bin centres every 0.15 m over x in
    # [-0.35, 0.40], y in [-1.90, -1.30] every cell was placeable, picked the cube
    # from its spawn strip (3 seeds x 4 robot poses, whenever the robot was clear of
    # the bin) and had a plannable stand direction at every standoff in
    # [1.25, 2.60]; at 2.6 m the only one is standing north of the bin. The block
    # is the part of the wider band x in [-0.35, 0.85], y in [-2.80, -1.30] where a
    # cube lying in the bin 0.072 m off centre is also pickable toward all four
    # walls; that wider band refused those picks (98/160 picked) in three zones:
    #   S offset for y <= -2.10 (the south wall behind the bin),
    #   E offset for x >= 0.40 (the barrier behind the approach),
    #   W offset for y <= -2.00 at x = -0.33 (the south-west diagonal corner).
    # Edges keep 0.10 m to the first refusal: x <= 0.30 (0.35 picked, 0.40
    # refused), y >= -1.90 (-1.95 picked, -2.00 refused at x = -0.33). West
    # x >= -0.33 is kindergarden's placement limit: its sampler reserves the
    # south-west diagonal wall's axis-aligned box, so centres west of -0.338 are
    # unplaceable. North y <= -1.30 is the probed edge, >= 0.9 m from the cube
    # spawn region. The former rectangle x in [-0.9, 0.2], y in [-1.0, 1.5] had
    # cells with no plannable direction at standoffs >= 2.15 m (17/500) and cells
    # whose pick was refused. The robot and cube-spawn exclusions are applied at
    # reset time by `bin_placement.BinPlacementRules`, not baked into these bounds.
    Tossing3DSide.ROBOT: Tossing3DResetRegion(
        ranges=((-0.33, -1.9, 0.3, -1.3),), yaw_ranges=((180, 180),)
    ),
    # Mirrors the installed task's graded receiver spawn range (kindergarden's
    # union of the original pre-#191 near-barrier support, west edge 1.48 =
    # barrier face 1.33 + bin half-footprint 0.15, with #191's far extension
    # to 3.42), so practice resets and evaluation sample the same distance
    # ramp. Near-barrier placements are measured usable: 12/12 scored at bins
    # x in {1.6, 2.0} including 1.25 m-standoff throws over the barrier.
    Tossing3DSide.OPPOSITE: Tossing3DResetRegion(
        ranges=((1.48, -2.3, 3.42, 2.3),), yaw_ranges=((0, 0),)
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
