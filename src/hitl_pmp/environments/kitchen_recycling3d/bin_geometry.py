"""The kitchen's recycling bin: a tapered open box, and what counts as inside it.

The real bin is the blue Rubbermaid FG295673BLUE deskside recycling bin (prpl-tidybot
`conf/toss_cycles.yaml`, `bin:`). Its walls taper from the rim to a narrower base; the
base was fitted in the ceiling camera and is not yet pinned exactly. MuJoCo and the
PyBullet planning scene both build the bin from `panels()`, so the two collision models
are the same boxes.

A cube is in the bin when its centre is inside the inner wall faces at its own height,
above the floor panel and below the rim: the whole interior scores. The real robot's
scorer keeps a 5 cm `in_bin_margin` inside the walls; this one deliberately does not.
"""

from __future__ import annotations

from math import atan, cos, sin
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

from .domain import KitchenConfig


class BinPanel(BaseModel):
    """One collision box in the bin's own frame (origin at the floor panel's base)."""

    model_config = ConfigDict(frozen=True)

    half_extents: tuple[float, float, float]
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    wall: bool


class TaperedBin(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Each wall is this many tilted strips, each as wide as the wall at its own top, so
    # the corners stay closed while the base narrows; the protrusion this buys is at
    # most a quarter of the taper per side (~4 mm).
    STRIPS: ClassVar[int] = 4

    length: float = Field(gt=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    wall: float = Field(gt=0)
    base_length: float = Field(gt=0)
    base_width: float = Field(gt=0)

    @model_validator(mode="after")
    def valid(self) -> TaperedBin:
        if not (self.base_length <= self.length and self.base_width <= self.width):
            raise ValueError("A tapered bin's base must not exceed its rim")
        if min(self.base_length, self.base_width) <= 2 * self.wall:
            raise ValueError("Bin wall must leave a positive interior at the base")
        return self

    @staticmethod
    def of(*, config: KitchenConfig) -> TaperedBin:
        length, width, height = config.bin_outside
        base_length, base_width = config.bin_base
        return TaperedBin(
            length=length,
            width=width,
            height=height,
            wall=config.simulation_wall,
            base_length=base_length,
            base_width=base_width,
        )

    def _tilt(self, *, axis: int) -> float:
        rim, base = (self.length, self.base_length) if axis == 0 else (self.width, self.base_width)
        return atan((rim - base) / 2 / self.height)

    def outer_half(self, *, axis: int, z: float) -> float:
        rim, base = (self.length, self.base_length) if axis == 0 else (self.width, self.base_width)
        return base / 2 + (rim - base) / 2 * z / self.height

    def inner_half(self, *, axis: int, z: float) -> float:
        """Half the interior's extent along `axis` (0 = the bin's x) at height `z`."""
        return self.outer_half(axis=axis, z=z) - self.wall / cos(self._tilt(axis=axis))

    def interior_box(self) -> tuple[float, float, float, float, float, float]:
        """The interior's bounding box in the bin frame: the rim opening from the floor
        panel's top to the rim."""
        x, y = (self.inner_half(axis=axis, z=self.height) for axis in (0, 1))
        return (-x, -y, self.wall, x, y, self.height)

    def contains_local(self, *, point: npt.ArrayLike) -> bool:
        x, y, z = (float(v) for v in np.asarray(point, dtype=float)[:3])
        if not self.wall <= z <= self.height:
            return False
        return abs(x) <= self.inner_half(axis=0, z=z) and abs(y) <= self.inner_half(axis=1, z=z)

    def contains(
        self,
        *,
        point: npt.ArrayLike,
        bin_position: npt.ArrayLike,
        bin_quaternion_xyzw: npt.ArrayLike,
    ) -> bool:
        """Whether a world point is inside the bin posed at (position, quaternion)."""
        offset = np.asarray(point, dtype=float)[:3] - np.asarray(bin_position, dtype=float)[:3]
        local = Rotation.from_quat(np.asarray(bin_quaternion_xyzw, dtype=float)).inv().apply(offset)
        return self.contains_local(point=local)

    def panels(self) -> tuple[BinPanel, ...]:
        """The floor panel, then each wall as `STRIPS` tilted boxes; outer faces run
        from the base footprint at z = 0 to the rim footprint at z = height."""
        t = self.wall
        panels = [
            BinPanel(
                half_extents=(self.base_length / 2, self.base_width / 2, t / 2),
                position=(0.0, 0.0, t / 2),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                wall=False,
            )
        ]
        for axis in (0, 1):
            tilt = self._tilt(axis=axis)
            slant = self.height / cos(tilt) / self.STRIPS
            for sign in (1.0, -1.0):
                for strip in range(self.STRIPS):
                    middle = (strip + 0.5) / self.STRIPS
                    top = (strip + 1) / self.STRIPS
                    z = self.height * middle
                    normal = sign * (self.outer_half(axis=axis, z=z) - t / 2 * cos(tilt))
                    centre_z = z + t / 2 * sin(tilt)
                    across = self.outer_half(axis=1 - axis, z=self.height * top)
                    if axis == 0:
                        half = (t / 2, across, slant / 2)
                        position = (normal, 0.0, centre_z)
                        # About y by +tilt leans the +x wall outward at its top.
                        rotation = Rotation.from_euler("y", sign * tilt)
                    else:
                        half = (across, t / 2, slant / 2)
                        position = (0.0, normal, centre_z)
                        rotation = Rotation.from_euler("x", -sign * tilt)
                    x, y, zq, w = (float(v) for v in rotation.as_quat())
                    panels.append(
                        BinPanel(
                            half_extents=half,
                            position=position,
                            quaternion_xyzw=(x, y, zq, w),
                            wall=True,
                        )
                    )
        return tuple(panels)


class TaperedBinCollision:
    """The tapered bin in a pinned-KINDER `PyBulletSim`, registered exactly where
    `PyBulletSim.add_bin` registers its straight-walled bin, so every planner query
    that sees a bin (collision sets, `set_state`, `bin_wall_clearance`) sees this one."""

    @staticmethod
    def add(*, sim: Any, name: str, pose: Any, geometry: TaperedBin) -> int:
        import pybullet as p
        from kinder_models.dynamic3d.utils import _PARKED_POSITION

        if name in sim._bins:  # noqa: SLF001
            raise ValueError(f"Bin {name} is already in the collision model")
        client = sim._physics_client_id  # noqa: SLF001

        def body(*, panels: tuple[BinPanel, ...], position: Any, orientation: Any) -> int:
            shape = p.createCollisionShapeArray(
                shapeTypes=[p.GEOM_BOX] * len(panels),
                halfExtents=[list(panel.half_extents) for panel in panels],
                collisionFramePositions=[list(panel.position) for panel in panels],
                collisionFrameOrientations=[list(panel.quaternion_xyzw) for panel in panels],
                physicsClientId=client,
            )
            return int(
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=shape,
                    basePosition=position,
                    baseOrientation=orientation,
                    physicsClientId=client,
                )
            )

        panels = geometry.panels()
        bin_body = body(panels=panels, position=pose.position, orientation=pose.orientation)
        sim._bounding_boxes[name] = (geometry.length, geometry.width, geometry.height)  # noqa: SLF001
        sim._bins[name] = bin_body  # noqa: SLF001
        sim._bin_walls[name] = body(  # noqa: SLF001
            panels=tuple(panel for panel in panels if panel.wall),
            position=_PARKED_POSITION,
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        sim._bin_poses[name] = pose  # noqa: SLF001
        return bin_body
