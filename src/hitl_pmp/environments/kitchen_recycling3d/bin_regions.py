"""Where the kitchen's bin starts and where a human puts it back, per side.

The kitchen used two fixed spots: the far bin at `KitchenConfig.bin_position` and the
robot-side bin at `placement.NEAR_BIN_POSITION`, so every evaluation toss was thrown at
one pose from one direction. These regions replace both, the way original Tossing3D's
`sides.BIN_RESET_REGION_BY_SIDE` does: the scene start draws the bin from the far region
by scene seed, and a human reset draws it from its destination side's region by scene
seed and reset count. Every draw is deterministic given those.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from hitl_pmp.environments.tossing3d.sides import Tossing3DResetRegion, Tossing3DSide

from .domain import KitchenConfig
from .placement import KitchenHumanPlacement

# The probe's grid spacing (`scripts/kitchen_recycling/bin_region_probe.py`): every
# region corner is within one step of a certified cell.
PROBE_GRID_STEP_M = 0.2

KITCHEN_BIN_REGION_BY_SIDE: Final[dict[Tossing3DSide, Tossing3DResetRegion]] = {
    # Measured 2026-11-08 in the live kitchen sim (`bin_region_probe.py`, table in
    # `bin_region_probe_results.jsonl`): on a 0.2 m grid over x in [-0.65, 0.95],
    # y in [-2.30, 2.30], 163/216 bin centres were placeable, dumped (`PourCubeOutOfBin`
    # emptied a cube dropped in the bin), picked the dumped cube from beside the bin,
    # planned a stand exactly 1.45 m away and scored a (160 deg/s, 740 ms) throw on the
    # first try. The other 53/216: 32/216 overlap the robot's start footprint and 21/216
    # failed the dump -- x >= 0.75 with |y| <= 0.50 or |y| >= 2.10, and (-0.25, 0.90).
    # This block is the north part, around the former fixed near spot (0.65, 1.40), where
    # all 45/45 cells were certified; its edges keep >= 0.2 m (one cell) to the nearest
    # failed dump. The dump was probed from the robot's scene-start pose only, and a
    # reset keeps the bin clear of wherever the robot then stands (`human_destination`).
    Tossing3DSide.ROBOT: Tossing3DResetRegion(
        ranges=((-0.65, 1.1, 0.95, 1.9),), yaw_ranges=((0, 0),)
    ),
    # Measured the same way on a 0.185 m grid over x in [1.79, 2.16], y in [-2.10, 0.49]:
    # 36/45 cells were placeable, picked the cube from a fixed floor spot, planned the
    # 1.45 m stand west of the bin (direction 0, the only one a far bin allows) and scored
    # (160, 740) on the first try. The other 9/45: 3/45 at y = -2.10 had no plannable
    # stand (the west stand leaves the drive zone below y = -2.06), and 6/45 at
    # y in {0.305, 0.49} overlap the chair `recycling_chair_0`. x is bounded by the
    # kitchen's own rules, not the probe: x >= 1.79 keeps the bin beyond
    # `KitchenConfig.direct_placement_reach` (`valid_layout` needs x > 1.78), and
    # x <= 2.16 keeps a 1.45 m stand west of the drive zone's east edge (0.71). The
    # y edges keep 0.16 m to the drive-zone limit and 0.125 m to the chair overlap.
    Tossing3DSide.OPPOSITE: Tossing3DResetRegion(
        ranges=((1.79, -1.9, 2.16, 0.1),), yaw_ranges=((0, 0),)
    ),
}

# Draws a robot-side reset tries before giving up on one clear of the robot.
_CLEAR_DRAWS = 200
# Distinct streams, so the bin draw never shifts the cube's scatter draw.
_SCENE_STREAM = 7_301
_RESET_STREAM = 7_302


class KitchenBinRegions:
    """A static-method container, never instantiated."""

    @staticmethod
    def rectangle(*, side: Tossing3DSide) -> tuple[float, float, float, float]:
        (extent,) = KITCHEN_BIN_REGION_BY_SIDE[side].ranges
        return extent

    @staticmethod
    def _uniform(*, side: Tossing3DSide, rng: np.random.Generator) -> tuple[float, float]:
        x0, y0, x1, y1 = KitchenBinRegions.rectangle(side=side)
        return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))

    @staticmethod
    def scene_start(*, seed: int) -> tuple[float, float, float]:
        """The scene's bin start: the far region, drawn by scene seed."""
        rng = np.random.default_rng((_SCENE_STREAM, seed))
        return (*KitchenBinRegions._uniform(side=Tossing3DSide.OPPOSITE, rng=rng), 0.0)

    @staticmethod
    def human_destination(
        *,
        config: KitchenConfig,
        side: Tossing3DSide,
        robot: npt.ArrayLike,
        furniture: Sequence[Sequence[float]],
        seed: int,
        index: int,
    ) -> tuple[float, float, float]:
        """Where a human puts the bin on `side` for reset `index` of scene `seed`: the
        first seeded draw from the side's region whose footprint keeps
        `KitchenHumanPlacement.MARGIN` from the robot and every `furniture` row
        `(x, y, size_x, size_y)`. Raises when none of `_CLEAR_DRAWS` is clear."""
        rng = np.random.default_rng((_RESET_STREAM, seed, index, list(Tossing3DSide).index(side)))
        half = np.array(config.bin_outside[:2]) / 2
        margin = KitchenHumanPlacement.MARGIN
        robot_xy = np.asarray(robot, dtype=float)[:2]
        for _ in range(_CLEAR_DRAWS):
            centre = np.array(KitchenBinRegions._uniform(side=side, rng=rng))
            outside = np.maximum(np.abs(robot_xy - centre) - half, 0.0)
            if float(np.linalg.norm(outside)) < config.robot_radius + margin:
                continue
            if any(
                np.all(np.abs(centre - np.array(row[:2])) < half + np.array(row[2:4]) / 2 + margin)
                for row in furniture
            ):
                continue
            return float(centre[0]), float(centre[1]), 0.0
        raise ValueError(f"No {side.value} bin placement clear of the robot at {robot_xy!r}")
