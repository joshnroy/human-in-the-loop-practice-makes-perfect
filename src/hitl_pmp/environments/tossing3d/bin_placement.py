"""Where a reset may put the bin: never on the robot, never near the cube's spawn.

Both rules are occupancy, not rejection. KINDER's reset sampler reserves static
geometry and the other movables but not the robot, so a partial reset used to drop
the bin inside the robot's footprint, after which every pick was refused (0/359 in
the 2026-09-24 sweep). Removing the forbidden zones from the bin's centre ranges
before sampling keeps the placement uniform over the valid free space; KINDER picks
a range with probability proportional to its area, so disjoint pieces stay uniform.

The spawn-region gap is the fine pick sweep of 2026-09-24: with the robot clear of
the bin, every pick at a footprint gap of >= 0.20 m to `blocks_init_region`
succeeded (998/998), while failures reached a 0.182 m gap. That sweep used three
robot poses, so it is a measured floor rather than a guarantee for every approach.
Zones are axis-aligned boxes, which is conservative for both rules: the robot's
rotated footprint is replaced by its bounding box, and the Euclidean gap by the
per-axis one.
"""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

Rect = tuple[float, float, float, float]

# A reset reads the bin back from a float32 observation after placement.
_CHECK_TOLERANCE_M = 1e-3


class BinPlacementViolationError(RuntimeError):
    """A reset left the bin somewhere the placement rules forbid."""


class BinPlacementRules:
    """The rules above. A static-method container, never instantiated."""

    CUBE_SPAWN_MIN_GAP_M: ClassVar[float] = 0.20

    @staticmethod
    def free_centre_ranges(
        *,
        ranges: Sequence[Rect],
        robot_aabb: Rect,
        cube_spawn: Sequence[Rect],
        bin_half: tuple[float, float],
        clearance: float,
    ) -> tuple[Rect, ...]:
        """Bin-centre ranges with the robot and the spawn-gap zones removed."""
        hx, hy = bin_half
        holes = [BinPlacementRules._grow(rect=robot_aabb, dx=hx + clearance, dy=hy + clearance)]
        gap = BinPlacementRules.CUBE_SPAWN_MIN_GAP_M + clearance
        holes += [BinPlacementRules._grow(rect=r, dx=hx + gap, dy=hy + gap) for r in cube_spawn]
        return BinPlacementRules.subtract(ranges=ranges, holes=holes)

    @staticmethod
    def subtract(*, ranges: Sequence[Rect], holes: Sequence[Rect]) -> tuple[Rect, ...]:
        """Disjoint rectangles covering `ranges` minus every hole."""
        pieces: list[Rect] = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in ranges]
        for hx0, hy0, hx1, hy1 in holes:
            remaining: list[Rect] = []
            for x0, y0, x1, y1 in pieces:
                if hx0 >= x1 or hx1 <= x0 or hy0 >= y1 or hy1 <= y0:
                    remaining.append((x0, y0, x1, y1))
                    continue
                cx0, cx1 = max(x0, hx0), min(x1, hx1)
                candidates = (
                    (x0, y0, cx0, y1),
                    (cx1, y0, x1, y1),
                    (cx0, y0, cx1, max(y0, hy0)),
                    (cx0, min(y1, hy1), cx1, y1),
                )
                remaining += [c for c in candidates if c[2] > c[0] and c[3] > c[1]]
            pieces = remaining
        return tuple(pieces)

    @staticmethod
    def aabb(*, center: Sequence[float], size: Sequence[float], yaw: float) -> Rect:
        """Axis-aligned bounds of a `size[0]` x `size[1]` box rotated by `yaw`."""
        cosine, sine = abs(float(np.cos(yaw))), abs(float(np.sin(yaw)))
        hx = (cosine * size[0] + sine * size[1]) / 2
        hy = (sine * size[0] + cosine * size[1]) / 2
        return (center[0] - hx, center[1] - hy, center[0] + hx, center[1] + hy)

    @staticmethod
    def check(
        *, bin_aabb: Rect, robot_aabb: Rect, cube_spawn: Sequence[Rect], context: str
    ) -> None:
        """Raise if a placed bin breaks either rule."""
        reasons = []
        overlap_x = min(bin_aabb[2], robot_aabb[2]) - max(bin_aabb[0], robot_aabb[0])
        overlap_y = min(bin_aabb[3], robot_aabb[3]) - max(bin_aabb[1], robot_aabb[1])
        if overlap_x > _CHECK_TOLERANCE_M and overlap_y > _CHECK_TOLERANCE_M:
            reasons.append(
                f"bin {BinPlacementRules._fmt(rect=bin_aabb)} overlaps the robot "
                f"{BinPlacementRules._fmt(rect=robot_aabb)}"
            )
        for spawn in cube_spawn:
            dx = max(spawn[0] - bin_aabb[2], bin_aabb[0] - spawn[2], 0.0)
            dy = max(spawn[1] - bin_aabb[3], bin_aabb[1] - spawn[3], 0.0)
            gap = float(np.hypot(dx, dy))
            if gap < BinPlacementRules.CUBE_SPAWN_MIN_GAP_M - _CHECK_TOLERANCE_M:
                reasons.append(
                    f"bin {BinPlacementRules._fmt(rect=bin_aabb)} is {gap:.3f} m from the cube "
                    f"spawn region {BinPlacementRules._fmt(rect=spawn)}, under the "
                    f"{BinPlacementRules.CUBE_SPAWN_MIN_GAP_M} m floor"
                )
        if reasons:
            raise BinPlacementViolationError(f"{context}: " + "; ".join(reasons))

    @staticmethod
    def _grow(*, rect: Rect, dx: float, dy: float) -> Rect:
        return (rect[0] - dx, rect[1] - dy, rect[2] + dx, rect[3] + dy)

    @staticmethod
    def _fmt(*, rect: Rect) -> str:
        return "[" + ", ".join(f"{v:.3f}" for v in rect) + "]"
