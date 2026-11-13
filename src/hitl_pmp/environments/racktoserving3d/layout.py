"""The one distribution over dumpling layouts on the cooling rack.

Every placement onto the rack draws from it: each episode's initial layout (so
evaluation tasks and practice resets alike), the person's batch return, and the
robot's put-back of a single dumpling. A layout is uniform over the rack's live
footprint inset by ``MARGIN``, each yaw uniform in [-``MAX_YAW``, ``MAX_YAW``], every
centre at least ``MIN_SEPARATION`` from every other, by joint rejection sampling.
A single put-back is the same distribution conditioned on the other dumplings'
current centres. The caller's seeded generator makes every draw deterministic.
"""

import itertools
import math
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from .potsticker import RackPotsticker
from .semantics import ServingRegion


class RackLayout:
    MARGIN: ClassVar[float] = 0.03
    # The potsticker's footprint fits a 60.4 mm circle, so 61 mm apart never overlap.
    MIN_SEPARATION: ClassVar[float] = 0.061
    MAX_YAW: ClassVar[float] = math.pi / 4
    MAX_DRAWS: ClassVar[int] = 10_000
    LINED_UP_FRONT_GAP: ClassVar[float] = 0.010
    LINED_UP_JITTER: ClassVar[float] = 0.005
    LINED_UP_YAW_JITTER: ClassVar[float] = 0.15

    @staticmethod
    def sample(
        *,
        rng: np.random.Generator,
        rack: ServingRegion,
        count: int,
        others: Sequence[Sequence[float]] = (),
    ) -> tuple[tuple[float, float, float], ...]:
        """`count` (x, y, yaw) centres, each clear of the others and of `others`' (x, y)."""
        low = (rack.x_min + RackLayout.MARGIN, rack.y_min + RackLayout.MARGIN)
        high = (rack.x_max - RackLayout.MARGIN, rack.y_max - RackLayout.MARGIN)
        fixed = [(float(x), float(y)) for x, y, *_ in others]
        for _ in range(RackLayout.MAX_DRAWS):
            xs = rng.uniform(low[0], high[0], size=count)
            ys = rng.uniform(low[1], high[1], size=count)
            yaws = rng.uniform(-RackLayout.MAX_YAW, RackLayout.MAX_YAW, size=count)
            points = list(zip(xs.tolist(), ys.tolist(), strict=True))
            if RackLayout._clear(points=points, others=fixed):
                return tuple((x, y, yaw) for (x, y), yaw in zip(points, yaws.tolist(), strict=True))
        raise RuntimeError(f"no rack layout of {count} dumplings within {RackLayout.MAX_DRAWS}")

    @staticmethod
    def lined_up(
        *, rng: np.random.Generator, rack: ServingRegion, center_y: float, kind: str
    ) -> tuple[tuple[float, float, float], ...]:
        """Five (x, y, yaw) centres packed against the rack's front edge on y = center_y.

        Opt-in starts for the Sweep vs GatherAndSweep probe (2026-11-05), each a guess
        at GatherAndSweep's post-gather state, not a measured one:
        ``pile2col``: long axis along x, columns at center_y +- 15 mm (3 + 2), 60 mm
        x pitch; ``stack``: long axis along y, one column, 32 mm x pitch; ``row``: long
        axis along x, one row across y, 32 mm pitch. The nearest pieces' front ends
        sit ``LINED_UP_FRONT_GAP`` behind the front edge. Each centre is jittered
        +-``LINED_UP_JITTER`` and each yaw +-``LINED_UP_YAW_JITTER``; the caller
        rejects colliding draws.
        """
        depth, width, _ = RackPotsticker.extents()
        along_x = math.pi / 2
        if kind == "pile2col":
            first = rack.x_min + RackLayout.LINED_UP_FRONT_GAP + width / 2
            nominal = [(first + 0.060 * i, center_y + 0.015, along_x) for i in range(3)]
            nominal += [(first + 0.060 * i, center_y - 0.015, along_x) for i in range(2)]
        elif kind == "stack":
            first = rack.x_min + RackLayout.LINED_UP_FRONT_GAP + depth / 2
            nominal = [(first + 0.032 * i, center_y, 0.0) for i in range(5)]
        elif kind == "row":
            x = rack.x_min + RackLayout.LINED_UP_FRONT_GAP + width / 2
            nominal = [(x, center_y + 0.032 * (i - 2), along_x) for i in range(5)]
        else:
            raise ValueError(f"unknown lined-up layout {kind!r}")
        jitter = rng.uniform(-1, 1, size=(5, 3)) * [
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_YAW_JITTER,
        ]
        return tuple(
            (x + dx, y + dy, yaw + dyaw)
            for (x, y, yaw), (dx, dy, dyaw) in zip(nominal, jitter.tolist(), strict=True)
        )

    @staticmethod
    def spaced(
        *,
        rng: np.random.Generator,
        rack: ServingRegion,
        center_y: float,
        pitch: float,
        long_axis: str,
    ) -> tuple[tuple[float, float, float], ...]:
        """Five (x, y, yaw) centres in one evenly spaced row across the rack.

        The opt-in two-route start (Josh, 2026-11-12): a row on the rack's x centre
        line, centred on y = center_y (the plate's centre line), `pitch` apart, each
        piece's long axis along world `long_axis`; jittered as `lined_up` is. How
        many a straight Sweep delivers from it is a measured fact, not a property of
        this function.
        """
        x = (rack.x_min + rack.x_max) / 2
        yaw = math.pi / 2 if long_axis == "x" else 0.0
        jitter = rng.uniform(-1, 1, size=(5, 3)) * [
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_YAW_JITTER,
        ]
        return tuple(
            (x + dx, center_y + pitch * (index - 2) + dy, yaw + dyaw)
            for index, (dx, dy, dyaw) in enumerate(jitter.tolist())
        )

    @staticmethod
    def middle_three(
        *,
        rng: np.random.Generator,
        rack: ServingRegion,
        center_y: float,
        pitch: float,
        side_offset: float,
        long_axis: str,
    ) -> tuple[tuple[float, float, float], ...]:
        """Three in the middle and one out on each side (Josh, 2026-11-12).

        One row on the rack's x centre line: three `pitch` apart, centred on
        y = center_y (the plate's centre line), and one more at `side_offset` on each
        side of it; jittered as `lined_up` is, ordered by y. The middle three are the
        ones a straight Sweep is meant to land; whether it does, and whether a
        GatherAndSweep brings the outer two in, are measured facts.
        """
        x = (rack.x_min + rack.x_max) / 2
        yaw = math.pi / 2 if long_axis == "x" else 0.0
        offsets = (-side_offset, -pitch, 0.0, pitch, side_offset)
        jitter = rng.uniform(-1, 1, size=(5, 3)) * [
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_JITTER,
            RackLayout.LINED_UP_YAW_JITTER,
        ]
        return tuple(
            (x + dx, center_y + offset + dy, yaw + dyaw)
            for offset, (dx, dy, dyaw) in zip(offsets, jitter.tolist(), strict=True)
        )

    @staticmethod
    def _clear(*, points: list[tuple[float, float]], others: list[tuple[float, float]]) -> bool:
        pairs = itertools.chain(
            itertools.combinations(points, 2), itertools.product(points, others)
        )
        return all(
            math.hypot(a[0] - b[0], a[1] - b[1]) >= RackLayout.MIN_SEPARATION for a, b in pairs
        )

    @staticmethod
    def quaternion(*, yaw: float) -> np.ndarray:
        """MuJoCo (w, x, y, z) for a rotation of `yaw` about world z."""
        return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
