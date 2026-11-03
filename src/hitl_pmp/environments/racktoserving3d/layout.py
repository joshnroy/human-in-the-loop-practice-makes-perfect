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

from .semantics import ServingRegion


class RackLayout:
    MARGIN: ClassVar[float] = 0.03
    MIN_SEPARATION: ClassVar[float] = 0.05
    MAX_YAW: ClassVar[float] = math.pi / 4
    MAX_DRAWS: ClassVar[int] = 10_000

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
