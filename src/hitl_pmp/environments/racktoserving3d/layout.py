"""The one distribution over dumpling layouts on the cooling rack.

Every placement onto the rack draws from it: each episode's initial layout (so
evaluation tasks and practice resets alike), the person's batch return, and the
robot's put-back of a single dumpling. A layout is uniform over the rack's live
footprint inset by ``MARGIN``, each yaw uniform in [-``MAX_YAW``, ``MAX_YAW``], every
centre at least ``MIN_SEPARATION`` from every other, by joint rejection sampling.
A single put-back is the same distribution conditioned on the other dumplings'
current centres. The caller's seeded generator makes every draw deterministic.
"""

import functools
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
    CLUSTER_SIGMA: ClassVar[tuple[float, float]] = (0.015, 0.10)
    MAX_CLUSTER_DRAWS: ClassVar[int] = 5_000
    MAX_CLUSTER_PILES: ClassVar[int] = 100
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
    def clustered(
        *,
        rng: np.random.Generator,
        rack: ServingRegion,
        count: int,
        center_y: float | None = None,
        center_y_sigma: float | None = None,
    ) -> tuple[tuple[tuple[float, float, float], ...], float]:
        """One random pile and its spread: (`count` (x, y, yaw) centres, sigma).

        The pile centre is uniform over the inset rack and sigma log-uniform in
        ``CLUSTER_SIGMA``; each centre is the pile centre plus N(0, sigma) per axis,
        clipped to the inset, each yaw uniform in [-``MAX_YAW``, ``MAX_YAW``]. The
        pile centre and sigma are drawn once, so rejection never loosens a tight
        sigma. Pieces are placed one at a time, each redrawn until its real footprint
        clears those already placed (`footprints_overlap`): joint rejection of all
        five found no layout in 200,000 draws at sigma = 0.017 m.

        With ``center_y_sigma`` (tau), the pile centre's y is instead N(``center_y``,
        tau) clipped to the inset, its x still uniform: a pile at the plate line.
        """
        low = np.array([rack.x_min + RackLayout.MARGIN, rack.y_min + RackLayout.MARGIN])
        high = np.array([rack.x_max - RackLayout.MARGIN, rack.y_max - RackLayout.MARGIN])
        if center_y_sigma is None:
            centre = rng.uniform(low, high)
        else:
            if center_y is None:
                raise ValueError("a plate-line pile needs center_y")
            centre = np.clip(
                [rng.uniform(low[0], high[0]), rng.normal(center_y, center_y_sigma)], low, high
            )
        sigma = float(np.exp(rng.uniform(*np.log(RackLayout.CLUSTER_SIGMA))))
        for _ in range(RackLayout.MAX_CLUSTER_PILES):
            layout: list[tuple[float, float, float]] = []
            for _ in range(RackLayout.MAX_CLUSTER_DRAWS):
                x, y = np.clip(centre + rng.normal(0.0, sigma, size=2), low, high).tolist()
                pose = (x, y, float(rng.uniform(-RackLayout.MAX_YAW, RackLayout.MAX_YAW)))
                if not any(RackLayout.footprints_overlap(a=pose, b=other) for other in layout):
                    layout.append(pose)
                    if len(layout) == count:
                        return tuple(layout), sigma
        raise RuntimeError(f"no clustered layout at sigma {sigma:.3f} m")

    @staticmethod
    def footprints_overlap(*, a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        """Whether two potsticker footprints (convex hulls, at (x, y, yaw)) intersect."""
        if math.hypot(a[0] - b[0], a[1] - b[1]) >= 2 * RackLayout._footprint_radius():
            return False
        polygons = [RackLayout._placed_footprint(pose=pose) for pose in (a, b)]
        for polygon in polygons:
            edges = np.roll(polygon, -1, axis=0) - polygon
            normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
            projections = [other @ normals.T for other in polygons]
            if np.any(
                (projections[0].max(axis=0) < projections[1].min(axis=0))
                | (projections[1].max(axis=0) < projections[0].min(axis=0))
            ):
                return False
        return True

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _footprint() -> np.ndarray:
        """The footprint's convex hull at zero yaw, counter-clockwise (x, y)."""
        from scipy.spatial import ConvexHull

        points = RackPotsticker.support_vertices()[:, :2]
        return np.asarray(points[ConvexHull(points).vertices])

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _footprint_radius() -> float:
        return float(np.linalg.norm(RackLayout._footprint(), axis=1).max())

    @staticmethod
    def _placed_footprint(*, pose: tuple[float, float, float]) -> np.ndarray:
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return RackLayout._footprint() @ np.array([[c, s], [-s, c]]) + [x, y]

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
