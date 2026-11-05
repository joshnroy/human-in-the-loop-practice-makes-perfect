"""The vectorised pitcher containment answers exactly as KinDER's per-point loop."""

import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.pitcher_containment import PitcherContainment
from hitl_pmp.environments.pouring3d.potsticker_shape import support_vertices


def test_vectorised_containment_matches_upstream_on_every_point_and_piece():
    env = KitchenEnvironment(requested_count=1)
    try:
        pitcher = env.controller().scene.get_object("pitcher")
        rng = np.random.default_rng(0)
        half = np.array([pitcher.length / 2 + 0.03, pitcher.width / 2 + 0.03])
        # Single points, dense around the walls, floor and rim, so both the near-wall
        # band and the ray-crossing branch decide many of them.
        points = np.column_stack((
            rng.uniform(-half[0], half[0], 6000),
            rng.uniform(-half[1], half[1], 6000),
            rng.uniform(-0.01, pitcher.height + 0.01, 6000),
        ))
        for point in points:
            expected = pitcher.contains_points(point[None])
            assert (
                PitcherContainment.contains_points(pitcher=pitcher, points=point[None]) == expected
            )
        # Whole payload pieces in random poses: inside, outside and straddling.
        vertices = support_vertices()
        answers = []
        for _ in range(1500):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rotation = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            centre = np.r_[rng.uniform(-half, half) * 0.8, rng.uniform(0.0, pitcher.height)]
            piece = vertices @ rotation.T + centre
            expected = pitcher.contains_points(piece)
            assert PitcherContainment.contains_points(pitcher=pitcher, points=piece) == expected
            answers.append(expected)
        assert 0 < sum(answers) < len(answers)
    finally:
        env.close()
