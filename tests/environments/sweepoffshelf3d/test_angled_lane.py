"""The rotated SweepShelf stroke's geometry: [offset, angle] -> head-centre line."""

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    STROKE_END_X,
    STROKE_START_X,
    AngledLane,
    SweepParams,
)

FRONT_X = 1.373


def test_zero_angle_is_the_straight_stroke() -> None:
    lane = AngledLane.plan(cube_xy=[1.5, 0.0], offset=0.05, angle=0.0, front_x=FRONT_X)
    assert lane.shift == 0.0
    np.testing.assert_allclose(lane.direction, [-1.0, 0.0])
    np.testing.assert_allclose(lane.start, [STROKE_START_X, 0.05])
    np.testing.assert_allclose(lane.end, [STROKE_END_X, 0.05])
    np.testing.assert_allclose(lane.mouth, [FRONT_X - 0.12, 0.05])


@pytest.mark.parametrize("angle", [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0])
@pytest.mark.parametrize("offset", [-0.1, 0.0, 0.05])
def test_line_passes_the_cube_at_offset_along_the_head(*, angle: float, offset: float) -> None:
    cube = np.array([1.5, 0.0])
    lane = AngledLane.plan(cube_xy=cube, offset=offset, angle=angle, front_x=FRONT_X)
    d, n = lane.direction, lane.normal
    assert abs(d @ n) < 1e-12
    # The head (along n) is perpendicular to the pull, and the pull is turned by angle.
    assert np.isclose(np.rad2deg(np.arctan2(d[1], -d[0])), angle)
    # Every point of the stroke is offset + shift from the cube along n.
    for p in (lane.start, lane.end, lane.mouth):
        assert np.isclose((p - cube) @ n, offset + lane.shift)
    # Start is behind the end along the pull, and the stroke is inside the cupboard.
    assert (lane.end - lane.start) @ d > 0.05
    assert AngledLane.footprint_ok(head=lane.start, angle=angle, front_x=FRONT_X)


def test_positive_angle_pulls_toward_plus_y() -> None:
    lane = AngledLane.plan(cube_xy=[1.5, 0.0], offset=0.0, angle=30.0, front_x=FRONT_X)
    assert lane.end[1] > lane.start[1]


def test_params_bound_the_angle() -> None:
    SweepParams(offset=0.1, angle=-45.0)
    with pytest.raises(ValueError):
        SweepParams(offset=0.0, angle=46.0)
