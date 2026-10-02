"""Regression for an advanced cube being pulled back toward a distant column mate."""

import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def test_distant_column_mate_does_not_move_narrow_start_behind_it() -> None:
    # v163: cube0 reached y1.15682, while cube1 at y.56074 shares its blade column.
    # Including the distant cube put the robot base on the advanced target cube.
    offset = FloorPrimitives.contact_behind_offset(
        relative_positions=[(0.0, 0.0), (0.56074 - 1.15682, 0.02209)], narrow=True
    )
    assert offset == 0.0
    robot_y = 1.15682 - 0.20 - offset + 0.70
    assert robot_y - 1.15682 > 0.275 + 0.01


@pytest.mark.parametrize("narrow,along", [(True, -0.30), (False, -0.06)])
def test_nearby_cube_in_blade_footprint_still_moves_start_back(
    *, narrow: bool, along: float
) -> None:
    assert FloorPrimitives.contact_behind_offset(
        relative_positions=[(0.0, 0.0), (along, 0.0)], narrow=narrow
    ) == pytest.approx(-along)
