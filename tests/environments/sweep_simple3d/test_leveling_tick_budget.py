"""Checked leveling paths receive bounded settling time without changing tolerance."""

import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize(
    ("waypoints", "expected"),
    [(0, 180), (2, 180), (6, 180), (7, 210), (41, 1230), (60, 1800), (100, 1800)],
)
def test_leveling_budget_preserves_short_paths_and_bounds_branch_changes(
    *, waypoints: int, expected: int
) -> None:
    assert FloorPrimitives.leveling_tick_budget(path_waypoints=waypoints) == expected
