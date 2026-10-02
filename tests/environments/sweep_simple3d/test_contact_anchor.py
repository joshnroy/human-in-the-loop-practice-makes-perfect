"""Other cubes must not shift the selected cube's broad contact anchor."""

import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


@pytest.mark.parametrize(
    "projections",
    [
        [1.61, 1.301, 1.493, 1.535, 1.455],
        [1.59277, 1.31755, 1.49292, 1.53576, 1.44581],
    ],
)
def test_selected_cube_anchor_is_independent_of_neighbors(*, projections: list[float]) -> None:
    for target in projections:
        assert FloorPrimitives.broad_blade_center(projections=projections, target=target) == target
