"""A dispersed group must not strand the sweep behind a barely covered cube."""

import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def test_dispersed_group_covers_target_and_near_neighbor_without_edge_contact() -> None:
    projections = [1.61, 1.301, 1.493, 1.535, 1.455]
    center = FloorPrimitives.broad_blade_center(projections=projections, target=1.61)
    assert abs(center - 1.61) <= 0.14 + 1e-9
    assert abs(center - 1.455) <= 0.14 + 1e-9
    assert not any(0.14 + 1e-9 < abs(p - center) < 0.17 - 1e-9 for p in projections)


def test_native_compact_group_preserves_the_original_center() -> None:
    projections = [1.59277, 1.31755, 1.49292, 1.53576, 1.44581]
    assert FloorPrimitives.broad_blade_center(
        projections=projections, target=1.59277
    ) == pytest.approx((1.59277 + 1.31755) / 2)
