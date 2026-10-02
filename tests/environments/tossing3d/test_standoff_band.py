"""The far-bin feasible standoff band, computed from geometry before any draw.

Offline: the geometry mirrors the live room's colliders (see test_toss_direction.py).
The two corner bins are the evaluation bins that raised NoFeasibleTossDirectionError in
the 2026-09-25 overnight run (seed 1 task 1, seed 2 task 3); their expected upper
bounds are where the direction-0 base footprint first touches the 45-degree corner wall:
x + y = -3.5 (south-west) and y - x = 3.5 (north-west), less the 0.01 m half-thickness.
"""

import math

import pytest

from hitl_pmp.environments.tossing3d.toss_direction import (
    InfeasibleStandoffBandError,
    TossStandoffBand,
)
from hitl_pmp.environments.tossing3d.types import PlanarCollisionBox, TossFeasibilityGeometry
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    FAR_STAND_X_LIMIT_M,
    WIDE_TOSS_STANDOFF_BOUNDS,
)

from .test_toss_direction import _geometry


def _corner_upper(*, bin_x: float, bin_y: float) -> float:
    """Standoff at which the footprint corner reaches the diagonal wall's face."""
    half = 0.275
    reach = 0.01 * math.sqrt(2)
    if bin_y < 0:
        return (bin_x - half) + (bin_y - half) + 3.5 - reach
    return 3.5 - reach - (bin_y + half) + (bin_x - half)


@pytest.fixture(autouse=True)
def _fresh_cache():
    TossStandoffBand.clear_cache()
    yield
    TossStandoffBand.clear_cache()


@pytest.mark.parametrize(
    ("bin_x", "bin_y"),
    [(1.830, -2.176), (1.582, 2.207)],
)
def test_a_corner_bin_gets_the_upper_bound_where_the_corner_wall_intrudes(
    *, bin_x: float, bin_y: float
) -> None:
    low, high = TossStandoffBand.far_bounds(geometry=_geometry(bin_xy=(bin_x, bin_y), bin_yaw=0.0))
    assert low == pytest.approx(max(WIDE_TOSS_STANDOFF_BOUNDS[0], bin_x - FAR_STAND_X_LIMIT_M))
    assert high == pytest.approx(_corner_upper(bin_x=bin_x, bin_y=bin_y), abs=2e-3)
    assert high < WIDE_TOSS_STANDOFF_BOUNDS[1]


def test_the_measured_corner_bounds() -> None:
    """The two bins' bounds, pinned: 2.590 m and 2.311 m."""
    first = TossStandoffBand.far_bounds(geometry=_geometry(bin_xy=(1.830, -2.176), bin_yaw=0.0))
    second = TossStandoffBand.far_bounds(geometry=_geometry(bin_xy=(1.582, 2.207), bin_yaw=0.0))
    assert first[1] == pytest.approx(2.590, abs=2e-3)
    assert second[1] == pytest.approx(2.311, abs=2e-3)


@pytest.mark.parametrize(
    ("bin_x", "bin_y"),
    [(2.105, -1.753), (3.42, 2.3), (3.42, -2.3), (1.6, 0.0), (2.6, -2.3), (3.2, 2.0)],
)
def test_a_bin_with_no_corner_intrusion_keeps_exactly_floor_to_the_controller_ceiling(
    *, bin_x: float, bin_y: float
) -> None:
    low, high = TossStandoffBand.far_bounds(geometry=_geometry(bin_xy=(bin_x, bin_y), bin_yaw=0.0))
    assert (low, high) == (
        max(WIDE_TOSS_STANDOFF_BOUNDS[0], bin_x - FAR_STAND_X_LIMIT_M),
        WIDE_TOSS_STANDOFF_BOUNDS[1],
    )


def _with_pillar(*, geometry: TossFeasibilityGeometry, center: tuple[float, float]):
    pillar = PlanarCollisionBox(name="pillar", center=center, width=0.2, height=0.2, yaw=0.0)
    return geometry.model_copy(update={"obstacles": (*geometry.obstacles, pillar)})


def test_a_feasible_set_that_is_not_one_interval_raises() -> None:
    """A pillar mid-way along the west ray splits the band in two; the rule never
    picks a piece."""
    geometry = _with_pillar(
        geometry=_geometry(bin_xy=(2.5, 0.0), bin_yaw=0.0), center=(2.5 - 2.0, 0.0)
    )
    with pytest.raises(InfeasibleStandoffBandError, match="not one interval"):
        TossStandoffBand.far_bounds(geometry=geometry)


def test_an_empty_band_raises() -> None:
    geometry = _with_pillar(
        geometry=_geometry(bin_xy=(2.5, 0.0), bin_yaw=0.0), center=(2.5 - 1.9, 0.0)
    )
    geometry = geometry.model_copy(
        update={
            "obstacles": (
                *geometry.obstacles,
                PlanarCollisionBox(name="slab", center=(-0.2, 0.0), width=2.0, height=0.8, yaw=0.0),
            )
        }
    )
    with pytest.raises(InfeasibleStandoffBandError, match="empty"):
        TossStandoffBand.far_bounds(geometry=geometry)


def test_the_band_is_deterministic_and_cached_per_geometry() -> None:
    geometry = _geometry(bin_xy=(1.830, -2.176), bin_yaw=0.0)
    first = TossStandoffBand.far_bounds(geometry=geometry)
    TossStandoffBand.clear_cache()
    assert TossStandoffBand.far_bounds(geometry=geometry) == first
