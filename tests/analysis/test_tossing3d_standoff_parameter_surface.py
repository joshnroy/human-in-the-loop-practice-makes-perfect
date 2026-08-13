"""The distance surface's arithmetic, on data whose answer is known in advance."""

import numpy as np
import pytest

from analysis.tossing3d_standoff_parameter_surface import (
    distance_surface,
    standoff_invariance,
    throw_distance_table,
)


def _cell(
    *,
    standoff: float,
    speed: float,
    release_ms: float,
    seed: int,
    impact: float = 1.0,
    threw: bool = True,
    base_x: float = 0.65,
) -> dict:
    """One grid row. `base_x_before_toss` is non-zero so a missing subtraction shows."""
    return {
        "standoff": standoff,
        "commanded_speed_deg": speed,
        "commanded_release_ms": release_ms,
        "seed": seed,
        "threw": threw,
        "base_x_before_toss": base_x,
        "ballistic_impact_x": base_x + impact if threw else None,
    }


def test_throw_distance_is_measured_from_the_base_not_the_world() -> None:
    """Bases 0.30 m apart: identical from the base, 0.30 apart in world x."""
    rows = [
        _cell(standoff=1.10, speed=140.0, release_ms=850.0, seed=0, base_x=0.90),
        _cell(standoff=1.40, speed=140.0, release_ms=850.0, seed=0, base_x=0.60),
    ]
    table = throw_distance_table(grid={"rows": rows})
    assert table[(1.10, 140.0, 850.0, 0)] == pytest.approx(1.0)
    assert table[(1.40, 140.0, 850.0, 0)] == pytest.approx(1.0)


def test_a_throw_that_never_happened_is_absent_rather_than_zero_distance() -> None:
    rows = [_cell(standoff=1.10, speed=60.0, release_ms=1400.0, seed=0, threw=False)]
    assert throw_distance_table(grid={"rows": rows}) == {}


def test_standoff_invariance_reports_a_planted_spread() -> None:
    """One group at two standoffs, differing by a planted 12 mm."""
    rows = [
        _cell(standoff=1.10, speed=140.0, release_ms=850.0, seed=0, impact=1.000),
        _cell(standoff=1.40, speed=140.0, release_ms=850.0, seed=0, impact=1.012),
    ]
    result = standoff_invariance(distances=throw_distance_table(grid={"rows": rows}))
    assert result["groups"] == 1
    assert result["max_spread_m"] == pytest.approx(0.012)
    assert result["over_10mm"] == 1
    assert result["over_20mm"] == 0


def test_standoff_invariance_ignores_groups_seen_at_one_standoff_only() -> None:
    rows = [
        _cell(standoff=1.10, speed=140.0, release_ms=850.0, seed=0, impact=1.0),
        _cell(standoff=1.10, speed=60.0, release_ms=300.0, seed=0, impact=0.4),
    ]
    result = standoff_invariance(distances=throw_distance_table(grid={"rows": rows}))
    assert result["groups"] == 0
    assert result["max_spread_m"] is None


def test_distance_surface_pools_standoffs_and_seeds_and_reports_its_counts() -> None:
    """An unmeasured cell stays `nan` with a zero count, not a neighbour's value."""
    rows = [
        _cell(standoff=1.10, speed=140.0, release_ms=850.0, seed=0, impact=1.00),
        _cell(standoff=1.40, speed=140.0, release_ms=850.0, seed=0, impact=1.10),
        _cell(standoff=1.10, speed=140.0, release_ms=850.0, seed=1, impact=1.20),
    ]
    surface, counts = distance_surface(
        distances=throw_distance_table(grid={"rows": rows}),
        speeds=[60.0, 140.0],
        release_ms_values=[300.0, 850.0],
    )
    assert surface[1][1] == pytest.approx(1.10)
    assert counts[1][1] == 3
    assert np.isnan(surface[0][0])
    assert counts[0][0] == 0
