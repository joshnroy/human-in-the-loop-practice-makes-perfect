"""The `(speed x release-ms)` surface's inference steps, on constructed grids whose answer
is known in advance rather than read off the code under test."""

import numpy as np
import pytest

from analysis.tossing3d_toss_parameter_surface import (
    adjacent_speed_reversals,
    dead_cells,
    distance_table,
    paired_difference_test,
    reach_interval,
    variance_shares,
)

SEEDS = (0, 1, 2)
SPEEDS = (60.0, 100.0, 140.0)
RELEASE_MS = (300.0, 850.0, 1400.0)


def _cell(*, speed: float, release_ms: float, seed: int, impact: float, threw: bool = True) -> dict:
    """One row in the grid's own JSON schema, `base_x_before_toss` non-zero so a forgotten
    subtraction is visible."""
    return {
        "standoff": 1.35,
        "commanded_speed_deg": speed,
        "commanded_release_ms": release_ms,
        "seed": seed,
        "threw": threw,
        "solved": False,
        "base_x_before_toss": 0.65,
        "ballistic_impact_x": None if not threw else 0.65 + impact,
        "cube_x_final": 0.65 + impact,
        "toss_error": None,
        "move_error": None,
    }


def _grid(*, rows: list[dict]) -> dict:
    return {
        "speeds": sorted({r["commanded_speed_deg"] for r in rows}),
        "release_ms": sorted({r["commanded_release_ms"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "rows": rows,
    }


def _surface(*, fn) -> dict:  # noqa: ANN001
    """A grid whose ballistic distance is exactly `fn(speed=, release_ms=, seed=)`.

    Keyword-only: at the call sites below, `(s, m, k)` and `(m, s, k)` look identical.
    """
    return _grid(
        rows=[
            _cell(speed=s, release_ms=m, seed=k, impact=fn(speed=s, release_ms=m, seed=k))
            for s in SPEEDS
            for m in RELEASE_MS
            for k in SEEDS
        ]
    )


def test_distance_table_reports_distance_from_the_base_not_world_x() -> None:
    grid = _surface(fn=lambda *, speed, release_ms, seed: 1.0)
    table = distance_table(grid=grid, field="ballistic_impact_x")
    assert table[(60.0, 300.0, 0)] == pytest.approx(1.0)
    assert len(table) == len(SPEEDS) * len(RELEASE_MS) * len(SEEDS)


def test_distance_table_omits_cells_where_nothing_was_thrown() -> None:
    rows = [
        _cell(speed=60.0, release_ms=300.0, seed=0, impact=1.0),
        _cell(speed=60.0, release_ms=850.0, seed=0, impact=0.0, threw=False),
    ]
    table = distance_table(grid=_grid(rows=rows), field="ballistic_impact_x")
    assert set(table) == {(60.0, 300.0, 0)}


def test_dead_cells_finds_exactly_the_cells_that_never_threw() -> None:
    rows = [
        _cell(speed=60.0, release_ms=300.0, seed=0, impact=1.0),
        _cell(speed=140.0, release_ms=1400.0, seed=0, impact=0.0, threw=False),
    ]
    assert dead_cells(grid=_grid(rows=rows)) == [(140.0, 1400.0, 0)]


def test_reach_interval_of_a_speed_only_surface_is_unchanged_by_the_second_dial() -> None:
    grid = _surface(fn=lambda *, speed, release_ms, seed: speed / 100.0)
    both = reach_interval(table=distance_table(grid=grid, field="ballistic_impact_x"))
    speed_only = reach_interval(
        table=distance_table(grid=grid, field="ballistic_impact_x"), release_ms=850.0
    )
    assert both == pytest.approx(speed_only)


def test_reach_interval_widens_when_the_second_dial_genuinely_moves_the_throw() -> None:
    grid = _surface(fn=lambda *, speed, release_ms, seed: speed / 100.0 + release_ms / 1000.0)
    table = distance_table(grid=grid, field="ballistic_impact_x")
    both = reach_interval(table=table)
    speed_only = reach_interval(table=table, release_ms=850.0)
    assert (both[1] - both[0]) > (speed_only[1] - speed_only[0]) + 1e-9


def test_variance_shares_attribute_a_pure_speed_surface_to_speed() -> None:
    shares = variance_shares(
        table=distance_table(
            grid=_surface(fn=lambda *, speed, release_ms, seed: speed / 100.0),
            field="ballistic_impact_x",
        )
    )
    assert shares["speed"] == pytest.approx(1.0, abs=1e-9)
    assert shares["release_ms"] == pytest.approx(0.0, abs=1e-9)
    assert shares["interaction"] == pytest.approx(0.0, abs=1e-9)


def test_variance_shares_attribute_a_pure_millisecond_surface_to_the_millisecond() -> None:
    shares = variance_shares(
        table=distance_table(
            grid=_surface(fn=lambda *, speed, release_ms, seed: release_ms / 1000.0),
            field="ballistic_impact_x",
        )
    )
    assert shares["release_ms"] == pytest.approx(1.0, abs=1e-9)
    assert shares["speed"] == pytest.approx(0.0, abs=1e-9)


def test_variance_shares_detect_a_purely_multiplicative_interaction() -> None:
    """A product term is what a speed-dependent swing duration would produce."""
    shares = variance_shares(
        table=distance_table(
            grid=_surface(
                fn=lambda *, speed, release_ms, seed: (speed / 100.0) * (release_ms / 1000.0)
            ),
            field="ballistic_impact_x",
        )
    )
    assert shares["interaction"] > 0.05
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)


def test_paired_difference_test_uses_the_pairing() -> None:
    """The arms differ by +0.10 on every pair while ranging over 1.0 m between pairs, so
    an unpaired test sees nothing and a paired test sees a certainty."""
    before = np.array([1.0, 1.4, 1.8, 2.0])
    result = paired_difference_test(before=before, after=before + 0.10)
    assert result["n"] == 4
    assert result["mean_difference"] == pytest.approx(0.10)
    assert result["p_value"] < 1e-6


def test_paired_difference_test_reports_no_effect_when_there_is_none() -> None:
    rng = np.random.default_rng(0)
    before = rng.normal(size=40)
    result = paired_difference_test(before=before, after=before + rng.normal(scale=1e-3, size=40))
    assert result["p_value"] > 0.05


def test_adjacent_speed_reversals_finds_a_planted_reversal() -> None:
    """Distance rises with speed except at 100 -> 140, planted to fall by 0.05 at every
    millisecond and seed."""

    def surface(*, speed: float, release_ms: float, seed: int) -> float:
        return {60.0: 1.00, 100.0: 1.30, 140.0: 1.25}[speed] + 1e-4 * seed

    reversals = adjacent_speed_reversals(
        table=distance_table(grid=_surface(fn=surface), field="ballistic_impact_x")
    )
    by_pair = {(r["speed_low"], r["speed_high"]): r for r in reversals}
    assert by_pair[(100.0, 140.0)]["mean_difference"] == pytest.approx(-0.05, abs=1e-9)
    assert by_pair[(100.0, 140.0)]["p_value"] < 0.01
    assert by_pair[(60.0, 100.0)]["mean_difference"] > 0


def test_adjacent_speed_reversals_invents_none_in_a_monotone_surface() -> None:
    reversals = adjacent_speed_reversals(
        table=distance_table(
            grid=_surface(fn=lambda *, speed, release_ms, seed: speed / 100.0 + 1e-4 * seed),
            field="ballistic_impact_x",
        )
    )
    assert all(r["mean_difference"] > 0 for r in reversals)
