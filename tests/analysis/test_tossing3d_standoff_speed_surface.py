"""The surface figure's two data-shaping steps, tested on data whose answer is known.

The figure itself draws no inference -- it draws cells and the geometric prediction over
them. What can still be silently wrong is what goes *into* those cells:

  * **`executable_standoffs`** decides which rows are real measurements. A standoff where
    `MoveToThrowPose` or `Toss` raised is not a failed throw, it is no throw; folding those
    rows in as `0/1` would draw a failure where nothing happened, and would also drag the
    measured `range(speed)` -- which the prediction band is derived from -- towards two rows
    whose recorded "flight" is the cube being dropped;
  * **`geometric_band_edges`** is the prediction being tested. If it were off by the sign
    or by the half-width, the figure would show measurement agreeing with the wrong theory.
"""

import numpy as np
import pytest

from analysis.tossing3d_standoff_speed_surface import (
    executable_standoffs,
    geometric_band_edges,
    range_by_speed,
)


def _grid(*, rows: list[dict]) -> dict:
    return {
        "standoffs": sorted({r["standoff"] for r in rows}),
        "speeds": sorted({r["commanded_speed_deg"] for r in rows}),
        "seeds": [0],
        "cells": {(r["standoff"], r["commanded_speed_deg"]): r for r in rows},
    }


def _cell(
    *, standoff: float, speed: float, move: str | None = None, toss: str | None = None
) -> dict:
    return {
        "standoff": standoff,
        "commanded_speed_deg": speed,
        "move_error": move,
        "toss_error": toss,
        "solved": False,
        "ballistic_impact_x": standoff + speed / 100.0,
        "base_x_before_toss": standoff,
    }


def test_geometric_band_edges_are_the_range_offset_by_the_half_box() -> None:
    """`standoff in [range - h, range + h]` is the whole geometric prediction."""
    lower, upper = geometric_band_edges(ranges=np.array([1.0, 1.5]), half_width=0.15)
    assert lower == [0.85, 1.35]
    assert upper == [1.15, 1.65]


def test_geometric_band_is_exactly_the_goal_box_wide() -> None:
    """A band narrower or wider than the box would be a different prediction."""
    lower, upper = geometric_band_edges(ranges=np.array([1.2]), half_width=0.15)
    assert upper[0] - lower[0] == pytest.approx(0.30)


def test_a_row_whose_skill_never_executed_is_not_an_executable_row() -> None:
    """The barrier-blocked and unreachable rows must not be drawn as failed throws."""
    rows = [
        _cell(standoff=0.90, speed=60.0, move="AssertionError"),
        _cell(standoff=0.90, speed=80.0, move="AssertionError"),
        _cell(standoff=1.35, speed=60.0),
        _cell(standoff=1.35, speed=80.0),
        _cell(standoff=2.10, speed=60.0, toss="Motion planning failed"),
        _cell(standoff=2.10, speed=80.0),
    ]
    assert executable_standoffs(grid=_grid(rows=rows)) == [1.35]


def test_a_row_is_executable_only_if_every_speed_in_it_ran() -> None:
    """One raised speed makes the whole row's `range(speed)` contributions untrustworthy."""
    rows = [
        _cell(standoff=1.35, speed=60.0),
        _cell(standoff=1.35, speed=80.0, toss="Motion planning failed"),
    ]
    assert executable_standoffs(grid=_grid(rows=rows)) == []


def test_range_by_speed_averages_only_the_rows_it_is_given() -> None:
    """The band is drawn from measured ranges, so a non-executable row must not enter it."""
    rows = [
        _cell(standoff=0.90, speed=60.0, move="AssertionError"),
        _cell(standoff=1.35, speed=60.0),
        _cell(standoff=1.50, speed=60.0),
    ]
    grid = _grid(rows=rows)
    # `_cell` builds `ballistic_impact_x - base_x = speed/100`, identical in every row, so
    # any leakage from the excluded row would have to show up as a different mean.
    assert range_by_speed(grid=grid, standoffs=[1.35, 1.50]) == pytest.approx({60.0: 0.60})
