"""Recipe selection must preserve failed and unresolved requested counts."""

import pytest

from scripts.pouring3d_wiggle_counts import WiggleCounts


def test_selection_uses_contained_settled_outcomes_and_preserves_closest_miss():
    spill = {"eligible_count": None, "parameters": {"tilt_angle": 108, "wiggle_seconds": 1}}
    three = {"eligible_count": 3, "parameters": {"tilt_angle": 107, "wiggle_seconds": 2}}
    five = {"eligible_count": 5, "parameters": {"tilt_angle": 130, "wiggle_seconds": 0}}
    assert WiggleCounts.select(rows=[spill, three, five], target=3) is three
    assert WiggleCounts.select(rows=[spill, three, five], target=2) is three
    assert WiggleCounts.select(rows=[spill, three, five], target=5) is five
    with pytest.raises(ValueError, match="No settled"):
        WiggleCounts.select(rows=[spill], target=2)


def test_refinement_avoids_replaying_a_duration_already_attempted():
    rows = [
        {"eligible_count": 0, "parameters": {"tilt_angle": 107, "wiggle_seconds": 0.5}},
        {"eligible_count": 3, "parameters": {"tilt_angle": 107, "wiggle_seconds": 2}},
    ]
    target, angle, duration = WiggleCounts.proposal(rows=rows, base=107, step=0)
    assert target in {1, 2, 4, 5}
    assert angle == 107
    assert 0.5 < duration < 2
    rows.append({
        "eligible_count": 0,
        "parameters": {"tilt_angle": angle, "wiggle_seconds": duration},
    })
    _, next_angle, next_duration = WiggleCounts.proposal(rows=rows, base=107, step=1)
    assert (next_angle, next_duration) != (angle, duration)
