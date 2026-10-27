"""Bounded sideways discovery must retain missing counts and avoid duplicate probes."""

from scripts.pouring3d_lateral_counts import LateralCounts


def test_refinement_bisects_an_observed_angle_gap_for_missing_count():
    rows = [
        {"eligible_count": 2, "parameters": {"tilt_angle": 105, "wiggle_seconds": 12}},
        {"eligible_count": 5, "parameters": {"tilt_angle": 107, "wiggle_seconds": 12}},
    ]
    target, angle, duration = LateralCounts.proposal(rows=rows, material="foam", step=0)
    assert target in {3, 4}
    assert angle == 106 and duration == 12
    rows.append({
        "eligible_count": 3,
        "parameters": {"tilt_angle": angle, "wiggle_seconds": duration},
    })
    _, next_angle, next_duration = LateralCounts.proposal(rows=rows, material="foam", step=1)
    assert (next_angle, next_duration) != (angle, duration)


def test_duration_gap_can_be_refined_without_changing_tilt():
    rows = [
        {"eligible_count": 0, "parameters": {"tilt_angle": 95.5, "wiggle_seconds": 0}},
        {"eligible_count": 2, "parameters": {"tilt_angle": 95.5, "wiggle_seconds": 12}},
    ]
    target, angle, duration = LateralCounts.proposal(rows=rows, material="wood", step=0)
    assert (target, angle, duration) == (1, 95.5, 6)


def test_render_selection_prefers_exact_full_sequence_over_closest_failure():
    miss = {"eligible_count": 3, "pour_success": False, "full_success": False, "seed": 0}
    pour_only = {"eligible_count": 2, "pour_success": True, "full_success": False, "seed": 1}
    full = {"eligible_count": 2, "pour_success": True, "full_success": True, "seed": 2}
    assert LateralCounts.render_choice(rows=[miss, pour_only, full], target=2) is full
    assert LateralCounts.render_choice(rows=[miss], target=2) is miss
