"""Representative clips must retain requested goals and physical provenance."""

from copy import deepcopy

import pytest

from scripts.pouring3d_video_reels import PourVideoReels


def test_closest_miss_keeps_the_original_request_and_prefers_retained_cubes_on_ties():
    under = {
        "requested_count": 4,
        "eligible_count": 3,
        "pour_success": False,
        "full_success": False,
        "seed": 0,
        "parameters": {"tilt_angle": 97, "wiggle_seconds": 2},
        "events": [],
        "source_path": "under",
    }
    over = {**under, "eligible_count": 5, "source_path": "over"}
    wrong_request = {**under, "requested_count": 3, "eligible_count": 4}
    assert (
        PourVideoReels.choose(rows=[over, wrong_request, under], target=4, success=False) is under
    )
    assert PourVideoReels.choose(rows=[under], target=4, success=True) is None
    assert PourVideoReels.choose(rows=[under], target=5, success=False) is None


def test_equal_counts_do_not_hide_a_different_physical_replay():
    reference = {
        "qpos": [0.0],
        "qvel": [0.0],
        "ctrl": [0.0],
        "time": 1.0,
        "locations": ["pitcher"] * 5,
        "settled": True,
        "goal": False,
    }
    rendered = deepcopy(reference)
    PourVideoReels.compare_snapshot(reference=reference, rendered=rendered, label="Pour")
    rendered["qpos"] = [0.01]
    with pytest.raises(ValueError, match="qpos"):
        PourVideoReels.compare_snapshot(reference=reference, rendered=rendered, label="Pour")
