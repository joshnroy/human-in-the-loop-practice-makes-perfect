"""Region selection must test neighboring settings rather than only isolated hits."""

import json
from unittest.mock import MagicMock

from scripts.pouring3d_wood_regions import WoodRegions


def test_patch_selects_interior_success_and_tests_two_parameter_neighbors():
    rows = [
        {"eligible_count": 1, "parameters": {"tilt_angle": 95.5, "wiggle_seconds": t}}
        for t in WoodRegions.DURATIONS
    ]
    patch = WoodRegions.patch(rows=rows, target=1)
    assert patch["center"] == [95.5, 2.5]
    assert len(patch["points"]) == 9
    assert [95.25, 2] in patch["points"]
    assert [95.75, 3] in patch["points"]


def test_missing_count_refines_an_observed_gap_without_assuming_success():
    rows = [
        {"eligible_count": 0, "parameters": {"tilt_angle": 94.75, "wiggle_seconds": 2.5}},
        {"eligible_count": 3, "parameters": {"tilt_angle": 95.5, "wiggle_seconds": 2.5}},
    ]
    patch = WoodRegions.patch(rows=rows, target=2)
    assert patch["center"] == [95.125, 2.5]
    assert patch["coarse_exact_found"] is False
    assert len(patch["points"]) == 9


def test_count_map_can_score_one_from_an_explicit_two_cube_trial():
    row = {"requested_count": 2, "eligible_count": 1, "pour_success": False}
    assert WoodRegions.scores(row=row, target=1)
    assert not WoodRegions.scores(row=row, target=2)
    assert not WoodRegions.scores(row={"eligible_count": None}, target=1)


def test_integer_angle_patch_can_be_saved_as_json():
    rows = [{"eligible_count": 1, "parameters": {"tilt_angle": 94, "wiggle_seconds": 6}}]
    patch = WoodRegions.patch(rows=rows, target=1)
    assert json.loads(json.dumps(patch))["center"] == [94, 6]


def test_resume_preserves_completed_cells_without_rerunning_physics(*, tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({"source_sha256": {}}))
    for index, duration in enumerate(WoodRegions.DURATIONS):
        directory = tmp_path / "coarse-s0-g0/0" / f"cell-{index}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps({
                "seed": 0,
                "requested_count": 2,
                "parameters": {"tilt_angle": WoodRegions.ANGLES[0], "wiggle_seconds": duration},
            })
        )
    native = MagicMock(side_effect=AssertionError("Completed cell was rerun"))
    monkeypatch.setattr("scripts.pouring3d_wood_regions.WiggleDemo.run", native)
    WoodRegions.worker(root=tmp_path, phase="coarse", seed=0, group=0)
    native.assert_not_called()
