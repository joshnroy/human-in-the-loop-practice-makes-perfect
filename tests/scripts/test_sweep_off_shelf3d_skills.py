"""`scripts/sweep_off_shelf3d_skills.py`: the parameter contract of `sweep_shelf` and
stroke planning run with no simulator; the smoke test builds the skills with no
recorder at all (no renderer, no video) and checks what they measure on the initial
scene. It skips where `kinder`, the `SweepOffShelf3D` task or the MimicLabs assets are
missing. The skills themselves are minutes long and are exercised by
`validate_sweep_off_shelf3d.py`, not here."""

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.sweep_off_shelf3d_skills import (
    SKILLS,
    ShelfSkills,
    Strokes,
    SweepSettings,
)


def test_the_six_planner_skills_are_exposed() -> None:
    assert SKILLS == (
        "pick_wiper",
        "sweep_shelf",
        "return_wiper",
        "pick_cube_from_shelf",
        "pick_cube_from_floor",
        "place_cube_in_box",
    )
    for name in SKILLS:
        assert callable(getattr(ShelfSkills, name))


@pytest.mark.parametrize(("height", "speed"), [(0.0, 0.05), (0.04, 0.25), (0.01, 0.1)])
def test_sweep_settings_accept_the_documented_ranges(*, height: float, speed: float) -> None:
    settings = SweepSettings(height=height, speed=speed)
    assert (settings.height, settings.speed) == (height, speed)


@pytest.mark.parametrize(
    ("height", "speed"), [(-0.001, 0.1), (0.041, 0.1), (0.01, 0.04), (0.01, 0.26)]
)
def test_sweep_settings_reject_values_outside_the_ranges(*, height: float, speed: float) -> None:
    with pytest.raises(ValidationError):
        SweepSettings(height=height, speed=speed)


def test_stroke_plan_keeps_every_cube_out_from_under_the_handle() -> None:
    ys = [-0.168, -0.092, -0.057, 0.051, 0.2]
    centres = Strokes.plan(ys=ys, reach=0.08, band=0.03, limit=0.17)
    assert centres
    for c in centres:
        assert all(abs(y - c) > 0.03 for y in ys)
        assert abs(c) <= 0.17 + 1e-9
    for y in ys:
        assert any(0.03 < abs(y - c) <= 0.08 + 1e-9 or abs(y) > 0.17 + 0.08 for c in centres)


def test_stroke_plan_covers_a_lone_cube_from_one_side() -> None:
    centres = Strokes.plan(ys=[0.0], reach=0.08, band=0.03, limit=0.17)
    assert len(centres) == 1
    assert 0.03 < abs(centres[0]) <= 0.08


def test_stroke_plan_is_empty_for_an_empty_shelf() -> None:
    assert Strokes.plan(ys=[], reach=0.08, band=0.03, limit=0.17) == []


def _task_and_assets_present() -> bool:
    if importlib.util.find_spec("kinder") is None:
        return False
    import kinder

    dynamic3d = Path(kinder.__path__[0]) / "envs" / "dynamic3d"
    return (dynamic3d / "tasks" / "SweepOffShelf3D" / "SweepOffShelf3D-o5.json").is_file() and (
        dynamic3d / "models" / "assets" / "mimiclabs_scenes" / "meshes"
    ).is_dir()


@pytest.mark.skipif(
    not _task_and_assets_present(),
    reason="kindergarden with SweepOffShelf3D and the MimicLabs assets is not installed",
)
def test_skills_run_with_no_recorder() -> None:
    skills = ShelfSkills(seed=0, tick_limit=10)
    try:
        assert skills.recorder is None
        assert skills.counts() == {"in_box": 0, "on_floor": 0, "on_shelf": 5}
        skills.hold(count=3)
        assert skills.ticks == 3
        assert skills.board[1] == pytest.approx(0.2794, abs=1e-3)
    finally:
        skills.close()
