"""`scripts/render_sweep_into_hole3d.py`: the view list and argument parsing need
no simulator; the render itself runs only where `kinder` is installed, which is the
case in the `hitl-pmp` env since the `tossing3d` extra, so it does execute locally
and on CI rather than skipping everywhere. The lab2 set additionally needs the
MimicLabs assets, which CI restores from its cache."""

import importlib.util
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from scripts.render_sweep_into_hole3d import (
    ENV_ID,
    SCENES,
    VIEWS,
    RenderPlan,
    SweepIntoHole3DRenderer,
)

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="kindergarden is not installed"
)


def test_the_four_views_cover_reset_and_containment_from_the_task_cameras() -> None:
    assert len(VIEWS) == 4
    assert {view.camera for view in VIEWS} == {"task_view", "hole_view", "top_view"}
    assert [view.camera for view in VIEWS if view.after_drop] == ["hole_view"]
    assert ENV_ID == "kinder/SweepIntoHole3D-o5-v0"


def test_the_lab_scene_is_primary_and_the_simple_scene_carries_an_infix() -> None:
    assert list(SCENES) == ["mimiclabs-lab2", "simple"]
    names = {view.filename(scene=scene) for scene in SCENES for view in VIEWS}
    assert len(names) == 8
    assert "sweep_into_hole3d_task_view.png" in names
    assert "sweep_into_hole3d_simple_task_view.png" in names


def test_parse_args_builds_a_plan_with_fixed_defaults(*, tmp_path: Path) -> None:
    plan = SweepIntoHole3DRenderer.parse_args(argv=["--output-dir", str(tmp_path)])
    assert plan == RenderPlan(output_dir=tmp_path, seed=0)
    assert plan.scenes == ("mimiclabs-lab2", "simple")
    narrowed = SweepIntoHole3DRenderer.parse_args(
        argv=["--output-dir", "x", "--seed", "7", "--scenes", "simple"]
    )
    assert narrowed.seed == 7
    assert narrowed.scenes == ("simple",)


@needs_kinder
def test_render_writes_four_distinct_non_blank_frames_per_scene(*, tmp_path: Path) -> None:
    scenes = ("simple",)
    if SweepIntoHole3DRenderer.mimiclabs_assets_present():
        scenes = ("mimiclabs-lab2", "simple")
    written = SweepIntoHole3DRenderer.render(plan=RenderPlan(output_dir=tmp_path, scenes=scenes))
    assert set(written) == {view.filename(scene=scene) for scene in scenes for view in VIEWS}
    frames = {name: iio.imread(path) for name, path in written.items()}
    for frame in frames.values():
        assert frame.shape == (480, 640, 3)
        assert frame.std() > 1.0
    for scene in scenes:
        infix = SCENES[scene]
        before = frames[f"sweep_into_hole3d_{infix}hole_view.png"].astype(np.int16)
        after = frames[f"sweep_into_hole3d_{infix}hole_view_two_cubes_in_pit.png"].astype(np.int16)
        assert np.abs(before - after).max() > 0
