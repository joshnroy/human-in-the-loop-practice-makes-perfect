"""`scripts/validate_sweep_off_shelf3d.py`: argument parsing, stroke planning, the
stacked two-camera captions, the contact sheet and the ffmpeg writer run with no
simulator; the smoke test builds the probe in lab2 and checks the measurements it
reports on the initial scene. It skips where `kinder`, the `SweepOffShelf3D` task or
the MimicLabs assets are missing. The episodes are minutes long and are run by hand,
not here."""

import importlib.util
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from scripts.validate_sweep_off_shelf3d import (
    EPISODES,
    FRAME_SIZE,
    PANEL_SIZE,
    Captions,
    ShelfProbe,
    Strokes,
    ValidationPlan,
    Validator,
    VideoWriter,
)


def _frame(*, shade: int) -> np.ndarray:
    width, height = FRAME_SIZE
    return np.full((height, width, 3), shade, dtype=np.uint8)


def test_parse_args_defaults_to_every_episode_in_lab2(*, tmp_path: Path) -> None:
    plan = Validator.parse_args(argv=["--output-dir", str(tmp_path)])
    assert plan == ValidationPlan(output_dir=tmp_path)
    assert plan.episodes == EPISODES
    assert plan.scene == "mimiclabs-lab2"
    narrowed = Validator.parse_args(
        argv=[
            "--output-dir",
            "x",
            "--episodes",
            "sweep_cubes_into_box",
            "--seed",
            "3",
            "--tick-limit",
            "50",
        ]
    )
    assert narrowed.episodes == ("sweep_cubes_into_box",)
    assert narrowed.seed == 3
    assert narrowed.tick_limit == 50


def test_strokes_cover_every_cube_with_the_fewest_head_widths() -> None:
    assert Strokes.cover(ys=[], reach=0.08, limit=0.17) == []
    assert Strokes.cover(ys=[0.01, -0.03], reach=0.08, limit=0.17) == pytest.approx([-0.03 + 0.08])
    centres = Strokes.cover(ys=[-0.22, -0.15, 0.0, 0.05, 0.22], reach=0.08, limit=0.17)
    assert len(centres) == 3
    for y in (-0.22, -0.15, 0.0, 0.05, 0.22):
        assert any(abs(y - c) <= 0.08 + 0.05 + 1e-9 for c in centres)
    assert all(abs(c) <= 0.17 + 1e-9 for c in centres)


def test_stack_puts_two_panels_side_by_side() -> None:
    width, height = PANEL_SIZE
    left = np.zeros((height, width, 3), dtype=np.uint8)
    right = np.full((height, width, 3), 200, dtype=np.uint8)
    stacked = Captions.stack(panels=[left, right])
    assert stacked.shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    assert stacked[:, :width].mean() == 0
    assert stacked[:, width:].mean() == 200


def test_annotate_keeps_the_frame_size_and_draws_header_and_banner() -> None:
    plain = Captions.annotate(frame=_frame(shade=120), header="h", banner=None)
    bannered = Captions.annotate(frame=_frame(shade=120), header="h", banner="wiper dropped")
    assert plain.shape == bannered.shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    assert plain[:26].mean() < 120
    assert bannered[-44:, :, 0].mean() > plain[-44:, :, 0].mean()


def test_contact_sheet_lays_out_captioned_keyframes(*, tmp_path: Path) -> None:
    keyframes = [(f"step {i}", _frame(shade=40 * i)) for i in range(5)]
    path = Captions.contact_sheet(keyframes=keyframes, path=tmp_path / "sheet.png")
    sheet = iio.imread(path)
    assert sheet.shape == (3 * (FRAME_SIZE[1] + 30), 2 * FRAME_SIZE[0], 3)


def test_video_writer_encodes_a_short_clip(*, tmp_path: Path) -> None:
    writer = VideoWriter(path=tmp_path / "clip.mp4", fps=10)
    for shade in range(0, 250, 25):
        writer.write(frame=_frame(shade=shade))
    writer.close()
    assert writer.encoder in {"h264_nvenc", "libx264"}
    assert writer.frames == 10
    assert (tmp_path / "clip.mp4").stat().st_size > 0


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
def test_probe_measures_the_initial_scene(*, tmp_path: Path) -> None:
    plan = ValidationPlan(output_dir=tmp_path, tick_limit=20)
    probe = ShelfProbe(plan=plan, episode="pick_wiper")
    try:
        counts = probe.counts()
        assert counts == {"in_box": 0, "on_floor": 0, "on_shelf": 5}
        assert probe.wiper_lift() == pytest.approx(0.0, abs=0.005)
        probe.hold(count=3, label="hold")
        assert probe.ticks == 3
        probe.writer.close()
    finally:
        probe.close()
