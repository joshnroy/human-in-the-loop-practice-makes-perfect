"""`scripts/validate_sweep_off_shelf3d.py`: argument parsing, the stacked two-camera
captions, the contact sheet and the ffmpeg writer run with no simulator. The episodes
are minutes long and are run by hand, not here; the skills they call are smoke-tested
in `test_sweep_off_shelf3d_skills.py`."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from pydantic import ValidationError

from scripts.validate_sweep_off_shelf3d import (
    DEFAULT_HEIGHT,
    DEFAULT_SPEED,
    EPISODES,
    FRAME_SIZE,
    PANEL_SIZE,
    Captions,
    ValidationPlan,
    Validator,
    VideoWriter,
)


def _frame(*, shade: int) -> np.ndarray:
    width, height = FRAME_SIZE
    return np.full((height, width, 3), shade, dtype=np.uint8)


def test_parse_args_defaults_to_every_episode_in_lab2_with_video(*, tmp_path: Path) -> None:
    plan = Validator.parse_args(argv=["--output-dir", str(tmp_path)])
    assert plan == ValidationPlan(output_dir=tmp_path)
    assert plan.episodes == EPISODES
    assert plan.scene == "mimiclabs-lab2"
    assert plan.video
    assert (plan.height, plan.speed) == (DEFAULT_HEIGHT, DEFAULT_SPEED)


def test_parse_args_takes_a_grid_cell_headless(*, tmp_path: Path) -> None:
    plan = Validator.parse_args(
        argv=[
            "--output-dir",
            str(tmp_path),
            "--episodes",
            "sweep_cubes_into_box",
            "--seed",
            "3",
            "--height",
            "0.03",
            "--speed",
            "0.2",
            "--suffix",
            "_grid",
            "--no-video",
        ]
    )
    assert plan.episodes == ("sweep_cubes_into_box",)
    assert (plan.seed, plan.height, plan.speed, plan.suffix) == (3, 0.03, 0.2, "_grid")
    assert not plan.video


def test_parse_args_rejects_a_sweep_setting_outside_its_range(*, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Validator.parse_args(argv=["--output-dir", str(tmp_path), "--speed", "0.5"])


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
