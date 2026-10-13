"""`scripts/validate_sweep_into_hole3d.py`: argument parsing, captions, the contact
sheet and the ffmpeg writer run on synthetic frames with no simulator; the smoke
test builds the probe in lab2, fires both human-reset prototypes and checks what
they restore, and skips where `kinder` or the MimicLabs assets are missing. The
full episodes are minutes long and are run by hand, not here."""

import importlib.util
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from scripts.validate_sweep_into_hole3d import (
    CUBES,
    EPISODES,
    FRAME_SIZE,
    Captions,
    HoleProbe,
    HumanResets,
    ValidationPlan,
    Validator,
    VideoWriter,
)

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="kindergarden is not installed"
)


def _frame(*, shade: int) -> np.ndarray:
    width, height = FRAME_SIZE
    return np.full((height, width, 3), shade, dtype=np.uint8)


def test_parse_args_defaults_to_both_episodes_in_lab2(*, tmp_path: Path) -> None:
    plan = Validator.parse_args(argv=["--output-dir", str(tmp_path)])
    assert plan == ValidationPlan(output_dir=tmp_path)
    assert plan.episodes == EPISODES
    assert plan.scene == "mimiclabs-lab2"
    narrowed = Validator.parse_args(
        argv=["--output-dir", "x", "--episodes", "drive_in_then_reset_robot", "--tick-limit", "50"]
    )
    assert narrowed.episodes == ("drive_in_then_reset_robot",)
    assert narrowed.tick_limit == 50


def test_annotate_keeps_the_frame_size_and_draws_header_and_banner() -> None:
    plain = Captions.annotate(frame=_frame(shade=120), header="h", banner=None)
    bannered = Captions.annotate(frame=_frame(shade=120), header="h", banner="HUMAN reset fires")
    assert plain.shape == bannered.shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    assert plain[:26].mean() < 120  # dark header strip
    assert bannered[-44:, :, 0].mean() > plain[-44:, :, 0].mean()  # red banner


def test_contact_sheet_lays_out_captioned_keyframes(*, tmp_path: Path) -> None:
    keyframes = [(f"step {i}", _frame(shade=40 * i)) for i in range(6)]
    path = Captions.contact_sheet(keyframes=keyframes, path=tmp_path / "sheet.png")
    sheet = iio.imread(path)
    assert sheet.shape == (2 * (FRAME_SIZE[1] + 30), 4 * FRAME_SIZE[0], 3)


def test_video_writer_encodes_a_short_clip(*, tmp_path: Path) -> None:
    writer = VideoWriter(path=tmp_path / "clip.mp4", fps=10)
    for shade in range(0, 250, 25):
        writer.write(frame=_frame(shade=shade))
    writer.close()
    assert writer.encoder in {"h264_nvenc", "libx264"}
    assert writer.frames == 10
    assert (tmp_path / "clip.mp4").stat().st_size > 0


@needs_kinder
def test_human_reset_prototypes_restore_cubes_robot_and_broom(*, tmp_path: Path) -> None:
    if not Validator.import_kinder() or not _assets_present():
        pytest.skip("MimicLabs assets are not downloaded")
    plan = ValidationPlan(output_dir=tmp_path, tick_limit=200)
    probe = HoleProbe(plan=plan, episode="drive_in_then_reset_robot")
    try:
        start_base = probe.data.qpos[probe.ba].copy()
        for index, name in enumerate(CUBES):
            probe.core.get_object(name).set_pose([0.85 + 0.08 * index, 0.0, -0.07], [1, 0, 0, 0])
        assert probe.in_hole() == 5
        HumanResets.reset_cubes(core=probe.core)
        assert probe.in_hole() == 0
        # The teleport places cubes just above the floor; the drop is physics.
        for _ in range(2000):
            probe.sim.step()
        for name in CUBES:
            x, y, z = probe.core.get_object(name).get_pose()[0]
            assert 0.45 <= x <= 0.6 and abs(y) <= 0.12 + 0.05 and abs(z - 0.01) < 3e-3
        probe.move_base(target=[0.3, 0.0, 0.0], label="move")
        probe.core.get_object("wiper_0").set_pose([0.9, 0.4, 0.3], [1, 0, 0, 0])
        HumanResets.reset_robot(
            core=probe.core, home_qpos=probe.home_qpos, robot_qpos_indices=probe.robot_qpos_indices
        )
        assert np.allclose(probe.data.qpos[probe.ba], start_base, atol=1e-3)
        wiper = probe.core.get_object("wiper_0").get_pose()[0]
        assert abs(wiper[0] - 0.25) < 0.06 and abs(wiper[1] + 0.5) < 0.06 and wiper[2] < 0.03
        probe.writer.close()
    finally:
        probe.close()


def _assets_present() -> bool:
    import kinder

    root = Path(kinder.__path__[0]) / "envs" / "dynamic3d" / "models" / "assets"
    return (root / "mimiclabs_scenes" / "meshes").is_dir()
