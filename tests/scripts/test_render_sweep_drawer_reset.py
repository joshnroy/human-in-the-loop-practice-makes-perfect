"""The replay renderer's bookkeeping: which step a tick belongs to, how fast it plays."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)


def _module():
    import scripts.render_sweep_drawer_reset as module

    return module


def _cycle() -> dict:
    return {
        "steps": [
            {"name": "sweep", "phase": "attempt_1", "ticks": 3},
            {"name": "pick_cube_2", "phase": "reset", "ticks": 0},
            {"name": "wiggle_drawer", "phase": "reset", "ticks": 2},
        ]
    }


def test_every_tick_is_captioned_with_the_step_that_was_running() -> None:
    steps = _module().Caption.step_at(cycle=_cycle())
    assert [steps[t]["name"] for t in (1, 2, 3)] == ["sweep"] * 3
    assert [steps[t]["name"] for t in (4, 5)] == ["wiggle_drawer"] * 2


def test_a_step_refused_before_it_moved_owns_no_tick() -> None:
    steps = _module().Caption.step_at(cycle=_cycle())
    assert "pick_cube_2" not in {s["name"] for s in steps.values()}
    assert 0 not in steps
    assert 6 not in steps


def test_the_caption_reports_counts_not_percentages() -> None:
    caption = _module().Caption
    head, tail = caption.lines(
        seed=6, tick=125, step=_cycle()["steps"][2], drawer=0.246, piled=3, speed=4.0
    )
    assert head == "robot self-reset:  wiggle drawer"
    assert tail.startswith("seed 6 ")
    assert "3/5" in tail
    assert "%" not in tail
    assert "12.5 s" in tail
    assert "24.6 cm" in tail
    assert "4x" in tail


def test_one_frame_per_tick_at_40_fps_plays_at_four_times_speed() -> None:
    """The scene is controlled, and logged, at 10 Hz."""
    video = _module().ResetVideo
    every = video(run_dir=Path("run"), output=Path("out.mp4"), stride=1, fps=40)
    assert every.speed() == pytest.approx(4.0)
    other = video(run_dir=Path("run"), output=Path("out.mp4"), stride=2, fps=40)
    assert other.speed() == pytest.approx(8.0)


def test_frames_have_even_dimensions() -> None:
    width, height = _module().Layout.size()
    assert width % 2 == 0
    assert height % 2 == 0


def test_a_composed_frame_holds_both_panels_and_the_caption_bar() -> None:
    layout = _module().Layout
    h = layout.HEIGHT
    wide, narrow = (v.width for v in layout.VIEWS)
    left = np.full((h, wide, 3), 200, dtype=np.uint8)
    right = np.full((h, narrow, 3), 90, dtype=np.uint8)
    frame = layout.compose(panels=[left, right], lines=("a step", "numbers"))
    assert frame.shape == (h + layout.BAR, wide + narrow, 3)
    assert tuple(frame[h // 2, wide // 2]) == (200, 200, 200)
    assert tuple(frame[h // 2, wide + narrow // 2]) == (90, 90, 90)
    assert tuple(frame[h + layout.BAR - 2, wide + narrow - 2]) == layout.BACKGROUND


def test_a_panels_title_is_legible_whatever_the_panel_shows() -> None:
    """Light text straight onto the frame vanished against a white floor."""
    layout = _module().Layout
    h = layout.HEIGHT
    panels = [np.full((h, v.width, 3), 255, dtype=np.uint8) for v in layout.VIEWS]
    frame = layout.compose(panels=panels, lines=("", ""))
    corner = frame[8:40, 10:200].reshape(-1, 3)
    dark = (corner.sum(axis=1) < 3 * 40).sum()
    light = (corner.sum(axis=1) > 3 * 200).sum()
    assert dark > 1000, "the title has a dark ground"
    assert light > 100, "and light lettering on it"


def _tiny_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="room"><geom size="1 1 0.1" type="box"/></body>
            <body name="drawer"><joint name="drawer_joint" type="slide"/>
              <geom size="0.1 0.1 0.1" type="box"/></body>
            <body name="cube"><joint name="cube_joint" type="free"/>
              <geom size="0.01 0.01 0.01" type="box"/></body>
          </worldbody>
        </mujoco>
        """
    )


def test_logged_joints_are_carried_over_by_name_not_by_position() -> None:
    """Here the log has the cube first and the drawer after it; the scene the reverse."""
    header = {
        "joints": [
            {"name": "cube_joint", "adr": 0, "n": 7},
            {"name": "drawer_joint", "adr": 7, "n": 1},
        ]
    }
    mapping = _module().ReplayLog.mapping(header=header, model=_tiny_model())
    assert mapping == [(0, 1, 7), (7, 0, 1)]


def test_a_log_of_a_joint_the_scene_lacks_is_refused() -> None:
    header = {"joints": [{"name": "kitchen_island_drawer_s0c1_joint", "adr": 0, "n": 1}]}
    with pytest.raises(ValueError, match="no joint"):
        _module().ReplayLog.mapping(header=header, model=_tiny_model())


def test_a_joint_that_changed_kind_is_refused() -> None:
    header = {"joints": [{"name": "cube_joint", "adr": 0, "n": 1}]}
    with pytest.raises(ValueError, match="different kind"):
        _module().ReplayLog.mapping(header=header, model=_tiny_model())


def test_nvenc_and_x264_get_their_own_rate_control_flags() -> None:
    video = _module().ResetVideo
    nvenc = video(run_dir=Path("run"), output=Path("out.mp4")).encoder()
    assert "h264_nvenc" in nvenc
    assert "-cq" in nvenc
    assert "-crf" not in nvenc
    x264 = video(run_dir=Path("run"), output=Path("out.mp4"), codec="libx264").encoder()
    assert "-crf" in x264
    assert "-cq" not in x264


def test_the_log_is_read_at_the_stride_asked(*, tmp_path: Path) -> None:
    log = tmp_path / "replay_log.jsonl"
    records = [{"kind": "header", "seed": 6, "nq": 2, "joints": []}]
    records += [{"kind": "tick", "t": t, "qpos": [0.0, float(t)]} for t in range(7)]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    replay = _module().ReplayLog
    assert replay.header(path=log)["seed"] == 6
    assert [r["t"] for r in replay.ticks(path=log, stride=1)] == list(range(7))
    assert [r["t"] for r in replay.ticks(path=log, stride=2)] == [0, 2, 4, 6]


def test_a_file_that_does_not_start_with_a_header_is_refused(*, tmp_path: Path) -> None:
    log = tmp_path / "replay_log.jsonl"
    log.write_text(json.dumps({"kind": "tick", "t": 0, "qpos": []}) + "\n")
    with pytest.raises(ValueError, match="header"):
        _module().ReplayLog.header(path=log)


def test_a_cube_is_in_the_pile_only_on_the_counter_inside_the_region() -> None:
    video = _module().ResetVideo
    assert video.in_pile(position=np.array([0.76, -0.08, 0.47]))
    assert not video.in_pile(position=np.array([0.826, -0.08, 0.47]))
    assert not video.in_pile(position=np.array([0.76, -0.08, 0.2373]))
