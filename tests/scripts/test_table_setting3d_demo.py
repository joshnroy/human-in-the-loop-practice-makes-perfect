"""Replayable action logs and provenance-checked video selection."""

import argparse
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("kinder")

from scripts.table_setting3d_comparison import TableSettingComparison
from scripts.table_setting3d_demo import TableSettingDemo, TableSettingVideo


@pytest.mark.parametrize("competence,choice", [(1.0, "human"), (0.0, "robot")])
def test_cost_only_demo_is_repeatable_and_accounts_for_every_action(
    *, tmp_path, competence, choice
):
    args = argparse.Namespace(
        output_dir=tmp_path,
        human_competence=competence,
        seed=0,
        cycles=4,
        calibration_resets=3,
        speedup=12.0,
        no_video=True,
    )
    first = TableSettingDemo.run(args=args)
    second = TableSettingDemo.run(args=args)
    assert first == second == json.loads((tmp_path / "results.json").read_text())
    assert first["cycles"][-1]["choice"] == choice
    assert first["video"] is None
    assert not (tmp_path / "episode.mp4").exists()
    events = [event for cycle in first["cycles"] for event in cycle["events"]]
    assert sum(event["cost"] for event in events) == pytest.approx(first["elapsed_seconds"])
    assert sum(event["action"] == 8 for event in events) == first["human_interventions"]
    for before, after in zip(events, events[1:], strict=False):
        assert before["state_after"] == after["state_before"]
    for cycle in first["cycles"]:
        assert sum(event["cost"] for event in cycle["events"][4:]) == pytest.approx(
            cycle["actual_reset_seconds"]
        )
    assert all(len(value) == 64 for value in first["source_sha256"].values())


def test_frame_ranges_are_exact_and_writer_closes_once():
    class Sink:
        count = 0
        closes = 0

        def append_data(self, frame):  # noqa: PLR0917 (imageio API)
            del frame
            self.count += 1

        def close(self):
            self.closes += 1

    sink = Sink()
    writer = TableSettingVideo(writer=sink)
    writer.append_frame(frame=np.zeros((16, 16, 3), dtype=np.uint8))
    writer.append_frame(frame=np.zeros((16, 16, 3), dtype=np.uint8))
    assert writer.frame_count == sink.count == 2
    writer.close()
    writer.close()
    assert sink.closes == 1
    with pytest.raises(RuntimeError):
        writer.append_frame(frame=np.zeros((16, 16, 3), dtype=np.uint8))


def test_comparison_rejects_stale_video(*, tmp_path):
    report = {
        "format_version": 1,
        "video": {"filename": "episode.mp4", "sha256": hashlib.sha256(b"original").hexdigest()},
    }
    (tmp_path / "results.json").write_text(json.dumps(report))
    (tmp_path / "episode.mp4").write_bytes(b"different")
    with pytest.raises(ValueError, match="does not match"):
        TableSettingComparison.load_run(directory=tmp_path)


def test_comparison_rejects_cost_only_run(*, tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"format_version": 1, "video": None}))
    with pytest.raises(ValueError, match="rerun the demo"):
        TableSettingComparison.load_run(directory=tmp_path)


def test_clip_ranges_use_recorded_decision_and_spill_boundaries():
    report = {
        "video": {"frames": 100},
        "cycles": [
            {
                "phase": "Calibration",
                "choice": "human",
                "reset_video_frame_start": 20,
                "events": [{"human_success": False, "video_frame_end": 30}],
            },
            {
                "phase": "Adaptive choice",
                "choice": "robot",
                "video_frame_start": 50,
                "video_frame_end": 100,
                "events": [],
            },
        ],
    }
    clips = TableSettingComparison.clip_ranges(report=report, include_spill=True)
    assert [(clip["start"], clip["end"]) for clip in clips] == [(20, 30), (50, 100)]
    report["cycles"][1]["video_frame_end"] = 101
    with pytest.raises(ValueError, match="Invalid recorded frame range"):
        TableSettingComparison.clip_ranges(report=report, include_spill=True)
