from pathlib import Path

import numpy as np
import pytest

from analysis.render_recording import RecordingVideo
from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.run_record.types import (
    ActionEntry,
    Phase,
    PlainState,
    RecordingHeader,
    SegmentStart,
)
from hitl_pmp.recording.run_record.writer import RunRecordWriter

FIXTURE = Path(__file__).parents[1] / "recording" / "fixtures" / "full_agentic_run"


class _Blank(ReplayRenderer):
    state: PlainState = {}

    def fps(self) -> int:
        return 4

    def show(self, *, state: PlainState) -> np.ndarray:
        self.state = state
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def current_state(self) -> PlainState:
        return self.state


def _write(*, path: Path, phase: Phase, checkpoint: int | None, scene: dict | None = None) -> None:
    state = {"cube": [0.0]}
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="toy", scene=scene or {}),
        state_provider=lambda: dict(state),
    )
    writer.begin_segment(phase=phase, checkpoint=checkpoint, task_index=0)
    writer.record_skill(name="Push", objects=(), params=(), controller={})
    state["cube"] = [1.0]
    writer.record_tick(state=state)
    writer.robot_action(label="Push()", start_step=0, end_step=1, success=True)
    writer.end_segment(success=True, outcome="goal", steps=1)
    writer.close()


def test_a_run_plays_practice_then_only_its_final_evaluation(*, tmp_path: Path) -> None:
    _write(path=tmp_path / "recording.jsonl", phase=Phase.PRACTICE, checkpoint=None)
    for checkpoint in (0, 1):
        _write(
            path=tmp_path / "evaluations" / f"{checkpoint:04d}" / "recording.jsonl",
            phase=Phase.EVALUATION,
            checkpoint=checkpoint,
        )
    sources = RecordingVideo.sources(run=tmp_path, checkpoint=None, full_agentic=False)
    segments = [
        entry for _, entries in sources for entry in entries if isinstance(entry, SegmentStart)
    ]
    assert [(s.phase, s.checkpoint) for s in segments] == [
        (Phase.PRACTICE, None),
        (Phase.EVALUATION, 1),
    ]

    summaries = RecordingVideo.render(
        run=tmp_path,
        output=tmp_path / "out.mp4",
        checkpoint=None,
        full_agentic=False,
        renderer_factory=lambda header: _Blank(),
    )
    assert [s.final_state for s in summaries] == [{"cube": [1.0]}, {"cube": [1.0]}]
    assert (tmp_path / "out.mp4").stat().st_size > 0


def test_a_full_agentic_run_loads_through_its_adapter() -> None:
    sources = RecordingVideo.sources(run=FIXTURE, checkpoint=None, full_agentic=True)
    header, _ = sources[0]
    assert header.environment == "sweepoffshelf3d"
    actions = [e for _, entries in sources for e in entries if isinstance(e, ActionEntry)]
    assert {a.phase for a in actions} == {Phase.PRACTICE, Phase.EVALUATION}


def test_an_environment_without_a_replay_renderer_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="toy"):
        RecordingVideo.default_renderer(
            header=RecordingHeader(environment="toy"), realistic_background=True
        )


@pytest.mark.parametrize("realistic", [True, False])
def test_the_shelf_replays_through_its_storyboard_renderer(*, realistic: bool) -> None:
    """Built lazily: choosing the renderer builds no scene."""
    from hitl_pmp.environments.sweepoffshelf3d.recording import SweepOffShelf3DReplayRenderer

    renderer = RecordingVideo.default_renderer(
        header=RecordingHeader(environment="sweepoffshelf3d", scene={"scene_bg": False}),
        realistic_background=realistic,
    )
    assert isinstance(renderer, SweepOffShelf3DReplayRenderer)
    assert renderer.scene_bg is (True if realistic else None)
    assert renderer.storyboard


class _Fast(_Blank):
    def fps(self) -> int:
        return 12


def test_the_video_runs_at_the_fastest_sources_rate(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storyboard practice and a full-motion evaluation share one video, at the
    motion's rate, so the motion is not slowed down to the storyboard's."""
    import analysis.render_recording as module

    _write(path=tmp_path / "recording.jsonl", phase=Phase.PRACTICE, checkpoint=None)
    _write(
        path=tmp_path / "evaluations" / "0000" / "recording.jsonl",
        phase=Phase.EVALUATION,
        checkpoint=0,
        scene={"motion": {"fps": 12}},
    )
    rates: list[int] = []

    class _Stream(module.VideoStream):
        def model_post_init(self, __context: object) -> None:
            rates.append(self.fps)

    monkeypatch.setattr(module, "VideoStream", _Stream)
    summaries = RecordingVideo.render(
        run=tmp_path,
        output=tmp_path / "out.mp4",
        checkpoint=None,
        full_agentic=False,
        renderer_factory=lambda header: _Fast() if header.scene else _Blank(),
    )
    assert rates == [12]
    assert len(summaries) == 2
