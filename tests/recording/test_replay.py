"""Playback pacing and fidelity, against a renderer that only records what it was
asked to show -- no simulator, so this runs everywhere."""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from hitl_pmp.core.problem.human.types import (
    SAMPLED_START_RESET_DESTINATION,
    SAMPLED_START_RESET_SKILL_NAME,
)
from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.recording.replay.compositor import ReplayCompositor
from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.replay.replayer import RecordingReplayer
from hitl_pmp.recording.replay.types import ReplayStatus
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import Phase, PlainState, RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

FPS = 10


class _SpyRenderer(ReplayRenderer):
    shown: list[PlainState] = []

    def fps(self) -> int:
        return FPS

    def show(self, *, state: PlainState) -> np.ndarray:
        self.shown.append({k: list(v) for k, v in state.items()})
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def current_state(self) -> PlainState:
        return self.shown[-1]


def _record(*, path: Path, phase: Phase, ticks: int, human: bool) -> None:
    state = {"cube": [0.0]}
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="toy", feature_names={"cube": ["x"]}),
        state_provider=lambda: {k: list(v) for k, v in state.items()},
    )
    writer.begin_segment(phase=phase, checkpoint=1, task_index=0)
    writer.record_skill(name="Push", objects=("cube",), params=(1.0,), controller={})
    for i in range(1, ticks + 1):
        state["cube"] = [float(i)]
        writer.record_tick(state=state)
    writer.robot_action(label="Push(cube)", start_step=0, end_step=ticks, success=True)
    if human:
        state["cube"] = [-1.0]
        writer.human_action(skill="reset", start_step=ticks, end_step=ticks + 1, success=True)
    writer.end_segment(success=True, outcome="goal", steps=ticks)
    writer.close()


def _play(*, path: Path, tmp_path: Path) -> tuple[_SpyRenderer, int]:
    renderer = _SpyRenderer()
    video = VideoStream(output_path=tmp_path / "out.mp4", fps=FPS)
    try:
        summary = RecordingReplayer(renderer=renderer).play(
            header=RunRecordReader.header(path=path),
            entries=RunRecordReader.entries(path=path),
            video=video,
        )
    finally:
        video.close()
    assert summary.frames_written == video.frames_written
    assert summary.final_state == renderer.current_state()
    return renderer, video.frames_written


def test_practice_plays_every_fifth_tick_but_always_its_last(*, tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    _record(path=path, phase=Phase.PRACTICE, ticks=12, human=False)
    renderer, frames = _play(path=path, tmp_path=tmp_path)
    shown = [s["cube"][0] for s in renderer.shown]
    # initial state, ticks 1, 6, 11 at 5x, then the action's final tick 12.
    assert shown == [0.0, 1.0, 6.0, 11.0, 12.0]
    assert frames == len(shown)


def test_a_human_request_is_a_one_second_pause_on_the_state_it_left(*, tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    _record(path=path, phase=Phase.PRACTICE, ticks=1, human=True)
    renderer, frames = _play(path=path, tmp_path=tmp_path)
    assert renderer.shown[-1] == {"cube": [-1.0]}
    # initial + 1 tick + FPS frames held on the human's result.
    assert frames == 2 + FPS


def test_evaluation_plays_every_tick_and_holds_the_task_outcome(*, tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    _record(path=path, phase=Phase.EVALUATION, ticks=12, human=False)
    renderer, frames = _play(path=path, tmp_path=tmp_path)
    assert [s["cube"][0] for s in renderer.shown] == [float(i) for i in range(13)]
    # 13 tick frames + a one-second outcome card.
    assert frames == 13 + FPS
    assert renderer.current_state() == {"cube": [12.0]}


def _statuses(*, monkeypatch: pytest.MonkeyPatch) -> list[ReplayStatus]:
    """Every status the compositor is asked to draw, in order."""
    seen: list[ReplayStatus] = []
    original = ReplayCompositor.compose

    def spy(*, frame: np.ndarray, status: ReplayStatus) -> np.ndarray:
        seen.append(status)
        return original(frame=frame, status=status)

    monkeypatch.setattr(ReplayCompositor, "compose", staticmethod(spy))
    return seen


def test_a_full_reset_to_a_sampled_start_is_named_as_one(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The always-ask wrapper's reset rebuilds the whole scene; the banner and the
    history say so rather than printing the skill's log name."""
    path = tmp_path / "recording.jsonl"
    state = {"cube": [0.0]}
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="toy"),
        state_provider=lambda: {k: list(v) for k, v in state.items()},
    )
    writer.begin_segment(phase=Phase.PRACTICE)
    state["cube"] = [3.0]
    writer.human_action(
        skill=SAMPLED_START_RESET_SKILL_NAME,
        objects=(SAMPLED_START_RESET_DESTINATION,),
        start_step=0,
        end_step=1,
        success=True,
    )
    writer.end_segment()
    writer.close()
    seen = _statuses(monkeypatch=monkeypatch)
    _play(path=path, tmp_path=tmp_path)
    banners = [s.banner for s in seen if s.banner is not None]
    assert banners and all(b.startswith("HUMAN RESET to a sampled start state") for b in banners)
    assert seen[-1].history[-1] == "HUMAN RESET to a sampled start state"
    assert seen[-1].banner_tone == "human"
    # The status bar prefixes the actor, so it names the reset without it.
    assert (seen[-1].actor, seen[-1].skill) == ("human", "RESET to a sampled start state")


class _StoryboardSpy(_SpyRenderer):
    storyboard: ClassVar[bool] = True


def test_a_storyboard_shows_every_action_once_even_one_that_moved_nothing(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording with one state per action boundary (SweepOffShelf3D's) gives one
    frame per action. A skill that changed nothing records no state; a storyboard still
    gives it a frame, on the scene as it stands, so the history never skips one."""
    path = tmp_path / "recording.jsonl"
    state = {"cube": [0.0]}
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="toy"),
        state_provider=lambda: {k: list(v) for k, v in state.items()},
    )
    writer.begin_segment(phase=Phase.PRACTICE)
    for label, value in (("Push(cube)", 1.0), ("Pick(cube)", 1.0), ("Place(cube)", 2.0)):
        state["cube"] = [value]
        writer.robot_action(label=label, start_step=0, end_step=1, success=True)
    writer.end_segment()
    writer.close()
    seen = _statuses(monkeypatch=monkeypatch)
    renderer = _StoryboardSpy()
    video = VideoStream(output_path=tmp_path / "out.mp4", fps=FPS)
    try:
        RecordingReplayer(renderer=renderer).play(
            header=RunRecordReader.header(path=path),
            entries=RunRecordReader.entries(path=path),
            video=video,
        )
    finally:
        video.close()
    # The segment's starting state, then one frame per action.
    assert [s.history for s in seen] == [
        (),
        ("Push(cube)",),
        ("Push(cube)", "Pick(cube)"),
        ("Push(cube)", "Pick(cube)", "Place(cube)"),
    ]
    # Its pace is one frame per action, not a speed-up of real time.
    assert {s.pacing for s in seen} == {"1 frame / action"}
    assert ReplayCompositor.pacing_label(status=seen[0]) == "1 frame / action"


def test_a_recording_plays_at_its_speed_up_by_default() -> None:
    status = ReplayStatus(phase=Phase.PRACTICE, speedup=5, environment="toy")
    assert ReplayCompositor.pacing_label(status=status) == "5x speed"


def test_the_compositor_output_is_a_fixed_encoder_friendly_size() -> None:
    from hitl_pmp.recording.replay.compositor import ReplayCompositor
    from hitl_pmp.recording.replay.types import ReplayStatus

    frame = ReplayCompositor.compose(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        status=ReplayStatus(phase=Phase.PRACTICE, speedup=5, environment="toy"),
    )
    assert frame.shape == (ReplayCompositor.height, ReplayCompositor.width, 3)
    assert ReplayCompositor.width % 16 == 0
    assert ReplayCompositor.height % 16 == 0
