"""SweepOffShelf3D's storyboard replay: a drawing scene posed from each recorded
action-boundary state, and a practice recording played one frame per action."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.problem.human.types import (
    SAMPLED_START_RESET_DESTINATION,
    SAMPLED_START_RESET_SKILL_NAME,
)
from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.sweepoffshelf3d.recording import SweepOffShelf3DReplayRenderer
from hitl_pmp.recording.replay.compositor import ReplayCompositor
from hitl_pmp.recording.replay.replayer import RecordingReplayer
from hitl_pmp.recording.replay.types import ReplayStatus
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import Phase, PlainState, RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

# Module-scoped fixture: keep it on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
    pytest.mark.xdist_group("sweepoffshelf3d_test_recording"),
]


@pytest.fixture(name="renderer", scope="module")
def _renderer():
    renderer = SweepOffShelf3DReplayRenderer(scene={"canonical_seed": 0}, scene_bg=False)
    try:
        yield renderer
    finally:
        renderer.close()


def _moved(*, start: PlainState) -> PlainState:
    """A scene the robot changed: base driven and turned, gripper closed, a cube on the
    floor, the wiper set down elsewhere and turned."""
    state = {name: list(values) for name, values in start.items()}
    state["robot"] = [0.4, -0.2, 0.5, 1.0]
    state["cube_0"] = [1.0, 0.3, 0.01]
    state["wiper_0"] = [0.9, -0.4, 0.02, 0.3]
    return state


def test_show_poses_the_scene_in_the_recorded_state(*, renderer) -> None:
    start = renderer.current_state()
    state = _moved(start=start)
    frame = renderer.show(state=state)
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.std() > 0
    back = renderer.current_state()
    for name in ("robot", "cube_0", "cube_3", "wiper_0", "box_0"):
        assert np.allclose(back[name], state[name], atol=1e-6), name
    renderer.show(state=start)
    assert np.allclose(renderer.current_state()["robot"], start["robot"], atol=1e-6)


def test_a_practice_recording_plays_one_frame_per_action_and_banners_the_reset(
    *, renderer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = renderer.current_state()
    state = {name: list(values) for name, values in start.items()}
    path = tmp_path / "recording.jsonl"
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="sweepoffshelf3d", scene={"canonical_seed": 0}),
        state_provider=lambda: {name: list(values) for name, values in state.items()},
    )
    writer.begin_segment(phase=Phase.PRACTICE)
    state.update(_moved(start=start))
    writer.robot_action(label="PickWiper(robot, wiper_0)", start_step=0, end_step=5, success=True)
    writer.robot_action(label="PickWiper(robot, wiper_0)", start_step=5, end_step=5, success=False)
    state.clear()
    state.update({name: list(values) for name, values in start.items()})
    writer.human_action(
        skill=SAMPLED_START_RESET_SKILL_NAME,
        objects=(SAMPLED_START_RESET_DESTINATION,),
        start_step=5,
        end_step=6,
        success=True,
    )
    writer.end_segment()
    writer.close()

    seen: list[ReplayStatus] = []
    original = ReplayCompositor.compose

    def spy(*, frame: np.ndarray, status: ReplayStatus) -> np.ndarray:
        seen.append(status)
        return original(frame=frame, status=status)

    monkeypatch.setattr(ReplayCompositor, "compose", staticmethod(spy))
    video = VideoStream(output_path=tmp_path / "out.mp4", fps=renderer.fps())
    try:
        RecordingReplayer(renderer=renderer).play(
            header=RunRecordReader.header(path=path),
            entries=RunRecordReader.entries(path=path),
            video=video,
        )
    finally:
        video.close()
    # The starting scene, one frame for each of the two robot actions (the second moved
    # nothing), then the human's reset held for its pause.
    assert [len(s.history) for s in seen[:3]] == [0, 1, 2]
    reset = [s for s in seen if s.banner is not None]
    assert reset and len(reset) == len(seen) - 3
    assert reset[0].banner.startswith("HUMAN RESET to a sampled start state")
    assert reset[-1].history[-1] == "HUMAN RESET to a sampled start state"
    assert (tmp_path / "out.mp4").stat().st_size > 0
