"""`PeriodRecorder` writes one already-finished video file per practice period and
per evaluation sweep, unlike `LoopRecorder` (one continuous file for a whole run) --
see `period_recorder.py`'s own module docstring for why that is a different
mechanism rather than a variant of the existing one. These tests use the same fake
`Environment`/`Renderer`/capturing-`VideoStream` scaffolding as
`test_loop_recorder.py`, plus a second fake pair that DOES report substep frames, to
pin the one genuinely new behaviour: a practice step's video comes from
`Environment.drain_substep_frames` when a domain has any, and falls back to a single
`Renderer.render_frame` call -- every non-Tossing3D domain today -- when it does
not."""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import Field

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream
from hitl_pmp.recording import period_recorder as period_recorder_module
from hitl_pmp.recording.period_recorder import PeriodRecorder
from hitl_pmp.recording.skill_chat import SkillChatOverlay
from hitl_pmp.recording.types import LoopPhase, LoopStatus, ResetKind

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)

_NUM_CYCLES = 4
_MAX_STEPS = 20
_FPS = 4


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _FakeEnv(Environment):
    """No substep frames -- `Environment`'s own no-op defaults, exactly like every
    domain but Tossing3D."""

    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def noop_action(self) -> Action:
        raise NotImplementedError

    def hard_reset(self) -> None:
        self.set_state(state=_state(x=0.0))


class _SubstepEnv(_FakeEnv):
    """A Tossing3D-shaped fake: `set_substep_recording` toggles a flag, and once on,
    `drain_substep_frames` returns a fixed-size burst and clears it -- standing in
    for `KinderBackend.set_substep_recording`/`drain_substep_frames`."""

    recording: bool = False
    pending: int = 0
    toggles: ClassVar[list[bool]] = []

    def set_substep_recording(self, *, enabled: bool) -> None:
        self.recording = enabled
        _SubstepEnv.toggles.append(enabled)
        self.pending = 3 if enabled else 0

    def drain_substep_frames(self) -> list[np.ndarray]:
        if not self.recording or self.pending == 0:
            return []
        frames = [np.full((4, 8, 3), i, dtype=np.uint8) for i in range(self.pending)]
        self.pending = 0
        return frames


class _SpyRenderer(Renderer):
    calls: ClassVar[list[tuple[float, str | None]]] = []

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del env
        value = int(state[_OBJ][0])
        _SpyRenderer.calls.append((float(value), label))
        return np.full((16, 32, 3), value, dtype=np.uint8)


class _SubstepRenderer(_SpyRenderer):
    """Mirrors `Tossing3DRenderer.render_substep_frames`: an empty input gives an
    empty output, a non-empty one is captioned and returned as-is (the caption
    itself is irrelevant to what this file checks)."""

    @staticmethod
    def render_substep_frames(
        *,
        frames: Sequence[np.ndarray],
        state: State,
        env: Environment,
        label: str | None = None,
    ) -> list[np.ndarray]:
        del state, env, label
        return list(frames)


class _CapturingStream(VideoStream):
    """Keeps every appended frame instead of encoding it, and self-registers into
    `_opened` by path -- `PeriodRecorder` constructs a fresh `VideoStream` per
    period/sweep internally, so a test cannot inject one the way
    `test_loop_recorder.py` does and must instead intercept the class."""

    captured: list[np.ndarray] = Field(default_factory=list)
    closed: bool = False

    def model_post_init(self, __context: object) -> None:
        _opened[str(self.output_path)] = self

    def append(self, *, frame: np.ndarray) -> None:
        self.captured.append(frame)
        self.frames_written += 1

    def close(self) -> None:
        self.closed = True


_opened: dict[str, _CapturingStream] = {}


@pytest.fixture(autouse=True)
def _capture_video_streams(*, monkeypatch: pytest.MonkeyPatch) -> None:
    _opened.clear()
    _SpyRenderer.calls = []
    _SubstepEnv.toggles = []
    monkeypatch.setattr(period_recorder_module, "VideoStream", _CapturingStream)


def _build(*, env: Environment, renderer: type[Renderer], output_dir: Path) -> PeriodRecorder:
    return PeriodRecorder(
        renderer=renderer,
        env=env,
        output_dir=output_dir,
        fps=_FPS,
        num_cycles=_NUM_CYCLES,
        max_steps_per_interaction=_MAX_STEPS,
    )


# --- file layout -------------------------------------------------------------


def test_practice_and_evaluation_write_separate_files_named_by_index(*, tmp_path: Path) -> None:
    env = _FakeEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SpyRenderer, output_dir=tmp_path)

    recorder.begin_practice(cycle_index=3, transitions=0, task="goal")
    recorder.record_period_reset(state=_state(x=1.0))
    recorder.end_practice()

    recorder.begin_evaluation(sweep_index=2, transitions=10)
    recorder.record_evaluation_episode(
        task_index=0,
        num_tasks=1,
        task="goal",
        frames=[np.zeros((4, 8, 3), dtype=np.uint8)],
        solved=True,
    )
    recorder.end_evaluation()

    practice_path = tmp_path / "period_videos" / "practice" / "cycle_0003.mp4"
    evaluation_path = tmp_path / "period_videos" / "evaluation" / "sweep_0002.mp4"
    assert str(practice_path) in _opened
    assert str(evaluation_path) in _opened
    assert _opened[str(practice_path)].closed
    assert _opened[str(evaluation_path)].closed


def test_a_new_period_closes_the_previous_files_stream(*, tmp_path: Path) -> None:
    """`end_practice`/`end_evaluation` are the normal way a file finishes, but
    `_open` also closes defensively -- see its own docstring for why a caller that
    skipped one must not leak a dangling writer into the next file."""
    env = _FakeEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SpyRenderer, output_dir=tmp_path)

    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    first = _opened[str(tmp_path / "period_videos" / "practice" / "cycle_0000.mp4")]
    assert not first.closed
    recorder.begin_practice(cycle_index=1, transitions=0, task="goal")
    assert first.closed


# --- substep capture, the one genuinely new behaviour -------------------------


def test_a_practice_step_on_a_domain_with_no_substeps_writes_exactly_one_frame(
    *, tmp_path: Path
) -> None:
    env = _FakeEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SpyRenderer, output_dir=tmp_path)
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_practice_step(state=_state(x=9.0), skill="Pick", step_index=0, transitions=1)
    recorder.end_practice()

    stream = _opened[str(tmp_path / "period_videos" / "practice" / "cycle_0000.mp4")]
    assert len(stream.captured) == 1, "no substep frames exist -- must fall back to render_frame"


def test_a_practice_step_on_a_domain_with_substeps_writes_every_substep_frame(
    *, tmp_path: Path
) -> None:
    """The property this class exists to fix: a Tossing3D-shaped domain's practice
    video must show every physics tick a skill produced, not one frame per skill --
    see the class docstring's "too fast" problem."""
    env = _SubstepEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SubstepRenderer, output_dir=tmp_path)
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_practice_step(state=_state(x=9.0), skill="Pick", step_index=0, transitions=1)
    recorder.end_practice()

    stream = _opened[str(tmp_path / "period_videos" / "practice" / "cycle_0000.mp4")]
    assert len(stream.captured) == 3, "the fake backend's substep burst must all land in the clip"


def test_substep_recording_is_scoped_to_exactly_one_practice_period(*, tmp_path: Path) -> None:
    env = _SubstepEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SubstepRenderer, output_dir=tmp_path)

    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    assert env.recording is True
    recorder.end_practice()
    assert env.recording is False
    assert _SubstepEnv.toggles == [True, False]


def test_an_evaluation_sweep_does_not_toggle_substep_recording_itself(*, tmp_path: Path) -> None:
    """`Problem.run_task_episode` owns substep recording for evaluation, scoped to
    one episode at a time -- `PeriodRecorder` must not also toggle it, or the two
    would race over the same flag."""
    env = _SubstepEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SubstepRenderer, output_dir=tmp_path)

    recorder.begin_evaluation(sweep_index=0, transitions=0)
    recorder.end_evaluation()
    assert _SubstepEnv.toggles == []


# --- reuse of the shared recording infrastructure ------------------------------


def test_a_period_reset_frame_is_composed_with_the_shared_status_bar_overlay(
    *, tmp_path: Path
) -> None:
    """Pins that this class draws through the same `StatusBarOverlay`/`LoopStatus`
    machinery as `LoopRecorder`, rather than a parallel implementation of the
    caption itself -- see the class docstring's "reuses infrastructure" claim."""
    from hitl_pmp.recording.overlay import StatusBarOverlay

    env = _FakeEnv()
    env.hard_reset()
    recorder = _build(env=env, renderer=_SpyRenderer, output_dir=tmp_path)
    recorder.begin_practice(cycle_index=1, transitions=20, task="InBin(cube, bin)")
    recorder.record_period_reset(state=_state(x=3.0))
    recorder.end_practice()

    stream = _opened[str(tmp_path / "period_videos" / "practice" / "cycle_0001.mp4")]
    _SpyRenderer.calls.pop()  # the probe render below must not pollute the fixture's own log
    expected_frame = _SpyRenderer.render_frame(state=_state(x=3.0), env=env, label=None)
    _SpyRenderer.calls.pop()
    expected = StatusBarOverlay.compose(
        frame=expected_frame,
        status=LoopStatus(
            phase=LoopPhase.PRACTICE,
            cycle_index=1,
            num_cycles=_NUM_CYCLES,
            num_steps=_MAX_STEPS,
            transitions=20,
            task="InBin(cube, bin)",
            reset=ResetKind.PERIOD,
        ),
    )
    assert len(stream.captured) == recorder.reset_hold_frames
    expected = SkillChatOverlay.compose(frame=expected, history=[], competences={})
    assert np.array_equal(stream.captured[0], expected)
