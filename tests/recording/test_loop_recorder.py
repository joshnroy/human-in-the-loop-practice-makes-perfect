from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream
from hitl_pmp.recording.loop_recorder import LoopRecorder
from hitl_pmp.recording.overlay import StatusBarOverlay
from hitl_pmp.recording.types import LoopPhase, LoopStatus, ResetKind

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="thing", type=_BLOCK)

_NUM_CYCLES = 4
_MAX_STEPS = 20


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _FakeEnv(Environment):
    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def hard_reset(self) -> None:
        self.set_state(state=_state(x=0.0))


class _SpyRenderer(Renderer):
    """Encodes the state it was handed into every pixel, and records the label it
    was passed -- so a test can tell *which* state and *which* skill label a
    recorded frame came from, rather than only counting frames."""

    calls: ClassVar[list[tuple[float, str | None]]] = []

    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del env
        value = int(state[_OBJ][0])
        _SpyRenderer.calls.append((float(value), label))
        return np.full((16, 32, 3), value, dtype=np.uint8)


class _CapturingStream(VideoStream):
    """Keeps every appended frame instead of encoding it, so assertions can look at
    real composed pixels without an ffmpeg round-trip losing them to h264."""

    captured: list[np.ndarray] = Field(default_factory=list)
    closed: bool = False

    def append(self, *, frame: np.ndarray) -> None:
        self.captured.append(frame)
        self.frames_written += 1

    def close(self) -> None:
        self.closed = True


def _build() -> tuple[LoopRecorder, _CapturingStream]:
    _SpyRenderer.calls = []
    env = _FakeEnv()
    env.hard_reset()
    stream = _CapturingStream(output_path=Path("unused.mp4"), fps=4)
    recorder = LoopRecorder(
        renderer=_SpyRenderer,
        env=env,
        video=stream,
        num_cycles=_NUM_CYCLES,
        max_steps_per_interaction=_MAX_STEPS,
    )
    return recorder, stream


def _expected(*, value: float, status: LoopStatus, label: str | None = None) -> np.ndarray:
    frame = _SpyRenderer.render_frame(state=_state(x=value), env=_FakeEnv(), label=label)
    # The probe render above must not pollute what the test asserts about.
    _SpyRenderer.calls.pop()
    return StatusBarOverlay.compose(frame=frame, status=status)


# --- the four reset kinds --------------------------------------------------


def test_hard_reset_is_held_for_several_frames_and_marked_as_the_harness_reset() -> None:
    recorder, stream = _build()
    recorder.record_hard_reset(state=_state(x=7.0))
    expected = _expected(
        value=7.0,
        status=LoopStatus(
            phase=LoopPhase.BASELINE_EVALUATION,
            num_cycles=_NUM_CYCLES,
            transitions=0,
            reset=ResetKind.HARD,
        ),
    )
    assert len(stream.captured) == recorder.reset_hold_frames
    assert recorder.reset_hold_frames > 1  # a one-frame marker is easy to miss
    assert all(np.array_equal(frame, expected) for frame in stream.captured)


def test_period_reset_is_marked_as_the_top_of_a_practice_period() -> None:
    recorder, stream = _build()
    recorder.begin_practice(cycle_index=1, transitions=20, task="InBin(trash, trash_bin)")
    recorder.record_period_reset(state=_state(x=3.0))
    expected = _expected(
        value=3.0,
        status=LoopStatus(
            phase=LoopPhase.PRACTICE,
            cycle_index=1,
            num_cycles=_NUM_CYCLES,
            num_steps=_MAX_STEPS,
            transitions=20,
            task="InBin(trash, trash_bin)",
            reset=ResetKind.PERIOD,
        ),
    )
    assert len(stream.captured) == recorder.reset_hold_frames
    assert np.array_equal(stream.captured[0], expected)


def test_interval_reset_is_marked_as_a_mid_period_reset_at_the_step_it_fired() -> None:
    recorder, stream = _build()
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_interval_reset(state=_state(x=5.0), step_index=6, transitions=7)
    expected = _expected(
        value=5.0,
        status=LoopStatus(
            phase=LoopPhase.PRACTICE,
            cycle_index=0,
            num_cycles=_NUM_CYCLES,
            step_index=6,
            num_steps=_MAX_STEPS,
            transitions=7,
            task="goal",
            reset=ResetKind.INTERVAL,
        ),
    )
    assert len(stream.captured) == recorder.reset_hold_frames
    assert np.array_equal(stream.captured[0], expected)


def test_every_evaluation_episode_opens_with_its_own_per_test_task_reset() -> None:
    recorder, stream = _build()
    recorder.begin_evaluation(sweep_index=2, transitions=40)
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (1, 2)]
    recorder.record_evaluation_episode(
        task_index=1, num_tasks=3, task="goal", frames=frames, solved=True
    )
    expected = StatusBarOverlay.compose(
        frame=frames[0],
        status=LoopStatus(
            phase=LoopPhase.EVALUATION,
            cycle_index=2,
            num_cycles=_NUM_CYCLES,
            num_steps=1,
            task_index=1,
            num_tasks=3,
            transitions=40,
            task="goal",
            reset=ResetKind.EVALUATION_TASK,
        ),
    )
    assert all(
        np.array_equal(frame, expected) for frame in stream.captured[: recorder.reset_hold_frames]
    )


def test_the_four_reset_kinds_are_all_visually_distinct_as_recorded() -> None:
    """The distinction between reset kinds is the whole point of the recording, so
    it is checked on the frames that actually reach the video, not only on the
    colour mapping."""
    marked: dict[ResetKind, np.ndarray] = {}

    recorder, stream = _build()
    recorder.record_hard_reset(state=_state(x=1.0))
    marked[ResetKind.HARD] = stream.captured[0]

    recorder, stream = _build()
    recorder.begin_practice(cycle_index=0, transitions=0, task="g")
    recorder.record_period_reset(state=_state(x=1.0))
    marked[ResetKind.PERIOD] = stream.captured[0]

    recorder, stream = _build()
    recorder.begin_practice(cycle_index=0, transitions=0, task="g")
    recorder.record_interval_reset(state=_state(x=1.0), step_index=0, transitions=0)
    marked[ResetKind.INTERVAL] = stream.captured[0]

    recorder, stream = _build()
    recorder.begin_evaluation(sweep_index=0, transitions=0)
    recorder.record_evaluation_episode(
        task_index=0,
        num_tasks=1,
        task="g",
        frames=[np.full((16, 32, 3), 1, dtype=np.uint8)],
        solved=True,
    )
    marked[ResetKind.EVALUATION_TASK] = stream.captured[0]

    keys = list(marked)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            assert not np.array_equal(marked[left], marked[right])


# --- practice periods ------------------------------------------------------


def test_each_practice_step_records_exactly_one_frame_carrying_its_skill() -> None:
    recorder, stream = _build()
    recorder.begin_practice(cycle_index=3, transitions=60, task="goal")
    recorder.record_practice_step(state=_state(x=9.0), skill="Throw", step_index=4, transitions=61)
    expected = _expected(
        value=9.0,
        label="Throw",
        status=LoopStatus(
            phase=LoopPhase.PRACTICE,
            cycle_index=3,
            num_cycles=_NUM_CYCLES,
            step_index=4,
            num_steps=_MAX_STEPS,
            transitions=61,
            task="goal",
            skill="Throw",
        ),
    )
    assert len(stream.captured) == 1
    assert np.array_equal(stream.captured[0], expected)


def test_practice_step_hands_the_skill_label_to_the_domain_renderer() -> None:
    recorder, _ = _build()
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_practice_step(state=_state(x=1.0), skill="Pickup", step_index=0, transitions=1)
    assert _SpyRenderer.calls == [(1.0, "Pickup")]


def test_an_early_ended_period_records_the_interaction_complete_event() -> None:
    recorder, stream = _build()
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_interaction_complete(state=_state(x=2.0), step_index=5, transitions=5)
    plain = _expected(
        value=2.0,
        status=LoopStatus(
            phase=LoopPhase.PRACTICE,
            cycle_index=0,
            num_cycles=_NUM_CYCLES,
            step_index=5,
            num_steps=_MAX_STEPS,
            transitions=5,
            task="goal",
        ),
    )
    assert len(stream.captured) > 1  # held, like a reset -- it is a real event
    assert not np.array_equal(stream.captured[0], plain)


# --- evaluation episodes ---------------------------------------------------


def test_evaluation_episode_records_a_reset_hold_one_frame_per_step_and_an_outcome_hold() -> None:
    recorder, stream = _build()
    recorder.begin_evaluation(sweep_index=1, transitions=20)
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (1, 2, 3, 4)]
    recorder.record_evaluation_episode(
        task_index=0, num_tasks=2, task="goal", frames=frames, solved=True
    )
    assert len(stream.captured) == (
        recorder.reset_hold_frames + (len(frames) - 1) + recorder.outcome_hold_frames
    )


def test_evaluation_outcome_distinguishes_a_solved_episode_from_a_failed_one() -> None:
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (1, 2)]
    outcomes: dict[bool, np.ndarray] = {}
    for solved in (True, False):
        recorder, stream = _build()
        recorder.begin_evaluation(sweep_index=1, transitions=20)
        recorder.record_evaluation_episode(
            task_index=0, num_tasks=2, task="goal", frames=frames, solved=solved
        )
        outcomes[solved] = stream.captured[-1]
    assert not np.array_equal(outcomes[True], outcomes[False])


def test_an_episode_whose_goal_already_held_still_records_a_reset_and_an_outcome() -> None:
    recorder, stream = _build()
    recorder.begin_evaluation(sweep_index=0, transitions=0)
    recorder.record_evaluation_episode(
        task_index=0,
        num_tasks=1,
        task="goal",
        frames=[np.full((16, 32, 3), 5, dtype=np.uint8)],
        solved=True,
    )
    assert len(stream.captured) == recorder.reset_hold_frames + recorder.outcome_hold_frames


# --- watching the evaluation policy ----------------------------------------


def test_watch_policy_returns_the_wrapped_policys_own_labeled_action_unchanged() -> None:
    recorder, _ = _build()
    original = LabeledAction(action=np.array([1.0]), label="Throw")
    watched = recorder.watch_policy(policy=lambda state: original)
    assert watched(_state(x=0.0)) is original


def test_watch_policy_labels_each_evaluation_step_with_the_skill_that_produced_it() -> None:
    recorder, stream = _build()
    recorder.begin_evaluation(sweep_index=0, transitions=0)
    watched = recorder.watch_policy(
        policy=lambda state: LabeledAction(action=np.zeros(1), label="MoveRoom")
    )
    watched(_state(x=0.0))
    frames = [np.full((16, 32, 3), value, dtype=np.uint8) for value in (1, 2)]
    recorder.record_evaluation_episode(
        task_index=0, num_tasks=1, task="goal", frames=frames, solved=True
    )
    expected = StatusBarOverlay.compose(
        frame=frames[1],
        status=LoopStatus(
            phase=LoopPhase.BASELINE_EVALUATION,
            cycle_index=0,
            num_cycles=_NUM_CYCLES,
            step_index=0,
            num_steps=1,
            task_index=0,
            num_tasks=1,
            transitions=0,
            task="goal",
            skill="MoveRoom",
        ),
    )
    assert np.array_equal(stream.captured[recorder.reset_hold_frames], expected)


def test_watch_policy_starts_a_fresh_label_buffer_per_episode() -> None:
    recorder, _ = _build()
    recorder.begin_evaluation(sweep_index=0, transitions=0)
    first = recorder.watch_policy(policy=lambda state: LabeledAction(action=np.zeros(1), label="A"))
    first(_state(x=0.0))
    second = recorder.watch_policy(
        policy=lambda state: LabeledAction(action=np.zeros(1), label="B")
    )
    second(_state(x=0.0))
    assert recorder.episode_skills == ["B"]


# --- bookkeeping -----------------------------------------------------------


def test_frames_written_counts_every_frame_handed_to_the_stream() -> None:
    recorder, stream = _build()
    recorder.record_hard_reset(state=_state(x=0.0))
    recorder.begin_practice(cycle_index=0, transitions=0, task="goal")
    recorder.record_practice_step(state=_state(x=1.0), skill="s", step_index=0, transitions=1)
    assert recorder.frames_written == len(stream.captured)


def test_close_closes_the_underlying_video_stream() -> None:
    recorder, stream = _build()
    recorder.record_hard_reset(state=_state(x=0.0))
    recorder.close()
    assert stream.closed
