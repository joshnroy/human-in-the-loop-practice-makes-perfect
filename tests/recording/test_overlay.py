import itertools

import numpy as np

from hitl_pmp.recording.overlay import StatusBarOverlay
from hitl_pmp.recording.types import LoopPhase, LoopStatus, ResetKind


def _frame(*, height: int = 32, width: int = 64, value: int = 90) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _practice_status(**overrides: object) -> LoopStatus:
    defaults: dict[str, object] = {
        "phase": LoopPhase.PRACTICE,
        "cycle_index": 2,
        "num_cycles": 10,
        "step_index": 46,
        "num_steps": 100,
        "transitions": 347,
        "task": "InBin(recycling, recycling_bin)",
        "skill": "Throw",
    }
    return LoopStatus(**{**defaults, **overrides})  # type: ignore[arg-type]


def _evaluation_status(**overrides: object) -> LoopStatus:
    defaults: dict[str, object] = {
        "phase": LoopPhase.EVALUATION,
        "cycle_index": 3,
        "num_cycles": 10,
        "task_index": 1,
        "num_tasks": 3,
        "step_index": 2,
        "num_steps": 9,
        "transitions": 60,
        "task": "InBin(trash, trash_bin)",
        "skill": "MoveRoom",
    }
    return LoopStatus(**{**defaults, **overrides})  # type: ignore[arg-type]


# --- the fields the bar shows, per phase -----------------------------------


def test_practice_bar_reports_phase_cycle_step_transitions_task_and_skill() -> None:
    fields = dict(StatusBarOverlay.format_fields(status=_practice_status()))
    assert fields["PHASE"] == "PRACTICE"
    # 0-indexed internally, 1-indexed for a human reading the bar.
    assert fields["CYCLE"] == "3/10"
    assert fields["STEP"] == "47/100"
    assert fields["TRANSITIONS"] == "347"
    assert fields["TASK"] == "InBin(recycling, recycling_bin)"
    assert fields["SKILL"] == "Throw"


def test_evaluation_bar_reports_the_sweep_and_which_test_task_is_running() -> None:
    fields = dict(StatusBarOverlay.format_fields(status=_evaluation_status()))
    assert fields["PHASE"] == "EVALUATION"
    # A sweep index, not a cycle: sweep 0 runs before any practice, so it is NOT
    # shifted to 1-indexing the way a practice cycle is.
    assert fields["SWEEP"] == "3/10"
    assert fields["TEST TASK"] == "2/3"
    assert fields["STEP"] == "3/9"
    assert fields["GOAL"] == "InBin(trash, trash_bin)"
    assert fields["SKILL"] == "MoveRoom"
    assert "CYCLE" not in fields


def test_baseline_evaluation_bar_names_itself_as_the_pre_practice_sweep() -> None:
    fields = dict(
        StatusBarOverlay.format_fields(
            status=_evaluation_status(phase=LoopPhase.BASELINE_EVALUATION, cycle_index=0)
        )
    )
    assert fields["PHASE"] == "BASELINE EVAL"
    assert fields["SWEEP"] == "0/10"


def test_bar_omits_the_skill_field_before_any_action_has_been_taken() -> None:
    fields = dict(StatusBarOverlay.format_fields(status=_practice_status(skill=None)))
    assert "SKILL" not in fields


def test_bar_reports_a_reset_as_its_own_field_naming_the_kind() -> None:
    fields = dict(StatusBarOverlay.format_fields(status=_practice_status(reset=ResetKind.PERIOD)))
    assert fields["RESET"] == StatusBarOverlay.reset_labels[ResetKind.PERIOD]


def test_bar_reports_an_event_when_one_is_attached() -> None:
    fields = dict(
        StatusBarOverlay.format_fields(status=_practice_status(event="INTERACTION COMPLETE"))
    )
    assert fields["EVENT"] == "INTERACTION COMPLETE"


# --- phase and reset colours are distinct ----------------------------------


def test_every_phase_has_its_own_colour() -> None:
    colors = [StatusBarOverlay.phase_colors[phase] for phase in LoopPhase]
    assert len(set(colors)) == len(list(LoopPhase))


def test_every_reset_kind_has_its_own_colour_and_label() -> None:
    colors = [StatusBarOverlay.reset_colors[kind] for kind in ResetKind]
    labels = [StatusBarOverlay.reset_labels[kind] for kind in ResetKind]
    assert len(set(colors)) == len(list(ResetKind))
    assert len(set(labels)) == len(list(ResetKind))


def test_the_reset_kinds_cover_every_reset_the_loop_performs() -> None:
    assert {kind.value for kind in ResetKind} == {
        "hard",
        "period",
        "interval",
        "human",
        "evaluation_task",
    }


# --- composition: shape, purity, and that the pixels actually change --------


def test_compose_appends_a_bar_below_the_environment_frame() -> None:
    frame = _frame()
    composed = StatusBarOverlay.compose(frame=frame, status=_practice_status())
    assert composed.shape[0] >= frame.shape[0] + StatusBarOverlay.bar_height
    assert composed.shape[1] >= frame.shape[1]
    assert composed.dtype == np.uint8


def test_compose_returns_dimensions_a_video_encoder_accepts() -> None:
    """imageio-ffmpeg silently resizes (and warns) unless both dimensions are a
    multiple of its macro block size, which would make the bar's text blurry."""
    composed = StatusBarOverlay.compose(
        frame=_frame(height=30, width=50), status=_practice_status()
    )
    assert composed.shape[0] % 16 == 0
    assert composed.shape[1] % 16 == 0


def test_compose_does_not_mutate_the_frame_it_is_given() -> None:
    frame = _frame()
    before = frame.copy()
    StatusBarOverlay.compose(frame=frame, status=_practice_status(reset=ResetKind.HARD))
    assert np.array_equal(frame, before)


def test_compose_returns_a_frame_that_shares_no_memory_with_its_input() -> None:
    frame = _frame()
    composed = StatusBarOverlay.compose(frame=frame, status=_practice_status())
    assert not np.shares_memory(composed, frame)


def test_compose_leaves_the_environment_pixels_intact_when_nothing_is_reset() -> None:
    frame = _frame()
    composed = StatusBarOverlay.compose(frame=frame, status=_practice_status())
    assert np.array_equal(composed[: frame.shape[0], : frame.shape[1]], frame)


def test_each_phase_paints_the_bar_a_visibly_different_colour() -> None:
    """Non-vacuity: proves the per-phase colour reaches real pixels rather than
    only existing in the phase_colors mapping."""
    frame = _frame()
    bars = {
        phase: StatusBarOverlay.compose(frame=frame, status=_practice_status(phase=phase))[
            frame.shape[0] :
        ]
        for phase in LoopPhase
    }
    for left, right in itertools.combinations(LoopPhase, 2):
        assert not np.array_equal(bars[left], bars[right])


def test_each_reset_kind_paints_a_visibly_different_marker() -> None:
    frame = _frame()
    composed = {
        kind: StatusBarOverlay.compose(frame=frame, status=_practice_status(reset=kind))
        for kind in ResetKind
    }
    for left, right in itertools.combinations(ResetKind, 2):
        assert not np.array_equal(composed[left], composed[right])


def test_a_reset_marks_the_environment_frame_itself_not_only_the_bar() -> None:
    """A reset is an instantaneous state jump, so the marker has to be visible
    where the viewer is looking -- over the environment drawing, not only in the
    status bar underneath it."""
    frame = _frame()
    plain = StatusBarOverlay.compose(frame=frame, status=_practice_status())
    marked = StatusBarOverlay.compose(frame=frame, status=_practice_status(reset=ResetKind.HARD))
    height = frame.shape[0]
    assert not np.array_equal(plain[:height], marked[:height])


def test_changing_a_single_field_changes_the_bar_pixels() -> None:
    """Non-vacuity for the text rendering itself: if the fields were never drawn,
    two statuses differing only in TRANSITIONS would compose identically."""
    frame = _frame()
    height = frame.shape[0]
    low = StatusBarOverlay.compose(frame=frame, status=_practice_status(transitions=1))[height:]
    high = StatusBarOverlay.compose(frame=frame, status=_practice_status(transitions=999))[height:]
    assert not np.array_equal(low, high)


def test_an_event_changes_the_bar_pixels() -> None:
    frame = _frame()
    height = frame.shape[0]
    plain = StatusBarOverlay.compose(frame=frame, status=_practice_status())[height:]
    with_event = StatusBarOverlay.compose(frame=frame, status=_practice_status(event="SOLVED"))[
        height:
    ]
    assert not np.array_equal(plain, with_event)
