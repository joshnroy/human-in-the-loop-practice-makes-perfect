from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class LoopStatus(BaseModel):
    """Where in `practice_loop.py`'s outer loop one recorded frame was taken --
    everything the status bar over that frame reports, and nothing else.

    Frozen: it is an immutable snapshot of one instant, not a mutable cursor the
    recorder edits in place. The recorder carries the running state (which phase,
    which cycle, how many transitions so far) and mints one of these per frame, so
    a frame's annotation cannot be retroactively changed by whatever the loop did
    next.

    Deliberately plain strings and ints rather than references to `Task`/
    `GroundSkill`/`State`: a status is what a *viewer* needs, and holding the real
    objects would keep a whole run's tasks and states alive for the length of the
    recording.

    Every field except `phase`/`transitions` is optional because the phases genuinely
    differ in what exists: an evaluation episode has a test-task index and no cycle
    step budget, a practice period has the reverse, and the frame at a reset has no
    skill to attribute (nothing was executed -- that is the whole point of it)."""

    model_config = ConfigDict(frozen=True)

    phase: LoopPhase
    # The x-axis every learning curve in this project uses, so a viewer can line a
    # moment in the video up against a point on a plot.
    transitions: int
    # 0-based: the practice cycle, or (during evaluation) the sweep index, where
    # sweep 0 is the one that runs before any practice at all.
    cycle_index: int = 0
    num_cycles: int = 0
    step_index: int | None = None
    num_steps: int | None = None
    task_index: int | None = None
    num_tasks: int | None = None
    # Goal.describe() of whatever task is being pursued; "" when none applies.
    task: str = ""
    # LabeledAction.label of the action that produced this frame, if any.
    skill: str | None = None
    reset: ResetKind | None = None
    # A one-off occurrence worth seeing: an episode's outcome, or a period that
    # ended early. Distinct from `reset`, which has its own fixed vocabulary.
    event: str | None = None


class LoopPhase(str, Enum):
    """The three stretches of the outer loop, in the order they first occur. A
    str-Enum so a phase survives a round trip through plain text unchanged.

    BASELINE_EVALUATION is split out from EVALUATION rather than being "sweep 0"
    because it is the one sweep measuring a policy that has practiced nothing --
    the reference point every later sweep is read against, and worth being able to
    find at a glance while scrubbing."""

    BASELINE_EVALUATION = "baseline_evaluation"
    PRACTICE = "practice"
    EVALUATION = "evaluation"


class ResetKind(str, Enum):
    """The four distinct ways the environment's state jumps in a run, which a
    viewer otherwise cannot tell apart from each other or from a skill's effect.

    Keeping them distinct is the point of the recording: HARD happens once, per
    run, before anything else; PERIOD is the per-cycle reset whose necessity is
    currently under review (`practice_loop.py`'s own docstring argues both sides);
    INTERVAL only exists when `--practice-reset-interval` is set, and is the knob
    that decouples reset frequency from refit frequency; EVALUATION_TASK is the
    reset inside `Problem.run_task_episode`, once per test task, and is not part of
    the practice protocol at all."""

    HARD = "hard"
    PERIOD = "period"
    INTERVAL = "interval"
    EVALUATION_TASK = "evaluation_task"
