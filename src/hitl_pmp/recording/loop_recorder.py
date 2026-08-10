import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream

from .overlay import StatusBarOverlay
from .types import LoopPhase, LoopStatus, ResetKind


class LoopRecorder(BaseModel):
    """Records an entire `PracticeLoop` run -- practice periods included -- as one
    continuous, seekable video, annotated with which phase the loop is in and every
    time the environment was reset.

    This is what `--num-render-checkpoints` is not. That flag records *evaluation*
    episodes only, one clip per checkpoint, of test task 0 only: a progression of
    the finished behaviour, with the practice that produced it invisible. Practice
    periods have never been rendered at all, which is exactly where the interesting
    behaviour is -- what the method chooses to practice, how often the harness
    rescues it, and what a reset costs it. This records the whole outer loop in the
    order it happened, so the practice/evaluate rhythm and the reset structure are
    visible by scrubbing.

    **A pure observer.** Recording must never change what a run does, or the video
    would not be a record of the run anyone else gets. Nothing here draws from an
    RNG, takes an action, or decides anything; every hook is handed state the loop
    already had, renders it, and writes it out. `watch_policy` is the one wrapper,
    and it returns the wrapped policy's own `LabeledAction` object unchanged -- it
    exists only so a frame can be attributed to the skill that produced it, which
    `Problem.run_task_episode` does not report back. The one observable difference
    is that a recorded run renders every test task of every sweep rather than test
    task 0 of the checkpointed ones; that costs time, not behaviour.
    (`tests/test_method_runner.py` pins the resulting `stats.json` byte-identical.)

    **Streamed, never accumulated.** Frames go to `VideoStream` one at a time and
    are dropped immediately, for the same reason `PracticeLoop` streams its
    checkpoints: a whole run's frames is an unbounded buffer, and an unbounded
    buffer on this project has already OOM-killed a session. Peak retention is one
    composed frame.

    The running state (which phase, which cycle, which task) lives here rather than
    being passed at every call, so `PracticeLoop`'s own hook calls stay short and
    the loop stays readable -- each `begin_*` sets the context a run of subsequent
    frames shares, and each `record_*` mints the per-frame `LoopStatus` from it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    renderer: type[Renderer]
    env: Environment
    video: VideoStream
    num_cycles: int
    max_steps_per_interaction: int
    # A reset is an instantaneous state jump, so a single frame of it is easy to
    # miss entirely while scrubbing; holding the marker gives it real duration.
    reset_hold_frames: int = 6
    # Long enough to read whether the episode was solved, short enough not to pad
    # the timeline -- an outcome is a fact about the frame already on screen, not a
    # discontinuity.
    outcome_hold_frames: int = 4

    phase: LoopPhase = LoopPhase.BASELINE_EVALUATION
    cycle_index: int = 0
    transitions: int = 0
    task: str = ""
    task_index: int | None = None
    num_tasks: int | None = None
    # The skill label behind each step of the evaluation episode currently running,
    # filled in by watch_policy. Bounded by one episode's length, and replaced (not
    # appended to) at the start of every episode.
    episode_skills: list[str] = Field(default_factory=list)

    @property
    def frames_written(self) -> int:
        return self.video.frames_written

    # --- the harness reset, once per run -----------------------------------

    def record_hard_reset(self, *, state: State) -> None:
        """`problem.hard_reset()`, which happens once, before the baseline sweep,
        and never again."""
        self._write(
            state=state,
            status=self._status(reset=ResetKind.HARD),
            repeat=self.reset_hold_frames,
        )

    # --- evaluation sweeps --------------------------------------------------

    def begin_evaluation(self, *, sweep_index: int, transitions: int) -> None:
        """Sweep 0 is the baseline one, before any practice has happened -- a
        different phase colour, not just a different number, since it is the
        reference every later sweep is read against."""
        self.phase = LoopPhase.BASELINE_EVALUATION if sweep_index == 0 else LoopPhase.EVALUATION
        self.cycle_index = sweep_index
        self.transitions = transitions

    def watch_policy(self, *, policy: Policy) -> Policy:
        """The evaluation policy, wrapped so this recorder learns which skill
        produced each step of the episode about to run.

        `Problem.run_task_episode` returns frames but not the labels it drew on
        them, so without this the bar could not report SKILL during evaluation. The
        wrapper is transparent: it returns the wrapped policy's own LabeledAction,
        the same object, and consults nothing.

        A lambda adapter around a keyword-only method, per this project's
        convention for the interfaces that demand a positional callable (Policy is
        Callable[[State], LabeledAction])."""
        self.episode_skills = []
        return lambda state: self._watch(policy=policy, state=state)

    def record_evaluation_episode(
        self,
        *,
        task_index: int,
        num_tasks: int,
        task: str,
        frames: list[np.ndarray],
        solved: bool,
    ) -> None:
        """One evaluation episode, already run and rendered: its opening frame is
        the per-test-task reset (`run_task_episode` resets to the task before
        acting), then one frame per step, then a held outcome frame.

        The outcome is a repeat of the final frame rather than a new render: the
        environment has moved on by now in general, and re-rendering it would show
        a state the episode never ended in."""
        self.task_index = task_index
        self.num_tasks = num_tasks
        self.task = task
        num_steps = len(frames) - 1
        skills = self.episode_skills
        self._write_frame(
            frame=frames[0],
            status=self._status(num_steps=num_steps, reset=ResetKind.EVALUATION_TASK),
            repeat=self.reset_hold_frames,
        )
        for index, frame in enumerate(frames[1:]):
            self._write_frame(
                frame=frame,
                status=self._status(
                    step_index=index,
                    num_steps=num_steps,
                    skill=skills[index] if index < len(skills) else None,
                ),
            )
        self._write_frame(
            frame=frames[-1],
            status=self._status(
                step_index=num_steps - 1 if num_steps else None,
                num_steps=num_steps,
                event="SOLVED" if solved else "NOT SOLVED",
            ),
            repeat=self.outcome_hold_frames,
        )

    # --- practice periods ---------------------------------------------------

    def begin_practice(self, *, cycle_index: int, transitions: int, task: str) -> None:
        self.phase = LoopPhase.PRACTICE
        self.cycle_index = cycle_index
        self.transitions = transitions
        self.task = task
        self.task_index = None
        self.num_tasks = None

    def record_period_reset(self, *, state: State) -> None:
        """The reset at the top of every practice period -- the one whose necessity
        is currently under review. No step index: nothing has been executed yet."""
        self._write(
            state=state,
            status=self._status(num_steps=self.max_steps_per_interaction, reset=ResetKind.PERIOD),
            repeat=self.reset_hold_frames,
        )

    def record_practice_step(
        self, *, state: State, skill: str, step_index: int, transitions: int
    ) -> None:
        self.transitions = transitions
        self._write(
            state=state,
            status=self._status(
                step_index=step_index,
                num_steps=self.max_steps_per_interaction,
                skill=skill,
            ),
            label=skill,
        )

    def record_interval_reset(self, *, state: State, step_index: int, transitions: int) -> None:
        """A mid-period reset (`--practice-reset-interval`), attributed to the step
        it fired after."""
        self.transitions = transitions
        self._write(
            state=state,
            status=self._status(
                step_index=step_index,
                num_steps=self.max_steps_per_interaction,
                reset=ResetKind.INTERVAL,
            ),
            repeat=self.reset_hold_frames,
        )

    def record_human_reset(self, *, state: State, step_index: int, transitions: int) -> None:
        """A human was asked to reposition the robot mid-period, attributed to the step
        it fired after.

        Labelled apart from INTERVAL despite looking identical in the pixels, because it
        is the one state jump that costs the run something -- see ResetKind.HUMAN."""
        self.transitions = transitions
        self._write(
            state=state,
            status=self._status(
                step_index=step_index,
                num_steps=self.max_steps_per_interaction,
                reset=ResetKind.HUMAN,
            ),
            repeat=self.reset_hold_frames,
        )

    def record_interaction_complete(
        self, *, state: State, step_index: int, transitions: int
    ) -> None:
        """The Method raised InteractionComplete: the period ended before its
        budget ran out. Held like a reset, because it is equally invisible in the
        pixels and equally worth seeing -- the transitions this run was charged
        stop here rather than at max_steps_per_interaction."""
        self.transitions = transitions
        self._write(
            state=state,
            status=self._status(
                step_index=step_index,
                num_steps=self.max_steps_per_interaction,
                event="INTERACTION COMPLETE — period ended early",
            ),
            repeat=self.reset_hold_frames,
        )

    def close(self) -> None:
        self.video.close()

    # --- internals ----------------------------------------------------------

    def _watch(self, *, policy: Policy, state: State) -> LabeledAction:
        labeled_action = policy(state)
        self.episode_skills.append(labeled_action.label)
        return labeled_action

    def _status(
        self,
        *,
        step_index: int | None = None,
        num_steps: int | None = None,
        skill: str | None = None,
        reset: ResetKind | None = None,
        event: str | None = None,
    ) -> LoopStatus:
        return LoopStatus(
            phase=self.phase,
            transitions=self.transitions,
            cycle_index=self.cycle_index,
            num_cycles=self.num_cycles,
            step_index=step_index,
            num_steps=num_steps,
            task_index=self.task_index,
            num_tasks=self.num_tasks,
            task=self.task,
            skill=skill,
            reset=reset,
            event=event,
        )

    def _write(
        self, *, state: State, status: LoopStatus, label: str | None = None, repeat: int = 1
    ) -> None:
        frame = self.renderer.render_frame(state=state, env=self.env, label=label)
        self._write_frame(frame=frame, status=status, repeat=repeat)

    def _write_frame(self, *, frame: np.ndarray, status: LoopStatus, repeat: int = 1) -> None:
        composed = StatusBarOverlay.compose(frame=frame, status=status)
        for _ in range(repeat):
            self.video.append(frame=composed)
