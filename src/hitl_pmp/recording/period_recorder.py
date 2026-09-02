from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.metrics.types import PracticeSessionEnd
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream

from .overlay import StatusBarOverlay
from .skill_chat import SkillChatOverlay
from .types import LoopPhase, LoopStatus, ResetKind


class PeriodRecorder(BaseModel):
    """Writes one separate, already-finished video file per practice period and per
    evaluation sweep, as a run happens -- so a long run's cycle 5 practice video (or
    its eval-sweep-3 video) is watchable while cycle 9 is still running, without
    waiting for the run to finish or killing it.

    **Not `LoopRecorder`.** `LoopRecorder` (`--record-full-loop`) writes ONE
    continuous file for a whole run, which is only complete -- and therefore only
    watchable -- once the run ends or is interrupted mid-file. This writes a fresh
    `VideoStream` per period/sweep and closes it the moment that period/sweep ends,
    so every finished file is a finished, playable clip the instant it appears on
    disk. The two recorders are independent and may run at once: `--record-full-loop`
    is unaffected by this, since neither drains the other's buffers.

    **Reuses `LoopRecorder`'s infrastructure, not its file layout.** Same
    `VideoStream`, `StatusBarOverlay`, `LoopStatus`/`LoopPhase`/`ResetKind` -- the
    only new thing here is *where* frames go, not how a frame is composed. See
    `LoopRecorder`'s own docstring for why frames are streamed rather than
    accumulated; the same OOM history applies here, one file at a time instead of one
    file total.

    **Practice periods get substep frames; evaluation sweeps already had them.** An
    evaluation episode already renders at physics rate when a renderer is passed to
    `Problem.run_task_episode` (see `Tossing3DProblem`'s own docstring), so
    `record_evaluation_episode` reuses that as-is. A practice period does not: `
    PracticeLoop.run` advances the environment via `Environment.take_action`
    directly, never through `run_task_episode`, so nothing has ever collected
    per-tick frames from a practice step before this class. `record_practice_step`
    closes that gap the same way `Tossing3DProblem._skill_frames` does for
    evaluation -- drain `Environment.drain_substep_frames`, and caption whatever
    substep frames come back with `Renderer.render_substep_frames`, falling back to
    one `render_frame` call when none do. Every domain but Tossing3D always takes
    that fallback, because `Environment.drain_substep_frames` is empty by default --
    so this generalizes to every domain for free, at the cost of one no-op call per
    step on a domain that has nothing finer to capture.

    `begin_practice`/`begin_evaluation` open a fresh file; `end_practice`/
    `end_evaluation` close it. A caller must pair every `begin_*` with the matching
    `end_*` -- `PracticeLoop.run` does. `close()` is the safety net for a crash
    mid-period: closing an already-closed stream is a no-op (`VideoStream.close` is
    idempotent), so a `finally` block may call it unconditionally."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    renderer: type[Renderer]
    env: Environment
    output_dir: Path
    fps: int
    num_cycles: int
    max_steps_per_interaction: int
    run_metadata: tuple[tuple[str, str], ...] = ()
    reset_hold_frames: int = 6
    outcome_hold_frames: int = 4

    practice_dirname: ClassVar[str] = "period_videos/practice"
    evaluation_dirname: ClassVar[str] = "period_videos/evaluation"

    phase: LoopPhase = LoopPhase.BASELINE_EVALUATION
    cycle_index: int = 0
    transitions: int = 0
    task: str = ""
    task_index: int | None = None
    num_tasks: int | None = None
    episode_skills: list[str] = Field(default_factory=list)
    practice_history: list[str] = Field(default_factory=list)
    action_values: dict[str, float] = Field(default_factory=dict)
    competences: dict[str, float] = Field(default_factory=dict)
    learning_rates: dict[str, float] = Field(default_factory=dict)

    video: VideoStream | None = None

    # --- practice periods ---------------------------------------------------

    def begin_practice(self, *, cycle_index: int, transitions: int, task: str) -> None:
        """Opens `<output_dir>/period_videos/practice/cycle_{cycle_index:04d}.mp4`
        and turns on substep frame capture for the length of this period -- see the
        class docstring for why that generalizes to a no-op on every domain but
        Tossing3D."""
        self.phase = LoopPhase.PRACTICE
        self.practice_history = []
        self.action_values = {}
        self.competences = {}
        self.learning_rates = {}
        self.cycle_index = cycle_index
        self.transitions = transitions
        self.task = task
        self.task_index = None
        self.num_tasks = None
        self._open(path=self.output_dir / self.practice_dirname / f"cycle_{cycle_index:04d}.mp4")
        self.env.set_substep_recording(enabled=True)

    def record_period_reset(self, *, state: State) -> None:
        self._write(
            state=state,
            status=self._status(num_steps=self.max_steps_per_interaction, reset=ResetKind.PERIOD),
            repeat=self.reset_hold_frames,
        )

    def record_practice_step(
        self, *, state: State, skill: str, step_index: int, transitions: int
    ) -> None:
        """One period step, at whatever granularity this domain's dynamics support.

        `Environment.drain_substep_frames` is drained unconditionally -- empty on
        every domain but Tossing3D, per its own default -- and the caption is
        applied to whatever comes back with `Renderer.render_substep_frames`. An
        empty result (no substeps this domain, or a Tossing3D controller that
        stepped the simulator zero times) falls back to one `render_frame` call, the
        same fallback `Tossing3DProblem._skill_frames` uses for evaluation."""
        self.transitions = transitions
        status = self._status(
            step_index=step_index, num_steps=self.max_steps_per_interaction, skill=skill
        )
        self.record_history(entry=f"{step_index + 1:02d}  {skill}")
        substeps = self.env.drain_substep_frames()
        rendered = self.renderer.render_substep_frames(
            frames=substeps, state=state, env=self.env, label=skill
        )
        if not rendered:
            rendered = [self.renderer.render_frame(state=state, env=self.env, label=skill)]
        for frame in rendered:
            self._write_frame(frame=frame, status=status)

    def record_interval_reset(self, *, state: State, step_index: int, transitions: int) -> None:
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
        self.transitions = transitions
        self.record_history(entry=f"{step_index + 1:02d}  HUMAN RESET cube + bin")
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
        self.transitions = transitions
        self.record_history(entry=f"{step_index + 1:02d}  interaction complete")
        self._write(
            state=state,
            status=self._status(
                step_index=step_index,
                num_steps=self.max_steps_per_interaction,
                event="INTERACTION COMPLETE — period ended early",
            ),
            repeat=self.reset_hold_frames,
        )

    def record_session_end(
        self, *, state: State, session_end: PracticeSessionEnd, transitions: int
    ) -> None:
        """Save a structured terminal event without inventing an extra decision."""
        self.transitions = transitions
        labels = {
            "planner_stop": "PLANNER CHOSE STOP",
            "interaction_complete": "INTERACTION COMPLETE",
            "session_action_cap": "SESSION ACTION CAP REACHED — no STOP decision",
        }
        label = labels[session_end.reason]
        self.record_history(entry=label)
        with (self.output_dir / "practice_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                LogTiming.encode(
                    record={
                        "event": "practice_session_end",
                        **session_end.model_dump(),
                        "transitions": transitions,
                    }
                )
            )
        self._write(
            state=state,
            status=self._status(
                step_index=(session_end.actions_executed - 1)
                if session_end.actions_executed
                else None,
                num_steps=session_end.action_limit,
                event=label,
            ),
            repeat=max(self.reset_hold_frames, self.fps * 2),
        )

    def end_practice(self) -> None:
        """Turns substep capture back off and closes this period's file. Mirrors
        `Tossing3DProblem.run_task_episode`'s own try/finally around
        `set_substep_recording` -- recording must not leak into whatever the loop
        does next (an evaluation sweep, which manages its own substep capture
        internally through `Problem.run_task_episode`)."""
        self.env.set_substep_recording(enabled=False)
        self._close()

    # --- evaluation sweeps ----------------------------------------------------

    def begin_evaluation(self, *, sweep_index: int, transitions: int) -> None:
        """Opens `<output_dir>/period_videos/evaluation/sweep_{sweep_index:04d}.mp4`.
        No substep-recording toggle here: `Problem.run_task_episode` already turns
        it on/off itself, scoped to one episode, whenever it is given a renderer --
        see that method's own docstring."""
        self.phase = LoopPhase.BASELINE_EVALUATION if sweep_index == 0 else LoopPhase.EVALUATION
        self.cycle_index = sweep_index
        self.transitions = transitions
        self._open(path=self.output_dir / self.evaluation_dirname / f"sweep_{sweep_index:04d}.mp4")

    def watch_policy(self, *, policy: Policy) -> Policy:
        """Identical contract to `LoopRecorder.watch_policy` -- see its docstring."""
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
        """Identical contract to `LoopRecorder.record_evaluation_episode`: `frames`
        already came from `Problem.run_task_episode` at whatever granularity that
        domain renders, so nothing here needs to know about substeps."""
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

    def end_evaluation(self) -> None:
        self._close()

    def close(self) -> None:
        """Safety net for a crash mid-period/mid-sweep: closes whatever is
        currently open, if anything is. Idempotent, like `VideoStream.close`."""
        self._close()

    # --- internals ------------------------------------------------------------

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

    def _open(self, *, path: Path) -> None:
        # A `begin_*` always follows the matching `end_*`/`close`, so `self.video` is
        # None here in practice; closing defensively rather than asserting means a
        # caller that skipped an `end_*` after a crash still gets a playable file for
        # the period that crashed, not two writers racing on one path.
        self._close()
        self.video = VideoStream(output_path=path, fps=self.fps)

    def _close(self) -> None:
        if self.video is not None:
            self.video.close()
            self.video = None

    def _write(
        self, *, state: State, status: LoopStatus, label: str | None = None, repeat: int = 1
    ) -> None:
        frame = self.renderer.render_frame(state=state, env=self.env, label=label)
        self._write_frame(frame=frame, status=status, repeat=repeat)

    def _write_frame(self, *, frame: np.ndarray, status: LoopStatus, repeat: int = 1) -> None:
        assert self.video is not None, (
            "PeriodRecorder._write_frame called with no file open -- a begin_practice/"
            "begin_evaluation call is missing before this write."
        )
        if self.phase is LoopPhase.PRACTICE:
            composed = SkillChatOverlay.compose(
                frame=frame,
                history=self.practice_history,
                values=self.action_values,
                competences=self.competences,
                learning_rates=self.learning_rates,
            )
            composed = StatusBarOverlay.compose(
                frame=composed,
                status=status.model_copy(update={"skill": None}),
                bar_position="top",
                extra_fields=self.run_metadata,
            )
        else:
            composed = StatusBarOverlay.compose(frame=frame, status=status)
        for _ in range(repeat):
            self.video.append(frame=composed)

    def record_history(self, *, entry: str) -> None:
        self.practice_history.append(entry)
        self.practice_history = self.practice_history[-SkillChatOverlay.max_entries :]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "practice_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                LogTiming.encode(
                    record={
                        "cycle": self.cycle_index,
                        "transitions": self.transitions,
                        "entry": entry,
                    }
                )
            )
