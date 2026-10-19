from collections.abc import Iterable
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.problem.human.types import SAMPLED_START_RESET_SKILL_NAME
from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.recording.run_record.types import (
    ActionEntry,
    CycleStart,
    Phase,
    RecordEntry,
    RecordingHeader,
    SegmentEnd,
    SegmentStart,
    TickEntry,
)

from .compositor import ReplayCompositor
from .renderer import ReplayRenderer
from .types import ReplayStatus, ReplaySummary


class RecordingReplayer(BaseModel):
    """Plays a recording into a video by drawing its recorded states, in order.

    Pacing: practice shows every `practice_speedup`-th tick (and always an action's
    last), so it runs that many times faster than real time; each human skill is a
    `pause_seconds` hold on the state the human left, labelled with the skill; an
    evaluation shows every tick, then holds its task number and outcome for
    `pause_seconds`. `play` may be called repeatedly on one stream -- practice, then an
    evaluation -- since the canvas size is fixed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    renderer: ReplayRenderer
    practice_speedup: int = 5
    pause_seconds: float = 1.0

    _status: ReplayStatus | None = PrivateAttr(default=None)
    _history: list[str] = PrivateAttr(default_factory=list)
    _skipped: TickEntry | None = PrivateAttr(default=None)
    _counter: int = PrivateAttr(default=0)
    _scene: np.ndarray | None = PrivateAttr(default=None)
    _video: VideoStream | None = PrivateAttr(default=None)
    _frames: int = PrivateAttr(default=0)
    _shown: int = PrivateAttr(default=0)
    _recorded: int = PrivateAttr(default=0)
    _action: ActionEntry | None = PrivateAttr(default=None)
    _action_drawn: bool = PrivateAttr(default=False)

    # The full reset rebuilds the whole scene, so it is named for what it does.
    SAMPLED_START_RESET_LABEL: ClassVar[str] = "HUMAN RESET to a sampled start state"

    def play(
        self, *, header: RecordingHeader, entries: Iterable[RecordEntry], video: VideoStream
    ) -> ReplaySummary:
        self._video = video
        self._frames = self._shown = self._recorded = 0
        self._status = ReplayStatus(
            environment=header.environment,
            method=header.method,
            phase=Phase.PRACTICE,
            speedup=self.practice_speedup,
            pacing="1 frame / action" if self.renderer.storyboard else None,
        )
        for entry in entries:
            if isinstance(entry, TickEntry):
                self._tick(tick=entry)
            elif isinstance(entry, ActionEntry):
                self._flush()
                self._begin_action(action=entry)
            elif isinstance(entry, SegmentStart):
                self._flush()
                self._begin_segment(segment=entry)
            elif isinstance(entry, CycleStart):
                self._flush()
                self._update(cycle=entry.cycle)
            elif isinstance(entry, SegmentEnd):
                self._flush()
                self._end_segment(end=entry)
        self._flush()
        return ReplaySummary(
            frames_written=self._frames,
            ticks_shown=self._shown,
            ticks_recorded=self._recorded,
            final_state=self.renderer.current_state() if self._shown else None,
        )

    def _begin_segment(self, *, segment: SegmentStart) -> None:
        self._action = None
        self._counter = 0
        self._update(
            phase=segment.phase,
            speedup=self.practice_speedup if segment.phase is Phase.PRACTICE else 1,
            checkpoint=segment.checkpoint,
            task_index=segment.task_index,
            actor=None,
            skill=None,
            start_step=None,
            end_step=None,
        )
        if segment.phase is Phase.EVALUATION:
            self._history = []

    def _begin_action(self, *, action: ActionEntry) -> None:
        self._action = action
        self._action_drawn = False
        self._counter = 0
        name = action.label or (
            f"{action.skill}({', '.join(action.objects)})" if action.objects else action.skill
        )
        skill = action.label or action.skill
        if action.actor == "human" and action.skill == SAMPLED_START_RESET_SKILL_NAME:
            name = RecordingReplayer.SAMPLED_START_RESET_LABEL
            skill = name.removeprefix("HUMAN ")
        elif action.actor == "human":
            name = f"HUMAN {name}"
        if action.success is False:
            name += " ✗"
        self._history.append(name)
        self._update(
            actor=action.actor,
            skill=skill,
            start_step=action.start_step,
            end_step=action.end_step,
            cycle=action.cycle if action.cycle is not None else self._require().cycle,
            history=tuple(self._history),
        )

    def _tick(self, *, tick: TickEntry) -> None:
        self._recorded += 1
        action = self._action
        if action is not None and action.actor == "human" and tick.seq == action.seq:
            moved = ", ".join(sorted(action.moved_objects or {})) or "nothing"
            title = (
                RecordingReplayer.SAMPLED_START_RESET_LABEL
                if action.skill == SAMPLED_START_RESET_SKILL_NAME
                else f"HUMAN: {action.skill}"
            )
            self._show(
                tick=tick,
                banner=f"{title}\nmoved {moved}",
                tone="human",
                repeat=self._hold(),
            )
            return
        if self._counter % self._require().speedup == 0:
            self._show(tick=tick)
            self._skipped = None
        else:
            self._skipped = tick
        self._counter += 1

    def _end_segment(self, *, end: SegmentEnd) -> None:
        if end.phase is not Phase.EVALUATION or self._scene is None:
            return
        verdict = "SOLVED" if end.success else "NOT SOLVED"
        detail = f"{end.outcome or 'ended'} after {end.steps} steps"
        self._append(
            frame=self._scene,
            banner=f"TASK {end.task_index}: {verdict}\n{detail}",
            tone="success" if end.success else "failure",
            repeat=self._hold(),
        )

    def _flush(self) -> None:
        """Show the last skipped tick, so every action ends on its recorded end state."""
        if self._skipped is not None:
            tick, self._skipped = self._skipped, None
            self._show(tick=tick)
        if (
            self.renderer.storyboard
            and self._action is not None
            and not self._action_drawn
            and self._scene is not None
        ):
            self._action_drawn = True
            self._append(frame=self._scene, banner=None, tone=None, repeat=1)

    def _show(
        self,
        *,
        tick: TickEntry,
        banner: str | None = None,
        tone: str | None = None,
        repeat: int = 1,
    ) -> None:
        self._scene = self.renderer.show(state=tick.state)
        self._action_drawn = True
        self._shown += 1
        self._append(frame=self._scene, banner=banner, tone=tone, repeat=repeat)

    def _append(
        self, *, frame: np.ndarray, banner: str | None, tone: str | None, repeat: int
    ) -> None:
        assert self._video is not None
        status = self._require().model_copy(update={"banner": banner, "banner_tone": tone})
        composed = ReplayCompositor.compose(frame=frame, status=status)
        for _ in range(repeat):
            self._video.append(frame=composed)
            self._frames += 1

    def _hold(self) -> int:
        return max(1, round(self.pause_seconds * self.renderer.fps()))

    def _update(self, **fields: object) -> None:
        self._status = self._require().model_copy(update=fields)

    def _require(self) -> ReplayStatus:
        assert self._status is not None
        return self._status
