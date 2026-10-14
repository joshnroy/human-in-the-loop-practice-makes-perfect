from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.log_timing import LogTiming

from .sink import SkillTickSink
from .types import (
    ActionEntry,
    CycleStart,
    Phase,
    PlainState,
    RecordingHeader,
    SegmentEnd,
    SegmentStart,
    TickEntry,
)


class _PendingSkill(BaseModel):
    name: str
    objects: list[str]
    params: list[float]
    controller: dict[str, Any]
    ticks: list[TickEntry] = []


class RunRecordWriter(SkillTickSink):
    """Streams one run's recording to `output_path`.

    Lines are written as they become final. A skill's ticks are held only until the
    caller reports that skill's outcome, so its `action` line precedes its ticks and
    carries the outcome; a few hundred states per skill is the most ever held.

    `state_provider` reads the live environment's full state. It supplies a segment's
    starting state, the state a human skill leaves behind, and -- if the environment
    moved after its last reported tick -- the settled state at the end of a robot
    skill, so the recording's state at every action boundary is the live one."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path | None
    header: RecordingHeader
    state_provider: Callable[[], PlainState] | None = None
    # Header fields only known once the environment has built its scene (feature names,
    # static objects), which can be after this writer exists; read once, when the header
    # is written.
    scene_description_provider: Callable[[], dict[str, Any]] | None = None
    task_seed_provider: Callable[[Any], int | None] | None = None

    _file: TextIO | None = PrivateAttr(default=None)
    _header_written: bool = PrivateAttr(default=False)
    _seq: int = PrivateAttr(default=0)
    _segment: SegmentStart | None = PrivateAttr(default=None)
    _pending: _PendingSkill | None = PrivateAttr(default=None)
    _last_state: PlainState | None = PrivateAttr(default=None)

    @staticmethod
    def disabled() -> "RunRecordWriter":
        return RunRecordWriter(output_path=None, header=RecordingHeader(environment="none"))

    def begin_segment(
        self,
        *,
        phase: Phase,
        checkpoint: int | None = None,
        task_index: int | None = None,
        task_seed: int | None = None,
        scene_seed: int | None = None,
    ) -> None:
        self._flush_pending()
        self._segment = SegmentStart(
            phase=phase,
            checkpoint=checkpoint,
            task_index=task_index,
            task_seed=task_seed,
            scene_seed=scene_seed,
        )
        self._write(entry=self._segment)
        if self.state_provider is not None and self.output_path is not None:
            self._write_tick(tick=TickEntry(seq=None, state=self.state_provider()))

    def begin_cycle(
        self,
        *,
        cycle: int,
        task: Any = None,
        task_seed: int | None = None,
        start_step: int | None = None,
    ) -> None:
        self._flush_pending()
        if task_seed is None and task is not None and self.task_seed_provider is not None:
            task_seed = self.task_seed_provider(task)
        self._write(entry=CycleStart(cycle=cycle, task_seed=task_seed, start_step=start_step))

    def record_skill(
        self,
        *,
        name: str,
        objects: Sequence[str],
        params: Sequence[float],
        controller: Mapping[str, Any],
    ) -> None:
        self._flush_pending()
        if self.output_path is None:
            return
        self._pending = _PendingSkill(
            name=name,
            objects=list(objects),
            params=[float(p) for p in params],
            controller=dict(controller),
        )

    def record_tick(
        self, *, state: Mapping[str, Sequence[float]], diagnostics: dict[str, Any] | None = None
    ) -> None:
        if self.output_path is None:
            return
        tick = TickEntry(
            state={name: [float(v) for v in values] for name, values in state.items()},
            diagnostics=diagnostics,
        )
        if self._pending is None:
            self._write_tick(tick=tick)
        else:
            self._pending.ticks.append(tick)

    def robot_action(
        self,
        *,
        label: str,
        start_step: int | None,
        end_step: int | None,
        success: bool | None,
        outcome: str | None = None,
        cost: float | None = None,
        cycle: int | None = None,
    ) -> None:
        if self.output_path is None:
            return
        pending = self._pending or _PendingSkill(
            name=label.split("(", 1)[0],
            objects=[
                part.strip()
                for part in label.partition("(")[2].partition(")")[0].split(",")
                if part.strip()
            ],
            params=[],
            controller={},
        )
        self._pending = None
        if self.state_provider is not None:
            settled = self.state_provider()
            last = pending.ticks[-1].state if pending.ticks else self._last_state
            if self._dynamic(state=settled) != self._dynamic(state=last):
                pending.ticks.append(TickEntry(state=settled))
        self._write_action(
            pending=pending,
            actor="robot",
            label=label,
            start_step=start_step,
            end_step=end_step,
            success=success,
            outcome=outcome,
            cost=cost,
            cycle=cycle,
        )

    def human_action(
        self,
        *,
        skill: str,
        start_step: int | None,
        end_step: int | None,
        success: bool | None,
        objects: Sequence[str] = (),
        outcome: str | None = None,
        cost: float | None = None,
        cycle: int | None = None,
    ) -> None:
        self._flush_pending()
        if self.output_path is None:
            return
        pending = _PendingSkill(name=skill, objects=list(objects), params=[], controller={})
        moved: PlainState | None = None
        if self.state_provider is not None:
            after = self.state_provider()
            before = self._last_state or {}
            moved = {
                name: values
                for name, values in self._dynamic(state=after).items()
                if before.get(name) != values
            }
            pending.ticks.append(TickEntry(state=after))
        self._write_action(
            pending=pending,
            actor="human",
            label=None,
            start_step=start_step,
            end_step=end_step,
            success=success,
            outcome=outcome,
            cost=cost,
            cycle=cycle,
            moved_objects=moved,
        )

    def end_segment(
        self, *, success: bool | None = None, outcome: str | None = None, steps: int | None = None
    ) -> None:
        self._flush_pending()
        segment = self._segment or SegmentStart(phase=Phase.PRACTICE)
        self._write(
            entry=SegmentEnd(
                phase=segment.phase,
                checkpoint=segment.checkpoint,
                task_index=segment.task_index,
                success=success,
                outcome=outcome,
                steps=steps,
            )
        )

    def close(self) -> None:
        self._flush_pending()
        if self._file is not None:
            self._file.close()
            self._file = None

    def _dynamic(self, *, state: PlainState | None) -> PlainState:
        static = set(self.header.static_objects)
        return {k: v for k, v in (state or {}).items() if k not in static}

    def _flush_pending(self) -> None:
        """A skill whose outcome was never reported is still recorded, outcome unknown."""
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        self._write_action(
            pending=pending,
            actor="robot",
            label=None,
            start_step=None,
            end_step=None,
            success=None,
            outcome=None,
            cost=None,
            cycle=None,
        )

    def _write_action(
        self,
        *,
        pending: _PendingSkill,
        actor: str,
        label: str | None,
        start_step: int | None,
        end_step: int | None,
        success: bool | None,
        outcome: str | None,
        cost: float | None,
        cycle: int | None,
        moved_objects: PlainState | None = None,
    ) -> None:
        segment = self._segment or SegmentStart(phase=Phase.PRACTICE)
        seq = self._seq
        self._seq += 1
        self._write(
            entry=ActionEntry.model_validate({
                "seq": seq,
                "phase": segment.phase,
                "checkpoint": segment.checkpoint,
                "task_index": segment.task_index,
                "cycle": cycle,
                "actor": actor,
                "skill": pending.name,
                "objects": pending.objects,
                "params": pending.params,
                "controller": pending.controller,
                "label": label,
                "start_step": start_step,
                "end_step": end_step,
                "success": success,
                "outcome": outcome,
                "cost": cost,
                "moved_objects": moved_objects,
                "num_ticks": len(pending.ticks),
            })
        )
        for tick in pending.ticks:
            self._write_tick(tick=tick.model_copy(update={"seq": seq}))

    def _write_tick(self, *, tick: TickEntry) -> None:
        self._last_state = tick.state
        self._write(entry=tick)

    def _write(self, *, entry: BaseModel) -> None:
        if self.output_path is None:
            return
        if self._file is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output_path.open("w", encoding="utf-8")
        if not self._header_written:
            if self.scene_description_provider is not None:
                self.header = self.header.model_validate({
                    **self.header.model_dump(),
                    **self.scene_description_provider(),
                })
            self._header_written = True
            self._emit(record=self.header.model_dump(mode="json"))
        self._emit(record=entry.model_dump(mode="json"))

    def _emit(self, *, record: dict[str, Any]) -> None:
        assert self._file is not None
        self._file.write(LogTiming.encode(record=record))
        self._file.flush()
