"""Loads a full-agentic run's own per-step logs as run-recording entries.

Full-agentic runs log every broker exchange: `practice.jsonl` (one request/response
per line -- `observe`, `step` for one control period, `request_help` for a human skill)
and `evaluations/NNNN/task-XX.jsonl` (`observe`, `step`, `finish`), each response
carrying the full object-centric observation. Every object's features are copied
verbatim, so playback of these runs draws exactly the logged states.

`step` is one control period, so a robot action spans exactly one counted step and is
recorded as skill `control_step` with the raw action vector as its parameters. The
files can exceed a gigabyte, so everything here streams.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .types import (
    ActionEntry,
    Phase,
    PlainState,
    RecordEntry,
    RecordingHeader,
    SegmentEnd,
    SegmentStart,
    TickEntry,
)


class FullAgenticRunLog:
    robot_skill = "control_step"

    @staticmethod
    def header(*, run: Path) -> RecordingHeader:
        protocol = json.loads((run / "protocol.json").read_text())
        observation = FullAgenticRunLog._first_observation(path=run / "practice.jsonl")
        return RecordingHeader(
            environment=str(protocol["environment"]),
            method=str(protocol.get("method")),
            run_seed=protocol.get("seed"),
            scene={"seed": protocol.get("seed")},
            feature_names={o["name"]: list(o["features"]) for o in observation["objects"]},
        )

    @staticmethod
    def checkpoints(*, run: Path) -> list[int]:
        return sorted(
            int(path.name)
            for path in (run / "evaluations").iterdir()
            if path.name.isdigit() and any(path.glob("task-*.jsonl"))
        )

    @staticmethod
    def practice_entries(*, run: Path) -> Iterator[RecordEntry]:
        yield SegmentStart(phase=Phase.PRACTICE, scene_seed=FullAgenticRunLog._seed(run=run))
        seq = 0
        previous_step: int | None = None
        last: PlainState | None = None
        with (run / "practice.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                request = record["request"]
                operation = request.get("operation")
                result = record.get("response", {}).get("result")
                if not isinstance(result, dict):
                    error = record.get("response", {}).get("error")
                    if operation in {"step", "request_help"}:
                        yield FullAgenticRunLog._action(
                            seq=seq,
                            request=request,
                            start=previous_step,
                            end=previous_step,
                            success=False,
                            outcome=str(error),
                        )
                        seq += 1
                    continue
                step = result.get("practice_steps")
                state = FullAgenticRunLog._state(observation=result.get("observation"))
                if operation == "observe":
                    if last is None and state is not None:
                        last = state
                        yield TickEntry(state=state)
                elif operation in {"step", "request_help"}:
                    human = operation == "request_help"
                    start = previous_step if human else (None if step is None else step - 1)
                    moved = None
                    if human and state is not None:
                        before = last or {}
                        moved = {k: v for k, v in state.items() if before.get(k) != v}
                    yield FullAgenticRunLog._action(
                        seq=seq,
                        request=request,
                        start=start,
                        end=step,
                        success=True,
                        outcome=None,
                        moved=moved,
                        num_ticks=int(state is not None),
                    )
                    if state is not None:
                        last = state
                        yield TickEntry(seq=seq, state=state)
                    seq += 1
                if step is not None:
                    previous_step = int(step)
        yield SegmentEnd(phase=Phase.PRACTICE, steps=previous_step)

    @staticmethod
    def evaluation_entries(*, run: Path, checkpoint: int) -> Iterator[RecordEntry]:
        directory = run / "evaluations" / f"{checkpoint:04d}"
        results_path = directory / "results.json"
        tasks = (
            {t["task"]: t for t in json.loads(results_path.read_text()).get("tasks", [])}
            if results_path.exists()
            else {}
        )
        for path in sorted(directory.glob("task-*.jsonl")):
            index = int(path.stem.split("-")[1])
            result = tasks.get(index, {})
            yield SegmentStart(
                phase=Phase.EVALUATION,
                checkpoint=checkpoint,
                task_index=index,
                task_seed=result.get("seed"),
            )
            seq = 0
            end: SegmentEnd | None = None
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    operation = record.get("operation")
                    state = FullAgenticRunLog._state(observation=record.get("observation"))
                    if operation == "observe" and state is not None and seq == 0:
                        yield TickEntry(state=state)
                    elif operation == "step":
                        step = int(record["step"])
                        yield ActionEntry(
                            seq=seq,
                            phase=Phase.EVALUATION,
                            checkpoint=checkpoint,
                            task_index=index,
                            actor="robot",
                            skill=FullAgenticRunLog.robot_skill,
                            params=[float(v) for v in record.get("action", [])],
                            start_step=step - 1,
                            end_step=step,
                            success=True,
                            num_ticks=int(state is not None),
                        )
                        if state is not None:
                            yield TickEntry(seq=seq, state=state)
                        seq += 1
                    elif operation == "finish":
                        end = SegmentEnd(
                            phase=Phase.EVALUATION,
                            checkpoint=checkpoint,
                            task_index=index,
                            success=result.get("solved"),
                            outcome=record.get("error") or ("done" if record.get("done") else None),
                            steps=result.get("steps", record.get("steps")),
                        )
            yield end or SegmentEnd(
                phase=Phase.EVALUATION,
                checkpoint=checkpoint,
                task_index=index,
                success=result.get("solved"),
                steps=result.get("steps"),
            )

    @staticmethod
    def _action(
        *,
        seq: int,
        request: dict[str, Any],
        start: int | None,
        end: int | None,
        success: bool,
        outcome: str | None,
        moved: PlainState | None = None,
        num_ticks: int = 0,
    ) -> ActionEntry:
        human = request.get("operation") == "request_help"
        return ActionEntry(
            seq=seq,
            phase=Phase.PRACTICE,
            actor="human" if human else "robot",
            skill=str(request.get("intervention_id")) if human else FullAgenticRunLog.robot_skill,
            params=[] if human else [float(v) for v in request.get("action", [])],
            start_step=start,
            end_step=end,
            success=success,
            outcome=outcome,
            moved_objects=moved,
            num_ticks=num_ticks,
        )

    @staticmethod
    def _state(*, observation: Any) -> PlainState | None:
        if not isinstance(observation, dict) or "objects" not in observation:
            return None
        return {
            o["name"]: [float(v) for v in o["features"].values()] for o in observation["objects"]
        }

    @staticmethod
    def _first_observation(*, path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                result = json.loads(line).get("response", {}).get("result")
                if isinstance(result, dict) and isinstance(result.get("observation"), dict):
                    return result["observation"]
        raise ValueError(f"{path} contains no observation")

    @staticmethod
    def _seed(*, run: Path) -> int | None:
        protocol = run / "protocol.json"
        return json.loads(protocol.read_text()).get("seed") if protocol.exists() else None
