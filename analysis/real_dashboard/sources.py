"""The real run's files, parsed into what the dashboard shows. Read-only, every one.

| source | written by | what is kept |
| --- | --- | --- |
| `step_events.jsonl` | `StepPracticeRunner` | every event; robot actions with toss parameters |
| `step_evaluations.json`, `evaluations/NNNN/` | the step protocol | solved x/N per evaluation |
| `pomdp_decisions.jsonl[.zst]` | the belief-space method | competences, skill outcomes |
| `robot_session.log` | `RealSessionLog` | progress line, running skill, open prompt |
| `skill_calls.jsonl` (optional) | the robot's skill server | landings, `in_bin`, holds, timings |

## A resumed run

`--resume` carries a run on from its last checkpoint. The interrupted process's events
after that checkpoint stay in `step_events.jsonl`, and the resume names them by line
(`resume.interrupted_first_line`..`interrupted_last_line`, 1-based; a `cycle_discarded`
event names the same span). Those lines are dropped here: their steps were uncharged and
their outcomes never learned from. Their *time* was spent, so the session clock counts
it. Any event this module does not know is kept as a note, never an error.

## The decision log is filtered before it is parsed

A decision record carries the whole search and belief, hundreds of KB to tens of MB. Only
the `decision` and `outcome` events are wanted, and of those only flat top-level fields,
so a record is matched by its leading `event` field and the fields are read by pattern;
`json.loads` runs only on the flat `competences` object.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

TOSS_SKILL: Final = "MoveToTossLocationAndToss"
PICK_SKILL: Final = "PickCube"
# The bin's spot, as the person at the robot is told it (`real/backend.py` SPOT_NAMES).
SPOT_NAMES: Final[dict[str, str]] = {"robot_side": "SAME", "opposite_side": "FAR"}
# Events with a practice meaning of their own; anything else is a note.
_STRUCTURAL: Final = frozenset({
    "session_start",
    "robot",
    "human",
    "measurement",
    "learning_update",
    "practice_end",
})
_DECISION_EVENT: Final = re.compile(r'^\{\s*"event":\s*"(decision|outcome)"')
_DECISION_INT: Final = r'"{key}":\s*(-?\d+)'
_DECISION_BOOL: Final = r'"{key}":\s*(true|false)'
_COMPETENCES: Final = re.compile(r'"competences":\s*(\{[^{}]*\})')
_SKILL: Final = re.compile(r'"skill":\s*"([^"]*)"')
_STAMP: Final = re.compile(r'"timestamp":\s*"([^"]*)"')
# The session log's last lines, shown as the terminal shows them.
_SESSION_TAIL: Final = 40
_SESSION_STAMP: Final = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (.*)$")


class TossParameters(BaseModel):
    """`MoveToTossLocationAndToss`'s action vector is `[1, standoff, direction, speed,
    release_ms]`; the bin's side is the skill's last object."""

    standoff: float
    direction: float
    speed_deg_s: float
    release_ms: float
    bin_side: str | None


class RobotAction(BaseModel):
    line: int
    cycle: int
    skill: str
    label: str
    start_step: int
    end_step: int
    success: bool
    skill_error: str | None
    timestamp: str | None
    # Seconds since the previous event of the same process: the action's wall time,
    # with the planning and the person's y before it.
    wall_seconds: float | None
    toss: TossParameters | None = None


class HumanReset(BaseModel):
    line: int
    cycle: int
    destination: str
    skill: str
    end_step: int
    success: bool
    timestamp: str | None


class EventNote(BaseModel):
    line: int
    event: str
    timestamp: str | None
    summary: str


class StepEvents(BaseModel):
    """`step_events.jsonl`, with a resume's interrupted lines dropped."""

    robot_actions: list[RobotAction]
    human_resets: list[HumanReset]
    measurements: list[dict[str, Any]]
    learning_updates: int
    cycle: int
    practice_steps: int
    practice_end: str | None
    discarded_lines: int
    notes: list[EventNote]
    # Summed over the processes that wrote the log (each restarts its own clock).
    session_seconds: float
    first_timestamp: str | None
    last_timestamp: str | None

    def tosses(self) -> list[RobotAction]:
        return [action for action in self.robot_actions if action.toss is not None]

    @staticmethod
    def parse(*, records: list[tuple[int, dict[str, Any]]]) -> "StepEvents":
        discarded: set[int] = set()
        for _, record in records:
            span = (
                (record.get("interrupted_first_line"), record.get("interrupted_last_line"))
                if record.get("event") == "resume"
                else (record.get("first_line"), record.get("last_line"))
                if record.get("event") == "cycle_discarded"
                else (None, None)
            )
            if isinstance(span[0], int) and isinstance(span[1], int):
                discarded.update(range(span[0], span[1] + 1))
        actions: list[RobotAction] = []
        resets: list[HumanReset] = []
        measurements: list[dict[str, Any]] = []
        notes: list[EventNote] = []
        learning_updates = 0
        cycle = 0
        practice_steps = 0
        practice_end: str | None = None
        sessions: list[float] = []
        previous: float | None = None
        for line, record in records:
            event = str(record.get("event"))
            elapsed = StepEvents._number(value=record.get("elapsed_seconds"))
            if elapsed is not None:
                # A new process (a resume, or a second launch) restarts the clock.
                if previous is None or elapsed < previous:
                    sessions.append(elapsed)
                    wall = None
                else:
                    wall = elapsed - previous
                    sessions[-1] = elapsed
                previous = elapsed
            else:
                wall = None
            if line in discarded:
                continue
            if isinstance(record.get("cycle"), int):
                cycle = record["cycle"]
            for key in ("end_step", "practice_steps"):
                if isinstance(record.get(key), int):
                    practice_steps = (
                        record[key] if event == "resume" else max(practice_steps, record[key])
                    )
            if event == "robot":
                actions.append(StepEvents._robot(line=line, record=record, wall=wall))
            elif event == "human":
                resets.append(
                    HumanReset(
                        line=line,
                        cycle=int(record.get("cycle", cycle)),
                        destination=str(record.get("destination")),
                        skill=str(record.get("skill")),
                        end_step=int(record.get("end_step", 0)),
                        success=bool(record.get("success", True)),
                        timestamp=record.get("timestamp"),
                    )
                )
            elif event == "measurement":
                measurements.append(record)
            elif event == "learning_update":
                learning_updates += 1
            elif event == "practice_end":
                practice_end = str(record.get("reason"))
            if event == "resume" and isinstance(record.get("learning_updates"), int):
                learning_updates = record["learning_updates"]
            if event not in _STRUCTURAL:
                notes.append(
                    EventNote(
                        line=line,
                        event=event,
                        timestamp=record.get("timestamp"),
                        summary=StepEvents._summary(record=record),
                    )
                )
        stamps = [record["timestamp"] for _, record in records if record.get("timestamp")]
        return StepEvents(
            robot_actions=actions,
            human_resets=resets,
            measurements=measurements,
            learning_updates=learning_updates,
            cycle=cycle,
            practice_steps=practice_steps,
            practice_end=practice_end,
            discarded_lines=len(discarded & {line for line, _ in records}),
            notes=notes,
            session_seconds=sum(sessions),
            first_timestamp=stamps[0] if stamps else None,
            last_timestamp=stamps[-1] if stamps else None,
        )

    @staticmethod
    def _robot(*, line: int, record: dict[str, Any], wall: float | None) -> RobotAction:
        label = str(record.get("label", ""))
        skill = label.split("(", 1)[0].strip()
        error = record.get("skill_error")
        action = record.get("action") or []
        toss = None
        if skill == TOSS_SKILL and len(action) >= 5:
            objects = label.split("(", 1)[1].split(")", 1)[0] if "(" in label else ""
            side = objects.split(",")[-1].strip() if objects else None
            toss = TossParameters(
                standoff=float(action[1]),
                direction=float(action[2]),
                speed_deg_s=float(action[3]),
                release_ms=float(action[4]),
                bin_side=side or None,
            )
        return RobotAction(
            line=line,
            cycle=int(record.get("cycle", 0)),
            skill=skill,
            label=label,
            start_step=int(record.get("start_step", 0)),
            end_step=int(record.get("end_step", 0)),
            success=error is None and not record.get("interrupted", False),
            skill_error=None if error is None else str(error),
            timestamp=record.get("timestamp"),
            wall_seconds=wall,
            toss=toss,
        )

    @staticmethod
    def _summary(*, record: dict[str, Any]) -> str:
        shown = {
            key: value
            for key, value in record.items()
            if key not in {"event", "timestamp", "elapsed_seconds", "git_commit"}
            and not isinstance(value, dict | list)
        }
        return ", ".join(f"{key}={value}" for key, value in shown.items())

    @staticmethod
    def _number(*, value: Any) -> float | None:
        return float(value) if isinstance(value, int | float) else None


class Evaluation(BaseModel):
    index: int
    practice_steps: int
    learning_updates: int | None
    num_solved: int
    num_total: int
    complete: bool


class Evaluations:
    """`step_evaluations.json` (finished evaluations, rewritten whole after each) and
    the newest `evaluations/NNNN/progress.json` past it (one still running)."""

    @staticmethod
    def read(*, run_dir: Path) -> list[Evaluation] | None:
        """`None` when a file is caught half-written: keep the last good read."""
        finished = Evaluations._json(path=run_dir / "step_evaluations.json")
        measurements = Evaluations._json(path=run_dir / "measurements.json")
        if finished is None or measurements is None:
            return None
        by_index: dict[int, Evaluation] = {}
        for record in finished if isinstance(finished, list) else []:
            tasks = record.get("tasks") or []
            solved = sum(bool(task.get("solved")) for task in tasks)
            by_index[int(record["index"])] = Evaluation(
                index=int(record["index"]),
                practice_steps=int(record.get("practice_steps", 0)),
                learning_updates=record.get("learning_updates"),
                num_solved=int(record.get("num_solved", solved)),
                num_total=int(record.get("num_total", len(tasks))),
                complete=bool(record.get("complete", True)),
            )
        points = {
            int(record["index"]): record
            for record in (measurements if isinstance(measurements, list) else [])
            if "index" in record
        }
        directory = run_dir / "evaluations"
        running = sorted(directory.glob("[0-9]*")) if directory.is_dir() else []
        for folder in running:
            if not folder.name.isdigit() or int(folder.name) in by_index:
                continue
            progress = Evaluations._json(path=folder / "progress.json")
            point = points.get(int(folder.name))
            if not isinstance(progress, dict) or point is None:
                continue
            tasks = progress.get("tasks") or []
            by_index[int(folder.name)] = Evaluation(
                index=int(folder.name),
                practice_steps=int(point.get("practice_steps", 0)),
                learning_updates=point.get("learning_updates"),
                num_solved=sum(bool(task.get("solved")) for task in tasks),
                num_total=len(tasks),
                complete=False,
            )
        return [by_index[index] for index in sorted(by_index)]

    @staticmethod
    def _json(*, path: Path) -> Any:
        """The file's JSON, `[]` if it does not exist yet, `None` if half-written."""
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class Competences(BaseModel):
    decision: int
    cycle: int
    timestamp: str | None
    values: dict[str, float]


class SkillOutcome(BaseModel):
    decision: int
    cycle: int
    skill: str
    success: bool
    random_exploration: bool
    timestamp: str | None


class DecisionLog(BaseModel):
    """`pomdp_decisions.jsonl`: competence per skill at each decision, and outcomes."""

    competences: list[Competences] = []
    outcomes: list[SkillOutcome] = []

    def ingest(self, *, line: str) -> None:
        kind = _DECISION_EVENT.match(line)
        if kind is None:
            return
        decision = DecisionLog._int(line=line, key="decision")
        cycle = DecisionLog._int(line=line, key="cycle")
        stamps = _STAMP.findall(line)
        # The record's own timestamp is its last field; nested ones come before it.
        stamp = stamps[-1] if stamps else None
        if kind.group(1) == "decision":
            found = _COMPETENCES.search(line)
            if found is None:
                return
            raw = json.loads(found.group(1))
            self.competences.append(
                Competences(
                    decision=decision,
                    cycle=cycle,
                    timestamp=stamp,
                    values={
                        name.removesuffix(" (belief mean)"): float(value)
                        for name, value in raw.items()
                    },
                )
            )
            return
        skill = _SKILL.search(line)
        success = re.search(_DECISION_BOOL.format(key="success"), line)
        if skill is None or success is None:
            return
        random = re.search(_DECISION_BOOL.format(key="random_exploration"), line)
        self.outcomes.append(
            SkillOutcome(
                decision=decision,
                cycle=cycle,
                skill=skill.group(1),
                success=success.group(1) == "true",
                random_exploration=random is not None and random.group(1) == "true",
                timestamp=stamp,
            )
        )

    @staticmethod
    def _int(*, line: str, key: str) -> int:
        found = re.search(_DECISION_INT.format(key=key), line)
        return int(found.group(1)) if found else 0


class TossResult(BaseModel):
    standoff: float | None
    speed_deg_s: float | None
    release_ms: float | None
    landed: tuple[float, float] | None
    in_bin: bool | None
    inside_bin: bool | None
    note: str | None


class BinPose(BaseModel):
    x: float
    y: float
    yaw: float
    length: float
    width: float


class Observation(BaseModel):
    robot: tuple[float, float, float] | None
    cube: tuple[float, float] | None
    bin: BinPose | None


class SkillCall(BaseModel):
    i: int
    call: str
    start: float | None
    end: float | None
    duration_s: float | None
    toss: TossResult | None = None
    held: bool | None = None
    observation: Observation | None = None


class SkillCalls(BaseModel):
    """The skill server's `skill_calls.jsonl`: one record per robot call, evaluation's
    included. Replies are cut down to the fields shown."""

    calls: list[SkillCall] = []

    def ingest(self, *, line: str) -> None:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        call = str(record.get("call"))
        args = record.get("args") or {}
        reply = record.get("reply")
        reply = reply if isinstance(reply, dict) else {}
        parsed = SkillCall(
            i=int(record.get("i", len(self.calls))),
            call=call,
            start=record.get("start"),
            end=record.get("end"),
            duration_s=record.get("duration_s"),
        )
        if call == "toss":
            landed = reply.get("landed")
            parsed.toss = TossResult(
                standoff=args.get("standoff"),
                speed_deg_s=args.get("speed"),
                release_ms=args.get("release_ms"),
                landed=(float(landed[0]), float(landed[1])) if landed else None,
                in_bin=reply.get("in_bin"),
                inside_bin=reply.get("inside_bin"),
                note=reply.get("note"),
            )
        elif call == "pick":
            parsed.held = reply.get("held")
        elif call == "observe":
            parsed.observation = SkillCalls._observation(reply=reply)
        self.calls.append(parsed)

    def last_observation(self) -> Observation | None:
        for call in reversed(self.calls):
            if call.observation is not None:
                return call.observation
        return None

    @staticmethod
    def _observation(*, reply: dict[str, Any]) -> Observation:
        robot, cube, box = reply.get("robot"), reply.get("cube"), reply.get("bin")
        return Observation(
            robot=(float(robot[0]), float(robot[1]), float(robot[2])) if robot else None,
            cube=(float(cube[0]), float(cube[1])) if cube else None,
            bin=BinPose(
                x=box["x"], y=box["y"], yaw=box["yaw"], length=box["length"], width=box["width"]
            )
            if isinstance(box, dict)
            else None,
        )


class SessionState(BaseModel):
    progress: str | None
    running: str | None
    prompt: str | None
    prompt_since: str | None
    prompt_context: str | None
    last_time: str | None
    tail: list[str]


class SessionLog(BaseModel):
    """`robot_session.log` (`RealSessionLog`): `YYYY-MM-DD HH:MM:SS.mmm message` lines.

    A prompt is a line ending `[y/n]`; it is open until the `answer: ...` line the log
    writes after it. A skill is running from `running <action> ...` until the
    `<action>: <outcome> (N s); ...` line after it."""

    lines: list[str] = []
    progress: str | None = None
    running: str | None = None
    prompt: str | None = None
    prompt_since: str | None = None
    prompt_context: str | None = None
    last_time: str | None = None

    def ingest(self, *, line: str) -> None:
        stamped = _SESSION_STAMP.match(line)
        if stamped is None:
            if self.lines:
                self.lines[-1] += "\n" + line
            return
        when, message = stamped.groups()
        previous = _SESSION_STAMP.match(self.lines[-1]) if self.lines else None
        self.lines = [*self.lines[-(_SESSION_TAIL - 1) :], line]
        self.last_time = when
        if message.startswith("cycle ") and " action " in message and " steps " in message:
            self.progress = message
        if message.startswith("running ") and message.endswith(" ..."):
            self.running = message.removeprefix("running ").removesuffix(" ...")
        elif self.running is not None and message.startswith(f"{self.running}: "):
            self.running = None
        if message.rstrip().endswith("[y/n]"):
            self.prompt, self.prompt_since = message.rstrip(), when
            self.prompt_context = previous.group(2) if previous else None
        elif message.startswith("answer: "):
            self.prompt = self.prompt_since = self.prompt_context = None

    def state(self) -> SessionState:
        return SessionState(
            progress=self.progress,
            running=self.running,
            prompt=self.prompt,
            prompt_since=self.prompt_since,
            prompt_context=self.prompt_context,
            last_time=self.last_time,
            tail=list(self.lines),
        )

    @staticmethod
    def parse_time(*, stamp: str) -> datetime:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f")
