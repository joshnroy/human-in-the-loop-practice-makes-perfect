"""`RealRunDashboard`: one run directory's live state, as the page draws it.

Every `refresh` reads only what the run appended since the last one (`tail.py`) and
rebuilds the whole view from what is held: the event log of a 40-cycle run is a few
thousand records, and the decision log is cut to competences and outcomes on the way in.

## A practice toss's outcome

The skill server records where each toss landed and whether it scored (`in_bin`; its
`inside_bin` is "inside the bin's footprint", which a cube can be without scoring). A
practice toss is matched to its server call by its three parameters, exactly: the
server echoes the floats the planner sent. Evaluation tosses are in the server log too
and match no practice toss. Without the server log, a toss's outcome is the decision
log's `outcome` for it (matched by timestamp), which has no landing point.
"""

import json
import math
import statistics
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, PrivateAttr

from hitl_pmp.environments.tossing3d.real.frame import BIN_SPOTS, ROOM_FRAME
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide

from .camera import CameraFeed, CameraState
from .room import BIN_SIZE_M, BinSpotView, Landing, Room
from .sources import (
    PICK_SKILL,
    SPOT_NAMES,
    TOSS_SKILL,
    Annotation,
    Annotations,
    DecisionLog,
    Evaluation,
    Evaluations,
    EventNote,
    Observation,
    RobotAction,
    SessionLog,
    SkillCall,
    SkillCalls,
    SkillOutcome,
    StepEvents,
)
from .tail import LogTail

# CLAUDE.md's training-curve style: colour carries the method.
_METHOD_STYLE: Final[dict[tuple[str, str | None], tuple[str, str]]] = {
    ("pomdp", "local_trend"): ("ours, Model B", "#0072B2"),
    ("pomdp", "global_curve"): ("ours, Model A", "#56B4E9"),
    ("ees", None): ("EES", "#009E73"),
}
_NEW_METHOD_COLOUR: Final = "#CC79A7"
ROLLING_WINDOW: Final = 10
# The tape is 0.05 m thick; `ROOM_FRAME.barrier_center` is its centre line.
TAPE_THICKNESS_M: Final = 0.05
# The skill server's drivable box on 2026-10-20 (`no cube found in the drivable area
# (-1.846, 2.046, -1.846, 1.846)`: x min, x max, y min, y max). Drawn for orientation
# only; the robot side owns the real one.
DRIVABLE_BOX: Final = (-1.846, 2.046, -1.846, 1.846)
_MATCH_TOLERANCE: Final = 1e-6
# A skill is never this long on the robot (a toss is ~45 s, a pick ~30 s, a look ~20 s).
STALE_LOG_SECONDS: Final = 180.0
_OUTCOME_MATCH_SECONDS: Final = 10.0


class Status(BaseModel):
    phase: str | None
    status_age_seconds: float | None
    cycle: int
    learning_updates: int
    projected_cycles: float | None
    practice_steps: int
    step_budget: int | None
    running: str | None
    last_skill: str | None
    last_skill_detail: str | None
    last_skill_time: str | None
    progress: str | None
    prompt: str | None
    prompt_since: str | None
    prompt_context: str | None
    session_seconds: float
    wall_seconds: float | None
    robot_call_seconds: float | None
    human_resets: dict[str, int]
    competences: dict[str, float]
    discarded_lines: int
    notes: list[EventNote]
    session_tail: list[str]
    # Seconds since any of the run's logs last changed, and whether that is too long:
    # a skill is running and nothing has been written for `STALE_LOG_SECONDS`.
    logs_age_seconds: float | None
    stale: bool
    sources: dict[str, bool]


class EvalPoint(BaseModel):
    index: int
    practice_steps: int
    cycle: int | None
    solved: int
    total: int
    label: str
    complete: bool
    # A human's correction (`--annotations`), beside the recorded value, never over it.
    corrected: int | None = None
    corrected_label: str | None = None
    corrections: list[str] = []


class EvalCurve(BaseModel):
    points: list[EvalPoint]
    method_label: str
    colour: str
    dashed: bool
    legend: str
    step_budget: int | None
    corrected_legend: str | None = None


class TossPoint(BaseModel):
    index: int
    step: int
    cycle: int
    spot: str | None
    standoff: float
    speed_deg_s: float
    release_ms: float
    in_bin: bool | None
    inside_bin: bool | None
    landed: tuple[float, float] | None
    note: str | None
    source: str | None
    timestamp: str | None
    # The method's epsilon-greedy draw (the decision log's `random_exploration`): a random
    # toss is not counted toward competence. `None` when the decision log has no match.
    random: bool | None = None


class Tosses(BaseModel):
    points: list[TossPoint]
    outcome_source: str | None
    in_bin_label: str
    rolling_window: int
    rolling_label: str
    # (toss index, in bin, known) over the trailing window ending at each toss.
    rolling: list[tuple[int, int, int]]
    greedy_label: str
    random_label: str


class TimingRow(BaseModel):
    name: str
    n: int
    median_s: float | None
    mean_s: float | None
    last_s: float | None


class Timing(BaseModel):
    skill_server: list[TimingRow]
    practice: list[TimingRow]
    pick_label: str
    server_pick_label: str | None


class CompetencePoint(BaseModel):
    decision: int
    cycle: int
    value: float


class DashboardState(BaseModel):
    run_dir: str
    updated: str
    status: Status
    eval_curve: EvalCurve
    tosses: Tosses
    room: Room
    timing: Timing
    competence: dict[str, list[CompetencePoint]]
    camera: CameraState | None


class RealRunDashboard(BaseModel):
    """Reads `run_dir` (and the skill server's log, if given) on every `refresh`."""

    run_dir: Path
    skill_calls: Path | None = None
    landings_shown: int = 30
    camera: CameraFeed | None = None
    annotations: Path | None = None

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _event_tail: LogTail | None = PrivateAttr(default=None)
    _events: list[tuple[int, dict[str, Any]]] = PrivateAttr(default_factory=list)
    _decision_tail: LogTail | None = PrivateAttr(default=None)
    _decisions: DecisionLog = PrivateAttr(default_factory=DecisionLog)
    _session_tail: LogTail | None = PrivateAttr(default=None)
    _session: SessionLog = PrivateAttr(default_factory=SessionLog)
    _calls_tail: LogTail | None = PrivateAttr(default=None)
    _calls: SkillCalls = PrivateAttr(default_factory=SkillCalls)
    _evaluations: list[Evaluation] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self._event_tail = LogTail(path=self.run_dir / "step_events.jsonl")
        self._decision_tail = LogTail(path=self.run_dir / "pomdp_decisions.jsonl")
        self._session_tail = LogTail(path=self.run_dir / "robot_session.log")
        if self.skill_calls is not None:
            self._calls_tail = LogTail(path=self.skill_calls)

    def refresh(self) -> DashboardState:
        with self._lock:
            self._read()
            return self._build()

    def _read(self) -> None:
        assert self._event_tail is not None and self._decision_tail is not None
        assert self._session_tail is not None
        read = self._event_tail.read()
        if read.restarted:
            self._events = []
        for line in read.lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = {"event": "unreadable_line"}
            self._events.append((len(self._events) + 1, record))
        read = self._decision_tail.read()
        if read.restarted:
            self._decisions = DecisionLog()
        for line in read.lines:
            self._decisions.ingest(line=line)
        read = self._session_tail.read()
        if read.restarted:
            self._session = SessionLog()
        for line in read.lines:
            self._session.ingest(line=line)
        if self._calls_tail is not None:
            read = self._calls_tail.read()
            if read.restarted:
                self._calls = SkillCalls()
            for line in read.lines:
                self._calls.ingest(line=line)
        evaluations = Evaluations.read(run_dir=self.run_dir)
        if evaluations is not None:
            self._evaluations = evaluations

    def _build(self) -> DashboardState:
        events = StepEvents.parse(records=self._events)
        config = _Files.json(path=self.run_dir / "config_snapshot.json") or {}
        protocol = _Files.json(path=self.run_dir / "step_protocol.json") or {}
        budget = protocol.get("practice_step_budget")
        tosses = TossBuilder.build(
            events=events,
            calls=self._calls.calls if self._calls_tail is not None else None,
            outcomes=self._decisions.outcomes,
        )
        room = RoomBuilder.build(
            tosses=tosses, observation=self._calls.last_observation(), shown=self.landings_shown
        )
        return DashboardState(
            run_dir=str(self.run_dir),
            updated=datetime.now().astimezone().isoformat(timespec="seconds"),
            status=self._status(events=events, budget=budget),
            eval_curve=EvalCurveBuilder.build(
                evaluations=self._evaluations,
                args=config.get("args", {}),
                budget=budget,
                annotations=Annotations.read(path=self.annotations),
            ),
            tosses=tosses,
            room=room,
            camera=self.camera.state(room=room) if self.camera is not None else None,
            timing=TimingBuilder.build(
                events=events, calls=self._calls.calls if self._calls_tail is not None else None
            ),
            competence={
                skill: [
                    CompetencePoint(decision=c.decision, cycle=c.cycle, value=c.values[skill])
                    for c in self._decisions.competences
                    if skill in c.values
                ]
                for skill in (PICK_SKILL, TOSS_SKILL)
            },
        )

    def _logs_age(self) -> float | None:
        paths = [
            self.run_dir / name
            for name in (
                "step_events.jsonl",
                "robot_session.log",
                "pomdp_decisions.jsonl",
                "status.json",
            )
        ]
        if self.skill_calls is not None:
            paths.append(self.skill_calls)
        newest = _LogAge.newest(paths=paths)
        return None if newest is None else max(0.0, datetime.now().timestamp() - newest)

    def _status(self, *, events: StepEvents, budget: int | None) -> Status:
        status = _Files.json(path=self.run_dir / "status.json") or {}
        session = self._session.state()
        last = events.robot_actions[-1] if events.robot_actions else None
        resets = {"SAME": 0, "FAR": 0}
        for reset in events.human_resets:
            name = SPOT_NAMES.get(reset.destination, reset.destination)
            resets[name] = resets.get(name, 0) + 1
        projected = None
        updates = [record for _, record in self._events if record.get("event") == "learning_update"]
        if budget and events.learning_updates and updates:
            steps = updates[-1].get("practice_steps")
            if isinstance(steps, int) and steps > 0:
                projected = budget / (steps / events.learning_updates)
        updated_unix = status.get("updated_unix")
        calls = self._calls.calls if self._calls_tail is not None else None
        logs_age = self._logs_age()
        return Status(
            phase=status.get("phase"),
            status_age_seconds=(datetime.now().timestamp() - updated_unix)
            if isinstance(updated_unix, int | float)
            else None,
            cycle=events.cycle,
            learning_updates=events.learning_updates,
            projected_cycles=projected,
            practice_steps=events.practice_steps,
            step_budget=budget,
            running=session.running,
            last_skill=last.skill if last else None,
            last_skill_detail=_Describe.action(action=last) if last else None,
            last_skill_time=last.timestamp if last else None,
            progress=session.progress,
            prompt=session.prompt,
            prompt_since=session.prompt_since,
            prompt_context=session.prompt_context,
            session_seconds=events.session_seconds,
            wall_seconds=_Describe.seconds_between(
                start=events.first_timestamp, end=events.last_timestamp
            ),
            robot_call_seconds=sum(call.duration_s or 0.0 for call in calls)
            if calls is not None
            else None,
            human_resets=resets,
            competences=self._decisions.competences[-1].values
            if self._decisions.competences
            else {},
            discarded_lines=events.discarded_lines,
            notes=events.notes[-10:],
            session_tail=session.tail,
            logs_age_seconds=logs_age,
            stale=session.running is not None
            and logs_age is not None
            and logs_age > STALE_LOG_SECONDS,
            sources={
                "step_events.jsonl": bool(self._events),
                "step_evaluations.json": bool(self._evaluations),
                "pomdp_decisions.jsonl": bool(
                    self._decisions.competences or self._decisions.outcomes
                ),
                "robot_session.log": bool(session.tail),
                "skill_calls.jsonl": bool(calls),
            },
        )


class _LogAge:
    @staticmethod
    def newest(*, paths: list[Path]) -> float | None:
        times = []
        for path in paths:
            for candidate in (path, path.with_name(path.name + ".zst")):
                try:
                    times.append(candidate.stat().st_mtime)
                except FileNotFoundError:
                    continue
        return max(times) if times else None


class EvalCurveBuilder:
    @staticmethod
    def build(
        *,
        evaluations: list[Evaluation],
        args: dict[str, Any],
        budget: int | None,
        annotations: list[Annotation],
    ) -> EvalCurve:
        method = str(args.get("method", ""))
        model = args.get("pomdp_competence_model") if method == "pomdp" else None
        label, colour = _METHOD_STYLE.get((method, model), (method or "?", _NEW_METHOD_COLOUR))
        points = [
            EvalPoint(
                index=evaluation.index,
                practice_steps=evaluation.practice_steps,
                cycle=evaluation.learning_updates,
                solved=evaluation.num_solved,
                total=evaluation.num_total,
                label=f"{evaluation.num_solved}/{evaluation.num_total}",
                complete=evaluation.complete,
            )
            for evaluation in evaluations
        ]
        # The last annotation of a task stands; each one moves the count by its change.
        latest: dict[tuple[int, int], Annotation] = {}
        for annotation in annotations:
            latest[(annotation.evaluation_index, annotation.task)] = annotation
        for point in points:
            mine = [a for (index, _), a in latest.items() if index == point.index]
            if not mine:
                continue
            change = sum(int(a.corrected_solved) - int(a.recorded_solved) for a in mine)
            point.corrected = point.solved + change
            point.corrected_label = f"{point.corrected}/{point.total}"
            point.corrections = [a.describe() for a in mine]
        corrected = sum(point.corrected is not None for point in points)
        complete = sum(point.complete for point in points)
        return EvalCurve(
            corrected_legend=f"corrected (human-confirmed), n={corrected}" if corrected else None,
            points=points,
            method_label=label,
            colour=colour,
            dashed=str(args.get("human_reset", "True")) == "False",
            legend=f"{label} — solved per evaluation, n={complete}",
            step_budget=budget,
        )


class TossBuilder:
    @staticmethod
    def build(
        *,
        events: StepEvents,
        calls: list[SkillCall] | None,
        outcomes: list[SkillOutcome],
    ) -> Tosses:
        server = [call for call in (calls or []) if call.toss is not None]
        used: set[int] = set()
        decided = [outcome for outcome in outcomes if outcome.skill == TOSS_SKILL]
        decided_used: set[int] = set()
        points: list[TossPoint] = []
        for index, action in enumerate(events.tosses()):
            toss = action.toss
            assert toss is not None
            point = TossPoint(
                index=index,
                step=action.end_step,
                cycle=action.cycle,
                spot=SPOT_NAMES.get(toss.bin_side or "", None),
                standoff=toss.standoff,
                speed_deg_s=toss.speed_deg_s,
                release_ms=toss.release_ms,
                in_bin=None,
                inside_bin=None,
                landed=None,
                note=action.skill_error,
                source=None,
                timestamp=action.timestamp,
            )
            for position, call in enumerate(server):
                result = call.toss
                assert result is not None
                if position in used or not all(
                    sent is not None and math.isclose(sent, got, abs_tol=_MATCH_TOLERANCE)
                    for sent, got in (
                        (result.standoff, toss.standoff),
                        (result.speed_deg_s, toss.speed_deg_s),
                        (result.release_ms, toss.release_ms),
                    )
                ):
                    continue
                used.add(position)
                point.in_bin, point.inside_bin = result.in_bin, result.inside_bin
                point.landed, point.source = result.landed, "skill_calls"
                point.note = result.note or point.note
                break
            match = TossBuilder._outcome(action=action, outcomes=decided, used=decided_used)
            if match is not None:
                point.random = decided[match].random_exploration
                if point.source is None:
                    point.in_bin, point.source = decided[match].success, "pomdp_decisions"
            points.append(point)
        sources = {point.source for point in points if point.source is not None}
        known = [point for point in points if point.in_bin is not None]
        rolling = []
        for point in points:
            window = [
                other
                for other in points[max(0, point.index - ROLLING_WINDOW + 1) : point.index + 1]
                if other.in_bin is not None
            ]
            rolling.append((point.index, sum(bool(other.in_bin) for other in window), len(window)))
        last = rolling[-1] if rolling else (0, 0, 0)
        greedy = [point for point in known if point.random is False]
        random = [point for point in known if point.random is True]
        return Tosses(
            greedy_label=f"{sum(bool(point.in_bin) for point in greedy)}/{len(greedy)}",
            random_label=f"{sum(bool(point.in_bin) for point in random)}/{len(random)}",
            points=points,
            outcome_source=None if not sources else sources.pop() if len(sources) == 1 else "mixed",
            in_bin_label=f"{sum(bool(point.in_bin) for point in known)}/{len(known)}",
            rolling_window=ROLLING_WINDOW,
            rolling_label=f"{last[1]}/{last[2]}",
            rolling=rolling,
        )

    @staticmethod
    def _outcome(
        *, action: RobotAction, outcomes: list[SkillOutcome], used: set[int]
    ) -> int | None:
        """The decision log's outcome for `action`: written within milliseconds of it."""
        when = _Describe.time(stamp=action.timestamp)
        best: tuple[float, int] | None = None
        for position, outcome in enumerate(outcomes):
            other = _Describe.time(stamp=outcome.timestamp)
            if position in used or when is None or other is None:
                continue
            gap = abs((other - when).total_seconds())
            if gap <= _OUTCOME_MATCH_SECONDS and (best is None or gap < best[0]):
                best = (gap, position)
        if best is None:
            return None
        used.add(best[1])
        return best[1]


class RoomBuilder:
    @staticmethod
    def build(*, tosses: Tosses, observation: Observation | None, shown: int) -> Room:
        # The tape's robot-side edge: its centre line moved half its thickness toward
        # the robot, as a line `y = intercept + slope x` in the map frame.
        side = ROOM_FRAME.robot_side
        slope = -side[0] / side[1]
        edge = (
            ROOM_FRAME.barrier_center[0] + TAPE_THICKNESS_M / 2 * side[0],
            ROOM_FRAME.barrier_center[1] + TAPE_THICKNESS_M / 2 * side[1],
        )
        spots = [
            BinSpotView(
                name="SAME" if side_name is Tossing3DSide.ROBOT else "FAR",
                x=spot.center[0],
                y=spot.center[1],
                yaw=spot.yaw,
                length=BIN_SIZE_M[0],
                width=BIN_SIZE_M[1],
            )
            for side_name, spot in BIN_SPOTS.items()
        ]
        landed = [point for point in tosses.points if point.landed is not None]
        bin_seen = observation.bin if observation is not None else None
        return Room(
            bin_spots=spots,
            tape_intercept=edge[1] - slope * edge[0],
            tape_slope=slope,
            tape_thickness=TAPE_THICKNESS_M,
            drivable_box=DRIVABLE_BOX,
            landings=[
                Landing(
                    index=point.index,
                    x=point.landed[0],
                    y=point.landed[1],
                    in_bin=point.in_bin,
                    inside_bin=point.inside_bin,
                    spot=point.spot,
                )
                for point in landed[-shown:]
                if point.landed is not None
            ],
            observation=observation,
            observed_bin=BinSpotView(
                name="observed",
                x=bin_seen.x,
                y=bin_seen.y,
                yaw=bin_seen.yaw,
                length=bin_seen.length,
                width=bin_seen.width,
            )
            if bin_seen is not None
            else None,
        )


class TimingBuilder:
    @staticmethod
    def build(*, events: StepEvents, calls: list[SkillCall] | None) -> Timing:
        server: list[TimingRow] = []
        if calls is not None:
            for name in ("pick", "toss", "observe", "go_home"):
                durations = [call.duration_s for call in calls if call.call == name]
                server.append(
                    TimingBuilder._row(name=name, values=[d for d in durations if d is not None])
                )
        practice = [
            TimingBuilder._row(
                name=skill,
                values=[
                    action.wall_seconds
                    for action in events.robot_actions
                    if action.skill == skill and action.wall_seconds is not None
                ],
            )
            for skill in (PICK_SKILL, TOSS_SKILL)
        ]
        picks = [action for action in events.robot_actions if action.skill == PICK_SKILL]
        held = [call.held for call in calls or [] if call.call == "pick" and call.held is not None]
        return Timing(
            skill_server=server,
            practice=practice,
            pick_label=f"{sum(action.success for action in picks)}/{len(picks)}",
            server_pick_label=f"{sum(held)}/{len(held)}" if calls is not None else None,
        )

    @staticmethod
    def _row(*, name: str, values: list[float]) -> TimingRow:
        return TimingRow(
            name=name,
            n=len(values),
            median_s=statistics.median(values) if values else None,
            mean_s=statistics.fmean(values) if values else None,
            last_s=values[-1] if values else None,
        )


class _Describe:
    @staticmethod
    def action(*, action: RobotAction) -> str:
        toss = action.toss
        if toss is None:
            return "ok" if action.success else f"error: {action.skill_error}"
        spot = SPOT_NAMES.get(toss.bin_side or "", "?")
        return (
            f"standoff {toss.standoff:.2f} m · {toss.speed_deg_s:.0f} deg/s · "
            f"release {toss.release_ms:.0f} ms · bin {spot}"
        )

    @staticmethod
    def time(*, stamp: str | None) -> datetime | None:
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            return None

    @staticmethod
    def seconds_between(*, start: str | None, end: str | None) -> float | None:
        first, last = _Describe.time(stamp=start), _Describe.time(stamp=end)
        return (last - first).total_seconds() if first and last else None


class _Files:
    @staticmethod
    def json(*, path: Path) -> dict[str, Any] | None:
        try:
            loaded = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None
