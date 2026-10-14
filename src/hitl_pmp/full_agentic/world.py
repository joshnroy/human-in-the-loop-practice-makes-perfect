"""Serialized persistent-world tools, durable receipts, and fixed resource limits."""

import hashlib
import json
import socketserver
import sqlite3
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np

from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.step_protocol import PracticeClock


class World:
    def __init__(
        self,
        *,
        bridge: Any,
        workspace: Path,
        output: Path,
        deadline: float,
    ) -> None:
        self.bridge, self.workspace, self.output = bridge, workspace, output
        self.deadline = deadline
        self.version = self.steps = self.trials = self.help_requests = self.trial_steps = 0
        self.active = self.finished = self.uncertain = False
        self.accounting = PracticeAccounting()
        self.accounting.set_observation_provider(provider=bridge.observe)
        self.clock: PracticeClock | None = None
        self.measurements: list[dict[str, Any]] = []
        self.on_measurement: Callable[[dict[str, Any]], Any] | None = None
        self.human_by_side = {"robot_side": 0, "opposite_side": 0}
        self.revision: str | None = None
        self.submission: Path | None = None
        self.db = sqlite3.connect(output / "receipts.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE receipts(id TEXT PRIMARY KEY, request TEXT, response TEXT)")
        (workspace / "evidence").mkdir(parents=True, exist_ok=True)
        self.status()

    @property
    def counted_steps(self) -> int:
        return self.accounting.practice_steps

    def configure_measurements(
        self,
        *,
        budget: int,
        interval: int,
        human_steps: int = 1,
        callback: Any = None,
        costs: PracticeCosts | None = None,
    ) -> None:
        if self.steps or self.help_requests:
            raise ValueError("Configure measurements before physical practice")
        if human_steps != 1:
            raise ValueError("Set per-skill durations in the shared cost configuration")
        self.accounting = PracticeAccounting(costs=costs or PracticeCosts())
        self.accounting.set_observation_provider(provider=self.bridge.observe)
        self.on_measurement = callback
        self.clock = PracticeClock(
            budget=budget, interval=interval, accounting=self.accounting, on_measure=self.measure
        )
        self.clock.robot_active = True
        self.measure()

    def measure(self) -> None:
        index = len(self.measurements)
        directory = self.output / "measurements" / f"{index:04d}"
        directory.mkdir(parents=True)
        frozen = directory / "submission"
        digest = Files.snapshot(
            workspace=self.workspace, destination=frozen, require_controller=False
        )
        record = dict(
            index=index,
            practice_steps=self.counted_steps,
            robot_steps=self.steps,
            human_requests=self.help_requests,
            human_by_side=dict(self.human_by_side),
            snapshot=str(frozen),
            sha256=digest,
            world_version=self.version,
            created_unix=time.time(),
            controller_present=(frozen / "approach.py").is_file(),
            **self.accounting.summary(),
        )
        Files.atomic_json(path=directory / "snapshot.json", value=record)
        self.measurements.append(record)
        if self.on_measurement:
            self.on_measurement(record)

    def final_measurement(self) -> None:
        digest = None
        if self.submission:
            manifest = json.loads((self.submission / "source_manifest.json").read_text())
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if self.clock and (
            not self.measurements
            or self.measurements[-1]["practice_steps"] != self.counted_steps
            or (digest is not None and self.measurements[-1]["sha256"] != digest)
        ):
            self.measure()

    def status(self, *, phase: str = "adapting", **extra: Any) -> None:
        Files.atomic_json(
            path=self.output / "status.json",
            value=dict(
                phase=phase,
                updated_unix=time.time(),
                world_version=self.version,
                robot_trials=self.trials,
                control_steps=self.steps,
                practice_steps=self.counted_steps,
                human_by_side=self.human_by_side,
                human_requests=self.help_requests,
                **self.accounting.summary(),
                trial_active=self.active,
                agent_finished=self.finished,
                unresolved_execution=self.uncertain,
                **extra,
            ),
        )

    def observation(self) -> dict[str, Any]:
        return dict(
            observation=self.bridge.observe(),
            world_version=self.version,
            control_steps=self.steps,
            robot_trials=self.trials,
            human_requests=self.help_requests,
            accumulated_cost=self.accounting.total_cost,
            robot_step_price=self.accounting.robot_charge().cost,
            human_skill_quotes={
                name: self.accounting.human_charge(skill=name).model_dump(mode="json")
                for name in self.accounting.costs.human_skills
            },
            **self.accounting.summary(),
            remaining_trials=None,
            remaining_control_steps=(
                max(0, self.clock.budget - self.counted_steps) if self.clock else None
            ),
            practice_steps=self.counted_steps,
            remaining_help_requests=None,
            remaining_wall_seconds=(
                max(0, self.deadline - time.monotonic()) if self.deadline != float("inf") else None
            ),
        )

    def dispatch(self, *, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("operation")
        allowed = {
            "observe",
            "begin_trial",
            "step",
            "end_trial",
            "request_help",
            "finish_adaptation",
        }
        if op not in allowed:
            raise ValueError("Unknown operation; there is no reset or set_state API")
        identifier = request.get("request_id")
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 80:
            raise ValueError("A bounded unique request_id is required")
        encoded = json.dumps(request, sort_keys=True, allow_nan=False)
        row = self.db.execute(
            "SELECT request,response FROM receipts WHERE id=?", (identifier,)
        ).fetchone()
        if row:
            if row[0] != encoded:
                raise ValueError("request_id was reused with different content")
            if row[1] is None:
                raise RuntimeError("Unresolved prior execution; action will not be replayed")
            return json.loads(row[1])
        if op != "observe":
            if self.finished or self.uncertain or time.monotonic() >= self.deadline:
                raise RuntimeError("Practice ended, deadline reached, or execution unresolved")
            if request.get("expected_version") != self.version:
                raise ValueError("Stale world version; observe before deciding again")
        charge = None
        if op == "step":
            if self.clock and self.counted_steps >= self.clock.budget:
                raise RuntimeError("Practice step budget exhausted")
            action = request.get("action")
            if isinstance(action, dict):
                if set(action) not in ({"values"}, {"schedule"}):
                    raise ValueError("Action must contain only values or schedule")
                action = action.get("values", action.get("schedule"))
            array = np.asarray(action, dtype=float)
            spec = self.bridge.action_spec()
            if (
                array.shape not in ((18,), (spec["schedule_rows"], 18))
                or not np.isfinite(array).all()
                or np.any(array < spec["low"])
                or np.any(array > spec["high"])
            ):
                raise ValueError(
                    "Action violates published shape or bounds; physics was not advanced"
                )
        if op == "begin_trial" and self.active:
            raise ValueError("End the active trial first, or trial budget exhausted")
        if op == "request_help":
            if request.get("intervention_id") not in {"reset_cube_far", "reset_cube_and_bin_near"}:
                raise ValueError("Unknown human intervention")
            charge = self.accounting.human_charge(skill=request["intervention_id"])
            if self.clock and self.counted_steps + charge.steps > self.clock.budget:
                raise RuntimeError("Practice step budget exhausted")
        if op == "step":
            charge = self.accounting.robot_charge()

        self.db.execute("INSERT INTO receipts VALUES (?,?,NULL)", (identifier, encoded))
        self.db.commit()
        response: dict[str, Any]
        try:
            if op == "begin_trial":
                target = self.output / "checkpoints" / f"trial-{self.trials:04d}"
                self.revision = Files.snapshot(workspace=self.workspace, destination=target)
                self.trials += 1
                self.trial_steps = 0
                self.active = True
            elif op == "step":
                if self.clock:
                    self.clock.before_robot_step()
                self.bridge.step(action=request["action"])
                self.steps += 1
                self.trial_steps += 1
                if self.clock:
                    self.clock.after_robot_step()
                else:
                    assert charge is not None
                    self.accounting.record(charge=charge)
            elif op == "end_trial":
                self.active = False
            elif op == "request_help":
                destination = (
                    "opposite_side"
                    if request["intervention_id"] == "reset_cube_far"
                    else "robot_side"
                )
                try:
                    if not self.bridge.env.reset_movables(destination=destination):
                        raise RuntimeError("Human executor declined the intervention")
                finally:
                    self.help_requests += 1
                    self.human_by_side[destination] += 1
                    assert charge is not None
                    if self.clock:
                        self.clock.human_invoked(charge=charge)
                    else:
                        self.accounting.record(charge=charge)
            elif op == "finish_adaptation":
                self.submission = self.output / "submission"
                self.revision = Files.snapshot(
                    workspace=self.workspace, destination=self.submission
                )
                self.finished = True
                self.final_measurement()
            if op != "observe":
                self.version += 1
            result = self.observation()
            result.update(
                policy_digest=self.revision,
                trial_steps=self.trial_steps,
                request_id=identifier,
                operation=op,
            )
            response = {"result": result}
        except Exception as exc:
            # Once the physical executor was entered, partial execution is possible.
            if op in {"step", "request_help"}:
                self.uncertain = True
            response = {"error": f"{type(exc).__name__}: {exc}"}
        self.db.execute(
            "UPDATE receipts SET response=? WHERE id=?",
            (json.dumps(response, allow_nan=False), identifier),
        )
        self.db.commit()
        event = dict(
            request=request, response=response, policy_digest=self.revision, timestamp=time.time()
        )
        line = json.dumps(event, allow_nan=False) + "\n"
        for path in (
            self.output / "practice.jsonl",
            self.workspace / "evidence" / "practice.jsonl",
        ):
            with path.open("a") as stream:
                stream.write(line)
        if op != "step" or self.steps % 10 == 0 or self.uncertain:
            self.status()
        return response


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(10)
        try:
            raw = self.rfile.readline(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("Request exceeds size limit")
            result = cast(Server, self.server).world.dispatch(request=json.loads(raw))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        # Durable receipts remain available after a lost connection.
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(json.dumps(result, allow_nan=False).encode() + b"\n")


class Server(socketserver.UnixStreamServer):
    def __init__(self, *, path: Any, world: Any) -> None:
        self.world = world
        super().__init__(str(path), Handler)
        self.timeout = 0.1


class Files:
    @staticmethod
    def atomic_json(*, path: Path, value: Any) -> None:
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(path)

    @staticmethod
    def snapshot(*, workspace: Path, destination: Path, require_controller: bool = True) -> str:
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {}
        total = 0
        for source in sorted(workspace.rglob("*")):
            relative = source.relative_to(workspace)
            if any(p.startswith(".") or p in {"evidence", "__pycache__"} for p in relative.parts):
                continue
            if source.is_symlink():
                raise ValueError("Submission symlinks are not supported")
            if not source.is_file() or source.suffix not in {".py", ".json", ".npy", ".npz"}:
                continue
            if relative.name in {"env_client.py", "planning-mcp.json"}:
                continue
            size = source.stat().st_size
            total += size
            if total > 32_000_000:
                raise ValueError("Submission exceeds 32 MB")
            raw = source.read_bytes()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            manifest[str(relative)] = hashlib.sha256(raw).hexdigest()
        if require_controller and "approach.py" not in manifest:
            raise ValueError("approach.py is required")
        Files.atomic_json(path=destination / "source_manifest.json", value=manifest)
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
