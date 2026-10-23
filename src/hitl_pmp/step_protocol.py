"""Practice measured in control steps, independently of learning sessions.

Any domain registered with a `StepProtocolEnvironment` (`cli_protocols.py`) runs here.

The training process publishes immutable deployment snapshots. Spawned workers own
their environments, policies and RNGs; they return measurements only to the logger.
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import pickle
import random
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from hitl_pmp.cli_protocols import EnvironmentCli, StepProtocolEnvironment
from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.log_timing import LogTiming
from hitl_pmp.core.method.method import HumanCubeBinResetRequested, InteractionComplete
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.metrics.types import PracticeSessionEnd, TaskOutcome
from hitl_pmp.core.practice_costs import ExecutionCharge, PracticeAccounting, PracticeCosts
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.human.types import (
    SAMPLED_START_RESET_DESTINATION,
    SAMPLED_START_RESET_SKILL_NAME,
)
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy
from hitl_pmp.recording.run_record.types import RECORDING_FILENAME, Phase
from hitl_pmp.recording.run_record.writer import RunRecordWriter


class PracticeClock(BaseModel):
    """Only actual practice controller calls and human invocations advance this clock."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    budget: int = Field(gt=0)
    interval: int = Field(gt=0)
    human_steps: int = Field(default=1, gt=0)
    robot_steps: int = 0
    stop_steps: int = 0
    human_invocations: int = 0
    counted_human_steps: int = 0
    accounting: PracticeAccounting | None = None
    observation: Callable[[], Any] = lambda: None
    _robot_charge: ExecutionCharge | None = PrivateAttr(default=None)
    robot_active: bool = False
    on_measure: Callable[[], None]
    on_progress: Callable[[], None] = lambda: None
    progress_interval: int = Field(default=100, gt=0)
    _measuring_at: int | None = PrivateAttr(default=None)

    @property
    def steps(self) -> int:
        return self.robot_steps + self.stop_steps + self.counted_human_steps

    @property
    def measurement_steps(self) -> int:
        """The practice step a measurement is taken at.

        Inside `on_measure` this is the threshold being crossed, which differs from
        `steps` only when one human invocation jumps over several thresholds at once:
        no learning happens inside that jump, so one policy is measured at each of them,
        and each lands at its own x rather than all of them at the jump's end."""
        return self.steps if self._measuring_at is None else self._measuring_at

    def before_robot_step(self) -> None:
        if self.robot_active and self.steps >= self.budget:
            raise ControlStepLimitReached()
        if self.robot_active and self.accounting is not None:
            self._robot_charge = self.accounting.robot_charge(observation=self.observation())

    def after_robot_step(self) -> None:
        if self.robot_active:
            previous = self.steps
            if self.accounting is not None:
                self.accounting.record(charge=self._robot_charge or self.accounting.robot_charge())
                self._robot_charge = None
            self.robot_steps += 1
            self._advanced(previous=previous)

    def stop_invoked(self) -> None:
        """A STOP: a skill whose competence is fixed at 1, charged one robot control step.

        Without a price a session-opening STOP would end practice for free, and a
        planner that stops every session would never spend the budget."""
        previous = self.steps
        if self.accounting is not None:
            self.accounting.record(
                charge=self.accounting.robot_charge(observation=self.observation())
            )
        self.stop_steps += 1
        self._advanced(previous=previous)

    def before_human(self, *, skill: str = "reset_cube_far") -> ExecutionCharge:
        charge = (
            self.accounting.human_charge(skill=skill, observation=self.observation())
            if self.accounting is not None
            else ExecutionCharge(actor="human", skill=skill, steps=self.human_steps, cost=0)
        )
        if self.steps + charge.steps > self.budget:
            raise ControlStepLimitReached()
        return charge

    def human_invoked(self, *, charge: ExecutionCharge | None = None) -> None:
        charge = charge or self.before_human()
        previous = self.steps
        self.human_invocations += 1
        self.counted_human_steps += charge.steps
        if self.accounting is not None:
            self.accounting.record(charge=charge)
        self._advanced(previous=previous)

    def _advanced(self, *, previous: int) -> None:
        # Once per crossed threshold: one increment can cross several (a human
        # invocation longer than the interval) or land past one without touching it.
        for threshold in range(
            (previous // self.interval + 1) * self.interval, self.steps + 1, self.interval
        ):
            self._measuring_at = threshold
            try:
                self.on_measure()
            finally:
                self._measuring_at = None
        crossed = self.steps // self.progress_interval - previous // self.progress_interval
        for _ in range(crossed):
            self.on_progress()


class EvaluationStopping:
    @staticmethod
    def reached(*, records: list[dict[str, Any]], num_tasks: int, patience: int = 3) -> bool:
        if patience <= 0:
            return False
        streak = 0
        previous = None
        for record in records:
            if not record.get("complete", True):
                streak = 0
                continue
            step = record["practice_steps"]
            if step == previous:
                continue
            previous = step
            perfect = record.get("num_solved") == record.get("num_total") == num_tasks
            streak = streak + 1 if perfect else 0
            if streak >= patience:
                return True
        return False

    @staticmethod
    def completed(*, futures: list[Any]) -> list[dict[str, Any]]:
        records = []
        for record, future in futures:
            if not future.done():
                break  # Never skip an unfinished earlier evaluation.
            try:
                result = future.result()
                if isinstance(result, list):
                    result = dict(
                        num_solved=sum(t["solved"] for t in result), num_total=len(result)
                    )
                records.append({**record, **result})
            except Exception:
                records.append({**record, "complete": False})
        return records


class PolicyFileHash:
    """A static-method container, never instantiated.

    The sha256 of a deployment snapshot's pickled bytes. It identifies the *file*, not
    the policy: pickle's memo order follows string-set iteration, which depends on
    PYTHONHASHSEED, so the same policy pickled in two processes hashes differently. Use
    it to tie an evaluation to the snapshot file it read, never to match runs."""

    KEY = "policy_file_sha256"
    LEGACY_KEY = "policy_sha256"

    @staticmethod
    def record(*, raw: bytes) -> dict[str, str]:
        return {PolicyFileHash.KEY: hashlib.sha256(raw).hexdigest()}

    @staticmethod
    def read(*, record: dict[str, object]) -> str | None:
        """The hash under the current key, or the key results were written with before."""
        value = record.get(PolicyFileHash.KEY, record.get(PolicyFileHash.LEGACY_KEY))
        return None if value is None else str(value)


class EvaluationClock(BaseModel):
    """Counts one held-out evaluation episode's control steps and scores it.

    **Known divergence, deliberately not fixed here.** `solved` latches on the first
    physics tick at which `check_goals()` is true, and `before` then stops the
    controller. A cube *passing through* the goal box mid-flight therefore scores as
    solved even if it would bounce out. The incumbent evaluation (`PracticeLoop`)
    scores the state after the whole skill has finished, so the two protocols' success
    rates are not the same measurement and must not be compared as if they were."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    budget: int = Field(gt=0)
    check_goal: Callable[[], bool]
    steps: int = 0
    solved: bool = False

    def before(self) -> None:
        if self.solved or self.steps >= self.budget:
            raise ControlStepLimitReached()

    def after(self) -> None:
        self.steps += 1
        self.solved = self.solved or self.check_goal()

    def interruption(self) -> str:
        """Why `before` cut the running skill off: the goal latch takes precedence."""
        return "stopped: goal reached" if self.solved else "interrupted: evaluation step budget"


class FreshProcessEvaluations(BaseModel):
    """Evaluations run one at a time, each in a freshly spawned process.

    A long-lived worker would build a new MuJoCo scene per evaluation and keep whatever
    the simulator fails to release, so memory grows with the number of measurements.
    `ProcessPoolExecutor(max_tasks_per_child=1)` would say this in one argument, but it
    needs Python 3.11 and this project runs 3.10, so a one-thread queue owns ordering
    and each job gets its own single-use spawn pool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _queue: ThreadPoolExecutor = PrivateAttr(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1)
    )
    _lock: Any = PrivateAttr(default_factory=threading.Lock)
    _current: ProcessPoolExecutor | None = PrivateAttr(default=None)
    _closing: bool = PrivateAttr(default=False)

    def submit(self, *, fn: Callable[..., Any], **kwargs: Any) -> Future:
        return self._queue.submit(lambda: self._isolated(fn=fn, kwargs=kwargs))

    def has_running_process(self) -> bool:
        with self._lock:
            return self._current is not None

    def finish(self) -> None:
        """Wait for every queued evaluation."""
        self._queue.shutdown(wait=True)

    def abort(self) -> None:
        """Cancel queued evaluations and terminate the running one, without waiting."""
        with self._lock:
            self._closing = True
            current = self._current
        self._queue.shutdown(wait=False, cancel_futures=True)
        if current is not None:
            # `_processes` becomes None if the job finished and shut its pool meanwhile.
            for process in list((current._processes or {}).values()):
                process.terminate()
            current.shutdown(wait=False, cancel_futures=True)

    def _isolated(self, *, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        # Submitted under the lock, so `abort` always sees a spawned worker to terminate.
        with self._lock:
            if self._closing:
                raise CancelledError()
            pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
            inner = pool.submit(fn, **kwargs)
            self._current = pool
        try:
            return inner.result()
        finally:
            with self._lock:
                self._current = None
            pool.shutdown(wait=not self._closing, cancel_futures=True)


class StepFiles:
    @staticmethod
    def json(*, path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def trace(*, environment: type[StepProtocolEnvironment], problem: Problem) -> dict[str, Any]:
        """The `skill_trace` field of a robot event, or nothing for a domain that
        keeps no trace -- so those domains' event logs are unchanged."""
        trace = environment.last_skill_trace(problem=problem)
        return {} if trace is None else {"skill_trace": trace}

    @staticmethod
    def event(*, path: Path, **record: Any) -> None:
        with path.open("a") as stream:
            stream.write(LogTiming.encode(record=record))

    @staticmethod
    def timing(
        *, path: Path, environment: type[StepProtocolEnvironment], problem: Problem, **record: Any
    ) -> None:
        """One `skill_timing.jsonl` line for the execution `record` names, if the domain
        timed it. A file of its own: wall-clock must never reach the byte-stable logs."""
        timing = environment.drain_execution_timing(problem=problem)
        if timing is not None:
            StepFiles.event(path=path, **record, **timing)


CHECKPOINT_FILENAME = "checkpoint.pickle"
SKILL_TIMING_FILENAME = "skill_timing.jsonl"
PREVIOUS_CHECKPOINT_FILENAME = "checkpoint.previous.pickle"


class PracticeCheckpoint:
    """Everything an interrupted practice run needs to carry on as if it had not stopped.

    Written at a cycle end, after the learning update, so nothing is in flight. One
    pickle graph holds the whole method (`DeploymentSnapshot` keeps only what deployment
    reads, not practice tallies or a POMDP's belief), the accounting, metrics, clock and
    loop counters, and every RNG practice draws from, the process-global ones included.

    The live environment, problem and predicates are pickled as references and bound to
    the resuming process's own, so a resumed method acts on the world as it is now,
    never on a copy (a predicate's classifier is a lambda pickle cannot carry). The file
    is a plain-data header then the graph, renamed into place after the last checkpoint
    is kept as `PREVIOUS_CHECKPOINT_FILENAME`, so an interruption at any instant leaves
    a whole checkpoint.
    """

    @staticmethod
    def predicate_key(*, name: str) -> str:
        return f"predicate:{name}"

    @staticmethod
    def live_objects(*, problem: Any, skill_provider: Any) -> dict[str, Any]:
        """The objects a checkpoint refers to rather than copies, by key."""
        live: dict[str, Any] = {"env": problem.env, "problem": problem}
        if skill_provider is not None:
            for predicate in skill_provider.predicates():
                live[PracticeCheckpoint.predicate_key(name=predicate.name)] = predicate
        return live

    @staticmethod
    def save(
        *, output: Path, live: dict[str, Any], header: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        buffer = io.BytesIO()
        pickle.dump(header, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        _ReferencingPickler(buffer, live=live).dump(payload)
        path = output / CHECKPOINT_FILENAME
        temporary = output / (CHECKPOINT_FILENAME + ".tmp")
        temporary.write_bytes(buffer.getvalue())
        if path.exists():
            # A second name for the last checkpoint, swapped in whole, so the current
            # one is never absent: at worst both names hold the same checkpoint.
            kept = output / (PREVIOUS_CHECKPOINT_FILENAME + ".tmp")
            kept.unlink(missing_ok=True)
            os.link(path, kept)
            kept.replace(output / PREVIOUS_CHECKPOINT_FILENAME)
        temporary.replace(path)

    @staticmethod
    def header(*, path: Path) -> dict[str, Any]:
        """The checkpoint's plain-data summary, readable without rebuilding the run."""
        with path.open("rb") as handle:
            header: dict[str, Any] = pickle.load(handle)  # trusted, locally written
        return header

    @staticmethod
    def load(*, path: Path, live: dict[str, Any]) -> dict[str, Any]:
        with path.open("rb") as handle:
            pickle.load(handle)  # the header
            payload: dict[str, Any] = _ReferencingUnpickler(handle, live=live).load()
        return payload


class _ReferencingPickler(pickle.Pickler):
    def __init__(self, file: Any, *, live: dict[str, Any]) -> None:  # noqa: PLR0917
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._references = {
            id(obj): key for key, obj in live.items() if not key.startswith("predicate:")
        }
        self._live = live

    def persistent_id(self, obj: Any) -> str | None:  # noqa: PLR0917 -- pickle's API
        key = self._references.get(id(obj))
        if key is not None:
            return key
        if isinstance(obj, Predicate):
            key = PracticeCheckpoint.predicate_key(name=obj.name)
            if self._live.get(key) is not obj:
                raise TypeError(
                    f"a practice checkpoint reached predicate {obj.name!r}, which is not "
                    "one of this run's skill provider's predicates"
                )
            return key
        if isinstance(obj, Environment | Problem):
            raise TypeError(
                f"a practice checkpoint reached a live {type(obj).__name__} that is not "
                "this run's own; pickling it would resume against a copy of the world"
            )
        return None


class _ReferencingUnpickler(pickle.Unpickler):
    def __init__(self, file: Any, *, live: dict[str, Any]) -> None:  # noqa: PLR0917
        super().__init__(file)  # trusted, locally written run checkpoints
        self._live = live

    def persistent_load(self, pid: Any) -> Any:  # noqa: PLR0917 -- pickle's API
        if pid not in self._live:
            raise pickle.UnpicklingError(
                f"the checkpoint refers to {pid!r}, which this run does not have"
            )
        return self._live[pid]


class RngStates:
    """Every RNG an object holds as a field or private attribute, by name."""

    @staticmethod
    def capture(*, owner: Any) -> dict[str, Any]:
        attributes = {**vars(owner), **(getattr(owner, "__pydantic_private__", None) or {})}
        return {
            name: value.bit_generator.state
            for name, value in attributes.items()
            if isinstance(value, np.random.Generator)
        }

    @staticmethod
    def restore(*, owner: Any, states: dict[str, Any]) -> None:
        for name, state in states.items():
            getattr(owner, name).bit_generator.state = state

    @staticmethod
    def capture_global() -> dict[str, Any]:
        return {
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
        }

    @staticmethod
    def restore_global(*, states: dict[str, Any]) -> None:
        np.random.set_state(states["numpy"])
        random.setstate(states["random"])
        torch.set_rng_state(states["torch"])


class StepProtocolResume:
    """`--resume <output-dir>`: what is checked before anything is built, and how the
    interrupted run's event log is accounted for."""

    @staticmethod
    def prepare(*, args: argparse.Namespace) -> None:
        """Refuse a resume that cannot carry the run on, before any world is touched."""
        resume = Path(args.resume)
        if args.output_dir is None:
            args.output_dir = resume
        elif Path(args.output_dir).resolve() != resume.resolve():
            raise ValueError(
                f"--resume {resume} names a different run than --output-dir "
                f"{args.output_dir}; pass the same directory, or omit --output-dir"
            )
        if not (resume / CHECKPOINT_FILENAME).exists():
            raise FileNotFoundError(
                f"--resume needs a cycle checkpoint at {resume / CHECKPOINT_FILENAME}; "
                "none was written there (no practice cycle finished, or the run did not "
                "checkpoint)"
            )
        snapshot = json.loads((resume / "config_snapshot.json").read_text())
        original = StepProtocolResume.comparable(arguments=snapshot["args"])
        now = StepProtocolResume.comparable(
            arguments={key: str(value) for key, value in vars(args).items()}
        )
        differing = sorted(
            key for key in original.keys() | now.keys() if original.get(key) != now.get(key)
        )
        if differing:
            details = "; ".join(
                f"{key}: {original.get(key, '<absent>')} -> {now.get(key, '<absent>')}"
                for key in differing
            )
            raise ValueError(
                f"refusing to resume {resume}: the resolved config differs from its "
                f"config_snapshot.json in {', '.join(differing)} ({details})"
            )

    @staticmethod
    def comparable(*, arguments: dict[str, str]) -> dict[str, str]:
        """The resolved flags a resume must repeat: all but `--resume` itself, and the
        output directory, which `prepare` has already matched as a path."""
        return {k: v for k, v in arguments.items() if k not in {"resume", "output_dir"}}

    @staticmethod
    def interrupted_events(*, events: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
        """The event log past the checkpoint's position: the number of lines before
        `offset`, and every event written after it by the interrupted process."""
        raw = events.read_bytes() if events.exists() else b""
        before, after = raw[:offset], raw[offset:]
        return before.count(b"\n"), [json.loads(line) for line in after.splitlines() if line]


class DeploymentSnapshot:
    """Both structured arms inherit EES deployment; practice-only state is excluded."""

    @staticmethod
    def encode(*, method: EesMethod) -> bytes:
        configuration = {
            key: getattr(method, key)
            for key in EesMethod.model_fields
            if key not in {"env", "skill_provider", "draw_recorder"}
        }
        return pickle.dumps(
            {
                "configuration": configuration,
                "samplers": method._samplers,
                "competence_models": [
                    (ground.skill.name, tuple(o.name for o in ground.objects), belief)
                    for ground, belief in method._competence_models.items()
                ],
                "rng": method._rng.bit_generator.state,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def restore(*, raw: bytes, env: Any, provider: Any) -> EesMethod:
        payload = pickle.loads(raw)  # trusted, locally produced experiment snapshots
        method = EesMethod(env=env, skill_provider=provider, **payload["configuration"])
        method._samplers = payload["samplers"]
        skills = {skill.name: skill for skill in provider.skills()}
        objects = {obj.name: obj for obj in provider.objects()}
        method._competence_models = {
            GroundSkill(skill=skills[name], objects=tuple(objects[obj] for obj in names)): belief
            for name, names, belief in payload["competence_models"]
            if name in skills  # Human actions are unavailable in deployment.
        }
        method._rng.bit_generator.state = payload["rng"]
        return method


class StepEvaluation:
    @staticmethod
    def run(
        *,
        snapshot: str,
        configuration: dict[str, Any],
        output: str,
        environment: type[StepProtocolEnvironment],
        checkpoint: int | None = None,
        label: str = "(final)",
    ) -> dict[str, Any]:
        """`label` names the update measured, for the person watching (`after cycle 3`)."""
        args = argparse.Namespace(**configuration)
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        raw = Path(snapshot).read_bytes()
        problem = environment.build_evaluation_problem(args=args)
        writer = environment.attach_recording(
            args=args, problem=problem, output_path=directory / RECORDING_FILENAME
        )
        results: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            problem.hard_reset()
            tasks = [problem.sample_test_task() for _ in range(args.num_test_tasks)]
            method = DeploymentSnapshot.restore(
                raw=raw,
                env=problem.env,
                provider=environment.evaluation_skill_provider(problem=problem),
            )
            for index, task in enumerate(tasks):

                def progress(*, task_number: int = index + 1) -> None:
                    environment.evaluation_progress(
                        problem=problem,
                        index=checkpoint or 0,
                        label=label,
                        task=task_number,
                        num_tasks=len(tasks),
                    )

                progress()
                state = problem.reset_to_task(task=task)
                seed = environment.task_seed(task=task)
                writer.begin_segment(
                    phase=Phase.EVALUATION, checkpoint=checkpoint, task_index=index, task_seed=seed
                )
                policy = method.get_task_policy(task=task)
                check_goal = environment.goal_check(problem=problem, task=task)
                clock = EvaluationClock(
                    budget=args.evaluation_control_steps,
                    check_goal=check_goal,
                    solved=check_goal(),
                )
                reason = "goal" if clock.solved else "step_budget"
                problem.env.set_control_step_observers(before=clock.before, after=clock.after)
                try:
                    while not clock.solved and clock.steps < args.evaluation_control_steps:
                        action = policy(state)
                        if np.array_equal(action.action, problem.env.noop_action()):
                            reason = "no_plan"
                            break
                        progress()
                        previous = clock.steps
                        try:
                            state = problem.take_action(action=action.action)
                        except ControlStepLimitReached:
                            environment.interrupted(problem=problem, action=action.action)
                            writer.robot_action(
                                label=action.label,
                                start_step=previous,
                                end_step=clock.steps,
                                success=False,
                                outcome=clock.interruption(),
                                goal_reached=clock.solved,
                            )
                            break
                        error = environment.last_skill_error(problem=problem)
                        StepFiles.event(
                            path=directory / "actions.jsonl",
                            task=index,
                            label=action.label,
                            start_step=previous,
                            end_step=clock.steps,
                            skill_error=error,
                            **StepFiles.trace(environment=environment, problem=problem),
                        )
                        writer.robot_action(
                            label=action.label,
                            start_step=previous,
                            end_step=clock.steps,
                            success=error is None,
                            outcome=error,
                        )
                        if clock.steps == previous:
                            reason = "zero_step_controller_failure"
                            break
                finally:
                    problem.env.set_control_step_observers(before=None, after=None)
                if clock.solved:
                    reason = "goal"
                results.append({
                    "task": index,
                    "seed": seed,
                    "goal": task.goal.describe(),
                    "solved": clock.solved,
                    "steps": clock.steps,
                    "reason": reason,
                })
                writer.end_segment(success=clock.solved, outcome=reason, steps=clock.steps)
                StepFiles.json(path=directory / "progress.json", value={"tasks": results})
            result = {
                "complete": True,
                "tasks": results,
                "num_solved": sum(r["solved"] for r in results),
                "num_total": len(results),
                **PolicyFileHash.record(raw=raw),
                "elapsed_seconds": time.monotonic() - started,
            }
            StepFiles.json(path=directory / "results.json", value=result)
            return result
        except BaseException:
            (directory / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            writer.close()
            problem.env.close()


class StepProtocolValidation:
    """Everything that can refuse a run, checked before anything is built or opened."""

    # Any of these in the output directory means a run already lives there. The state
    # log is opened in mode "w" while the problem is built, so the check must come first
    # or rerunning into an old directory truncates the run it then refuses to replace.
    RUN_OUTPUTS: ClassVar[tuple[str, ...]] = (
        "step_events.jsonl",
        SKILL_TIMING_FILENAME,
        RECORDING_FILENAME,
        "tossing3d_state_log.jsonl",
        "measurements.json",
        "step_protocol.json",
        "step_evaluations.json",
        "stats.json",
        "status.json",
        "snapshots",
        "evaluations",
    )

    @staticmethod
    def check(*, args: argparse.Namespace) -> None:
        if args.method not in {"ees", "pomdp"}:
            raise ValueError("Step protocol currently supports the EES and PDDL methods")
        if args.practice_reset_policy != PracticeResetPolicy.NEVER:
            raise ValueError(
                "Step protocol requires persistent practice (--practice-reset-policy never)"
            )
        if args.practice_reset_interval is not None or args.output_dir is None:
            raise ValueError("Step protocol requires an output directory and no automatic resets")
        if not args.defer_rendering:
            raise ValueError("Use --defer-rendering; videos are reconstructed from recorded states")
        if getattr(args, "record_episode_traces", False):
            raise ValueError(
                "The step protocol writes no --record-episode-traces file: it never runs "
                "run_task_episode. Every evaluation action is in evaluations/NNNN/"
                "actions.jsonl and recording.jsonl instead; drop the flag."
            )
        cost_path = getattr(args, "practice_cost_config", None)
        if cost_path and args.human_skill_steps != 1:
            raise ValueError("Configure human durations in the cost file, not --human-skill-steps")
        if (
            cost_path
            and getattr(args, "always_ask_reset", False)
            and SAMPLED_START_RESET_SKILL_NAME
            not in PracticeCosts.load(path=cost_path).human_skills
        ):
            # Otherwise the first forced reset raises mid-run, after the first sweep/toss.
            raise ValueError(
                f"{cost_path} does not price {SAMPLED_START_RESET_SKILL_NAME!r}, which "
                "--always-ask-reset requests"
            )
        output = Path(args.output_dir)
        if getattr(args, "resume", None) is not None:
            return  # Carries on the run there; `StepProtocolResume.prepare` checked it.
        existing = [name for name in StepProtocolValidation.RUN_OUTPUTS if (output / name).exists()]
        if existing:
            raise FileExistsError(
                f"Preserve the existing run in {output} (found {', '.join(existing)}); "
                "choose a new output directory"
            )


class StepProtocolEntry:
    @staticmethod
    def run(
        *,
        args: argparse.Namespace,
        method_cli: Any,
        env_cli: type[EnvironmentCli],
        environment: type[StepProtocolEnvironment],
    ) -> None:
        resuming = getattr(args, "resume", None) is not None
        if resuming:
            StepProtocolResume.prepare(args=args)
        StepProtocolValidation.check(args=args)
        if environment.evaluation_pool() is None:
            # Evaluations spawned beside practice never disturb its scene, and cannot
            # promise to be finished when a checkpoint is taken.
            if getattr(args, "measure_every_cycles", None):
                raise ValueError(
                    f"--measure-every-cycles needs in-process evaluations ({args.env})"
                )
            if resuming or getattr(args, "checkpoint_cycles", False):
                raise ValueError(
                    f"--checkpoint-cycles and --resume need in-process evaluations, which "
                    f"--env {args.env} does not run; only --env tossing3d_real resumes"
                )

        # Method CLI factories remain authoritative for all algorithm configuration.
        def run_method(*, args: Any, method_factory: Any, **unused: Any) -> None:
            problem = environment.build_practice_problem(args=args)
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            recording = output / RECORDING_FILENAME
            if resuming:
                # Its own file: the recording writer truncates, and the interrupted
                # process's recording is part of this run.
                earlier = len(list(output.glob(f"resume-*-{RECORDING_FILENAME}")))
                recording = output / f"resume-{earlier + 1:02d}-{RECORDING_FILENAME}"
            writer = environment.attach_recording(args=args, problem=problem, output_path=recording)
            context = environment.practice_context(args=args, problem=problem)
            saved = (
                PracticeCheckpoint.load(
                    path=output / CHECKPOINT_FILENAME,
                    live=PracticeCheckpoint.live_objects(
                        problem=problem, skill_provider=context.skill_provider
                    ),
                )
                if resuming
                else None
            )
            try:
                StepPracticeRunner.run(
                    args=args,
                    # A resumed run's method is the checkpoint's, never a fresh one.
                    method=method_factory(context) if saved is None else saved["method"],
                    problem=problem,
                    environment=environment,
                    evaluations=environment.evaluation_pool(),
                    recorder=writer,
                    saved=saved,
                )
            finally:
                writer.close()
                problem.env.close()

        # Named after the selected domain's CLI: method CLIs that support only some
        # domains (`--method pomdp`) dispatch on `env_cli.__name__`.
        step_cli = type(env_cli.__name__, (), {"run_method": staticmethod(run_method)})
        method_cli.run(args=args, env_cli=step_cli)


class StepPracticeRunner:
    # A run whose planner opens this many sessions in a row with a STOP has stopped
    # practising: nothing changes between them, so the next would stop too.
    REPEATED_STOP_LIMIT: ClassVar[int] = 10
    # A session of skills each refused at planning executes no control step, so the
    # next session plans from the same state and refuses the same way.
    REPEATED_ZERO_STEP_LIMIT: ClassVar[int] = 5
    # A run whose sessions are this many human resets in a row with no robot skill has
    # stopped practising too: only the person acts, and the robot never does.
    REPEATED_NO_ROBOT_ACTION_LIMIT: ClassVar[int] = 10

    @staticmethod
    def configure_accounting(
        *, args: argparse.Namespace, method: Any, problem: Any, output: Path
    ) -> PracticeAccounting | None:
        cost_path = getattr(args, "practice_cost_config", None)
        if not cost_path:
            return None
        accounting = PracticeAccounting(costs=PracticeCosts.load(path=cost_path))
        accounting.set_observation_provider(provider=problem.env.cost_observation)
        method.configure_practice_accounting(accounting=accounting)
        StepFiles.json(
            path=output / "practice_costs.json", value=accounting.costs.model_dump(mode="json")
        )
        return accounting

    @staticmethod
    def human_skill_names(*, environment: type[StepProtocolEnvironment]) -> dict[str, str]:
        """Every reset destination a practice method may request, mapped to the name it
        is charged and logged under: the domain's own, plus the full reset to a sampled
        start state, which `Problem` implements for any domain whose `Tasks` samples
        start states."""
        return {
            **environment.human_skill_names(),
            SAMPLED_START_RESET_DESTINATION: SAMPLED_START_RESET_SKILL_NAME,
        }

    @staticmethod
    def outcome(*, future: Any) -> tuple[str, Any]:
        """A finished evaluation future as plain data a checkpoint can carry."""
        try:
            return ("result", future.result())
        except Exception as error:  # noqa: BLE001 -- recorded, as the final report does
            return ("error", repr(error))

    @staticmethod
    def evaluation_status(
        *,
        records: list[dict[str, Any]],
        futures: list[tuple[dict[str, Any], Any]],
        completed_cycle: int,
        pending_evaluation: int | None,
    ) -> str | None:
        """A checkpoint's account of its cycle's evaluation: `evaluation 3 pending`, or
        `evaluation 3 done: 2/5` once recorded; `None` for a cycle with none."""
        if pending_evaluation is not None:
            return f"evaluation {len(records)} pending (after cycle {pending_evaluation})"
        if not futures or futures[-1][0].get("after_cycle") != completed_cycle:
            return None
        record, future = futures[-1]
        kind, value = StepPracticeRunner.outcome(future=future)
        if kind != "result":
            return f"evaluation {record['index']} failed: {value}"
        return f"evaluation {record['index']} done: {value['num_solved']}/{value['num_total']}"

    @staticmethod
    def completed_future(*, outcome: tuple[str, Any]) -> Future:
        future: Future = Future()
        kind, value = outcome
        if kind == "result":
            future.set_result(value)
        else:
            future.set_exception(RuntimeError(value))
        return future

    @staticmethod
    def set_aside(*, directory: Path) -> None:
        """Move an interrupted process's partial evaluation out of a resumed run's way:
        the checkpoint predates it, so the resumed run measures at the same index."""
        if not directory.exists():
            return
        attempt = 1
        while (aside := directory.with_name(f"{directory.name}.interrupted-{attempt}")).exists():
            attempt += 1
        directory.rename(aside)

    @staticmethod
    def record_resume(*, args: argparse.Namespace, events: Path, saved: dict[str, Any]) -> None:
        """Account in the event log for what the interrupted process did after its last
        checkpoint. Those events stay where they are; a cycle it had begun is named by a
        `cycle_discarded` event, its steps uncharged and its outcomes never learned
        from, since the method and clock carry on from the checkpoint before it."""
        header = saved["header"]
        lines_before, after = StepProtocolResume.interrupted_events(
            events=events, offset=header["event_log_bytes"]
        )
        first_line = lines_before + 1 if after else None
        last_line = lines_before + len(after) if after else None
        if any(event.get("event") == "session_start" for event in after):
            reached = max(
                (
                    event.get("end_step", event.get("practice_steps", header["practice_steps"]))
                    for event in after
                ),
                default=header["practice_steps"],
            )
            StepFiles.event(
                path=events,
                event="cycle_discarded",
                cycle=header["next_cycle"],
                first_line=first_line,
                last_line=last_line,
                practice_steps=header["practice_steps"],
                practice_steps_discarded=reached - header["practice_steps"],
                robot_actions_discarded=sum(event.get("event") == "robot" for event in after),
                human_resets_discarded=sum(event.get("event") == "human" for event in after),
                stops_discarded=sum(event.get("event") == "stop" for event in after),
            )
        snapshot = ConfigSnapshot.collect(args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH"))
        StepFiles.event(
            path=events,
            event="resume",
            cycle=header["next_cycle"],
            practice_steps=header["practice_steps"],
            learning_updates=header["learning_updates"],
            checkpoint_written_unix=header["written_unix"],
            interrupted_first_line=first_line,
            interrupted_last_line=last_line,
            git_commit=snapshot.git_commit,
            git_dirty=snapshot.git_dirty,
        )

    @staticmethod
    def session_endpoint(*, actions: int, stop_on_no_practice_action: bool) -> str | None:
        """The endpoint a finished session forces, or `None` to keep practising.

        A session with no actions ended runs unconditionally until the rule became
        opt-in: one session-opening STOP was final. Kept reachable to reproduce them."""
        if actions == 0 and stop_on_no_practice_action:
            return "no_practice_action"
        return None

    @staticmethod
    def run(
        *,
        args: argparse.Namespace,
        method: Any,
        problem: Any,
        environment: type[StepProtocolEnvironment],
        evaluations: Any = None,
        recorder: RunRecordWriter | None = None,
        saved: dict[str, Any] | None = None,
    ) -> Metrics:
        """Run persistent practice to the step budget. `StepProtocolValidation.check`
        must already have passed: by the time a problem exists, the state log beside
        these outputs has been opened, so the existing-run check cannot live here.

        `saved` is a loaded `PracticeCheckpoint` payload (`--resume`), whose `method`
        must be the `method` passed: the run carries on from the end of its cycle.

        `--evaluate-only` takes the untrained policy's measurement and stops there. The
        practice problem is built but never reset: its reset asks the person for a
        practice scene nothing would use, and the policy measured is the method the
        factory built, which no reset touches."""
        evaluate_only = getattr(args, "evaluate_only", False)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "snapshots").mkdir(exist_ok=saved is not None)
        configuration = vars(args).copy()
        events = output / "step_events.jsonl"
        timings = output / SKILL_TIMING_FILENAME
        metrics = Metrics()
        cycle = 0
        robot_invocations = 0
        learning_updates = 0
        human_skills = StepPracticeRunner.human_skill_names(environment=environment)
        # The domain's own destinations only, in their order, as before; a full reset
        # gets its key the first time one is granted.
        reset_counts = dict.fromkeys(environment.human_skill_names(), 0)
        records: list[dict[str, Any]] = []
        futures: list[tuple[dict[str, Any], Any]] = []
        started = time.monotonic()
        endpoint = "running"
        early_stop = False
        stop_streak = 0
        zero_step_streak = 0
        no_robot_action_streak = 0
        accounting: PracticeAccounting | None
        if saved is None:
            if not evaluate_only:
                problem.hard_reset()
            accounting = StepPracticeRunner.configure_accounting(
                args=args, method=method, problem=problem, output=output
            )
        else:
            # Before anything is written: a domain that cannot re-observe refuses here.
            environment.resume_practice_environment(problem=problem, state=saved["env_state"])
            RngStates.restore(owner=problem.tasks, states=saved["rngs"]["tasks"])
            RngStates.restore(owner=problem.env, states=saved["rngs"]["env"])
            RngStates.restore_global(states=saved["rngs"]["global"])
            accounting = saved["accounting"]
            metrics = saved["metrics"]
            cycle = saved["cycle"]
            robot_invocations = saved["robot_invocations"]
            learning_updates = saved["learning_updates"]
            reset_counts = saved["reset_counts"]
            records = saved["records"]
            futures = [
                (record, StepPracticeRunner.completed_future(outcome=outcome))
                for record, outcome in saved["evaluations"]
            ]
            stop_streak = saved["stop_streak"]
            # Absent from a checkpoint written before this streak existed: it then
            # starts again, which can only delay `repeated_zero_step`, never trip it.
            zero_step_streak = saved.get("zero_step_streak", 0)
            no_robot_action_streak = saved.get("no_robot_action_streak", 0)
            started -= saved["elapsed_seconds"]
            StepPracticeRunner.record_resume(args=args, events=events, saved=saved)
        recorder = recorder or RunRecordWriter.disabled()
        if not evaluate_only:  # A practice segment needs the scene no reset observed.
            recorder.begin_segment(phase=Phase.PRACTICE, scene_seed=args.seed)
        stop_on_no_practice_action = getattr(args, "stop_on_no_practice_action", False)
        zero_step_limit = getattr(
            args, "repeated_zero_step_limit", StepPracticeRunner.REPEATED_ZERO_STEP_LIMIT
        )
        pool = evaluations if evaluations is not None else FreshProcessEvaluations()

        def status() -> None:
            nonlocal early_stop
            early_stop = EvaluationStopping.reached(
                records=EvaluationStopping.completed(futures=futures),
                patience=getattr(args, "stop_after_perfect_evaluations", 0),
                num_tasks=args.num_test_tasks,
            )
            StepFiles.json(
                path=output / "status.json",
                value={
                    "phase": endpoint,
                    "updated_unix": time.time(),
                    "practice_steps": clock.steps,
                    "robot_steps": clock.robot_steps,
                    "human_invocations": clock.human_invocations,
                    "human_resets_by_destination": reset_counts,
                    "learning_updates": learning_updates,
                    "session_index": cycle,
                    "measurement_snapshots": len(records),
                    "evaluations_completed": sum(f.done() for _, f in futures),
                    "elapsed_seconds": time.monotonic() - started,
                    **(accounting.summary() if accounting else {}),
                },
            )

        def measure(*, after_cycle: int | None = None, before_practice: bool = False) -> None:
            label = "(final)" if after_cycle is None else f"after cycle {after_cycle}"
            if before_practice:
                label = "(before practice)"
            raw = DeploymentSnapshot.encode(method=method)
            index = len(records)
            path = output / "snapshots" / f"{index:04d}.pickle"
            path.write_bytes(raw)
            StepPracticeRunner.set_aside(directory=output / "evaluations" / f"{index:04d}")
            record = {
                "index": index,
                "practice_steps": clock.measurement_steps,
                "clock_steps": clock.steps,
                "robot_steps": clock.robot_steps,
                "human_invocations": clock.human_invocations,
                "human_resets_by_destination": dict(reset_counts),
                "robot_invocations": robot_invocations,
                "learning_updates": learning_updates,
                "session_index": cycle,
                **PolicyFileHash.record(raw=raw),
                **(accounting.summary() if accounting else {}),
                # Only a cycle-end measurement says so: every other record is unchanged.
                **({} if after_cycle is None else {"after_cycle": after_cycle}),
            }
            records.append(record)
            StepFiles.event(path=events, event="measurement", **record)
            futures.append((
                record,
                pool.submit(
                    fn=StepEvaluation.run,
                    snapshot=str(path),
                    configuration=configuration,
                    output=str(output / "evaluations" / f"{index:04d}"),
                    environment=environment,
                    checkpoint=index,
                    label=label,
                ),
            ))
            StepFiles.json(path=output / "measurements.json", value=records)
            status()

        clock = PracticeClock(
            # Never spent under `--evaluate-only`, which needs no budget.
            budget=args.practice_step_budget or 1,
            interval=args.measurement_interval_steps,
            human_steps=args.human_skill_steps,
            on_measure=measure,
            on_progress=status,
            accounting=accounting,
        )
        # `--measure-every-cycles K` (in-process evaluations only): after every K-th
        # learning update, measure the policy just updated, then have the person put
        # practice's scene back. A cycle the budget ends is never measured here, so the
        # final measurement is never a duplicate of one. With `--checkpoint-cycles` the
        # order is: learn, checkpoint (evaluation pending), evaluate, checkpoint again
        # (evaluation recorded), restore, practise -- so a resume re-runs an evaluation
        # that never finished and never one that did.
        measure_every_cycles = getattr(args, "measure_every_cycles", None)

        def learn(*, actions: int, reason: str) -> None:
            nonlocal learning_updates
            method.end_cycle()
            learning_updates += 1
            StepFiles.event(
                path=events,
                event="learning_update",
                cycle=cycle,
                practice_steps=clock.steps,
                actions=actions,
                reason=reason,
            )

        def restore_practice(*, scene: Any) -> None:
            restored = environment.restore_practice_scene(problem=problem, scene=scene)
            StepFiles.event(
                path=events,
                event="practice_scene_restored",
                cycle=cycle - 1,
                practice_steps=clock.steps,
                **restored,
            )

        def evaluate_and_restore(*, completed_cycle: int, scene: Any) -> None:
            """Measure the policy after `completed_cycle` (-1: the untrained one), then
            checkpoint and have the person put practice's `scene` back."""
            if completed_cycle < 0:
                measure(before_practice=True)
            else:
                measure(after_cycle=completed_cycle)
            if checkpoint_cycles:
                checkpoint(completed_cycle=completed_cycle, practice_scene=scene)
            restore_practice(scene=scene)

        if saved is not None:
            for field, value in saved["clock"].items():
                setattr(clock, field, value)
        checkpoint_cycles = getattr(args, "checkpoint_cycles", False)
        live = PracticeCheckpoint.live_objects(
            problem=problem, skill_provider=getattr(method, "skill_provider", None)
        )

        def checkpoint(
            *,
            completed_cycle: int,
            practice_scene: Any = None,
            pending_evaluation: int | None = None,
        ) -> None:
            """`practice_scene` is what a restore prompt asks for (a resume asks for it
            too); `pending_evaluation` the cycle whose evaluation is due and not yet run."""
            pending = sum(not future.done() for _, future in futures)
            if pending:
                raise RuntimeError(
                    f"a cycle checkpoint needs every evaluation finished; {pending} running"
                )
            evaluation = StepPracticeRunner.evaluation_status(
                records=records,
                futures=futures,
                completed_cycle=completed_cycle,
                pending_evaluation=pending_evaluation,
            )
            # The event first, then the checkpoint that points past it: an interruption
            # between the two leaves the previous checkpoint current, and its log
            # position puts this event in the cycle a resume discards.
            StepFiles.event(
                path=events,
                event="checkpoint",
                cycle=completed_cycle,
                next_cycle=cycle,
                practice_steps=clock.steps,
                learning_updates=learning_updates,
                **({} if evaluation is None else {"evaluation": evaluation}),
            )
            header = {
                "next_cycle": cycle,
                "practice_steps": clock.steps,
                "learning_updates": learning_updates,
                "human_invocations": clock.human_invocations,
                "event_log_bytes": events.stat().st_size,
                "written_unix": time.time(),
                "evaluation": evaluation,
            }
            PracticeCheckpoint.save(
                output=output,
                live=live,
                header=header,
                payload={
                    "header": header,
                    "method": method,
                    "accounting": accounting,
                    "metrics": metrics,
                    "cycle": cycle,
                    "robot_invocations": robot_invocations,
                    "learning_updates": learning_updates,
                    "reset_counts": reset_counts,
                    "records": records,
                    "evaluations": [
                        (record, StepPracticeRunner.outcome(future=future))
                        for record, future in futures
                    ],
                    "stop_streak": stop_streak,
                    "zero_step_streak": zero_step_streak,
                    "no_robot_action_streak": no_robot_action_streak,
                    "elapsed_seconds": time.monotonic() - started,
                    "clock": clock.model_dump(
                        include={
                            "robot_steps",
                            "stop_steps",
                            "human_invocations",
                            "counted_human_steps",
                        }
                    ),
                    "env_state": problem.get_current_state(),
                    "rngs": {
                        "tasks": RngStates.capture(owner=problem.tasks),
                        "env": RngStates.capture(owner=problem.env),
                        "global": RngStates.capture_global(),
                    },
                    "practice_scene": practice_scene,
                    "pending_evaluation": pending_evaluation,
                },
            )
            environment.run_saved(args=args, output=output, final=False)

        problem.env.set_control_step_observers(
            before=clock.before_robot_step, after=clock.after_robot_step
        )
        # A resumed run's protocol and config records stay as the first process wrote
        # them: they are what `--resume` was checked against.
        if saved is None:
            StepFiles.json(
                path=output / "step_protocol.json",
                value={
                    "practice_step_budget": args.practice_step_budget,
                    "evaluate_only": evaluate_only,
                    "measurement_interval_steps": clock.interval,
                    "human_skill_steps": clock.human_steps,
                    "evaluation_control_steps": args.evaluation_control_steps,
                    "learning_session_action_limit": args.max_steps_per_interaction,
                    "learning_cycle_count_limit": None,
                    "original_num_cycles_flag_superseded_by_step_budget": args.num_cycles,
                    "evaluation_process": "spawn, one fresh process per evaluation",
                    "evaluation_x_axis": "practice_steps",
                    "evaluation_feedback": False,
                    "normalized_costs": accounting is not None,
                    "stop_after_perfect_evaluations": getattr(
                        args, "stop_after_perfect_evaluations", 0
                    ),
                    "stop_on_no_practice_action": stop_on_no_practice_action,
                    "stop_skill_steps": 1,
                    "repeated_stop_limit": StepPracticeRunner.REPEATED_STOP_LIMIT,
                    "repeated_zero_step_limit": zero_step_limit,
                    "repeated_no_robot_action_limit": (
                        StepPracticeRunner.REPEATED_NO_ROBOT_ACTION_LIMIT
                    ),
                    "robot_duration_prior_steps": getattr(
                        method, "robot_duration_prior_steps", None
                    ),
                    "robot_skill_step_costs": getattr(method, "pomdp_skill_step_costs", None),
                },
            )
            StepFiles.json(
                path=output / "config_snapshot.json",
                value=ConfigSnapshot.collect(
                    args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH")
                ).model_dump(mode="json"),
            )
        budget_reached = False
        # The scene a human reset starts from: set by every session before one can run.
        state: Any = None

        def grant_human(
            *, destination: str, request_cost: float, forced: str | None = None
        ) -> bool:
            """One human reset through the practice clock; `False`, with the budget reached,
            when it does not fit in what remains."""
            nonlocal state, budget_reached
            name = human_skills[destination]
            try:
                charge = clock.before_human(skill=name)
            except ControlStepLimitReached:
                # The human does not fit in what remains: the budget is
                # reached here exactly as it is by a robot step, so practice
                # ends normally and the final measurement still runs.
                budget_reached = True
                StepFiles.event(
                    path=events,
                    event="human_refused_at_budget",
                    cycle=cycle,
                    practice_steps=clock.steps,
                    destination=destination,
                    skill=name,
                    cost=request_cost,
                )
                return False
            before_count = clock.steps
            before_cost = accounting.total_cost if accounting else 0
            method.observe_environment_reset(state=state)
            success = False
            # Discards a timing no event owns (the run's opening scene reset), so the
            # one written below is this reset's own.
            environment.drain_execution_timing(problem=problem)
            try:
                problem.execute_movables_reset(destination=destination)
                success = True
            finally:
                reset_counts[destination] = reset_counts.get(destination, 0) + 1
                clock.human_invoked(charge=charge)
                cost = accounting.total_cost - before_cost if accounting else request_cost
                metrics.record_human_intervention(cost=cost)
                method.observe_execution_cost(cost=cost, steps=charge.steps, complete=success)
                StepFiles.event(
                    path=events,
                    event="human",
                    cycle=cycle,
                    start_step=before_count,
                    end_step=clock.steps,
                    destination=destination,
                    skill=name,
                    cost=cost,
                    success=success,
                    **({"forced": forced} if forced else {}),
                )
                StepFiles.timing(
                    path=timings,
                    environment=environment,
                    problem=problem,
                    event="human",
                    cycle=cycle,
                    start_step=before_count,
                    end_step=clock.steps,
                    label=name,
                )
                recorder.human_action(
                    skill=name,
                    objects=(destination,),
                    start_step=before_count,
                    end_step=clock.steps,
                    success=success,
                    cost=cost,
                    cycle=cycle,
                )
            state = problem.get_current_state()
            method.observe_help_granted(state=state)
            return True

        try:
            # The untrained policy's measurement, never repeated by a resumed run.
            if evaluate_only:
                StepFiles.json(
                    path=output / "evaluate_only.json",
                    value={
                        "evaluate_only": True,
                        "meaning": "the untrained policy's step-0 measurement only: no "
                        "practice, no learning update, no checkpoint",
                        "practice_step_budget_ignored": args.practice_step_budget,
                    },
                )
                measure(before_practice=True)
                endpoint = "evaluate_only"
            elif getattr(args, "initial_measurement", True) and saved is None:
                if environment.evaluation_pool() is None:  # Spawned beside practice.
                    measure(before_practice=True)
                else:
                    # In this process, it uses the room practice's reset just set up. As a
                    # cycle-end evaluation is, it is checkpointed before the restore, so a
                    # run that dies in cycle 0 never re-runs the baseline.
                    evaluate_and_restore(
                        completed_cycle=-1, scene=environment.practice_scene(problem=problem)
                    )
            if saved is not None and saved.get("practice_scene") is not None:
                # The checkpoint's cycle-end evaluation, if it never finished, then the
                # person puts practice's scene back as the checkpoint remembered it.
                pending_evaluation = saved.get("pending_evaluation")
                if pending_evaluation is not None:
                    evaluate_and_restore(
                        completed_cycle=pending_evaluation, scene=saved["practice_scene"]
                    )
                else:
                    restore_practice(scene=saved["practice_scene"])
            while not evaluate_only and clock.steps < clock.budget:
                if (output / "STOP").exists():
                    endpoint = "external_stop"
                    break
                task = PracticeLoop._sample_practice_task(
                    problem=problem, practice_reset_policy=PracticeResetPolicy.NEVER
                )
                policy = method.get_practice_policy(task=task)
                state = problem.get_current_state()
                reason: Literal["session_action_cap", "planner_stop", "interaction_complete"] = (
                    "session_action_cap"
                )
                actions = 0
                robot_skills = 0
                stopped = False
                session_robot_steps = clock.robot_steps
                session_human_invocations = clock.human_invocations
                StepFiles.event(
                    path=events, event="session_start", cycle=cycle, practice_steps=clock.steps
                )
                recorder.begin_cycle(cycle=cycle, task=task, start_step=clock.steps)
                for action_index in range(args.max_steps_per_interaction):
                    status()
                    if clock.steps >= clock.budget or early_stop:
                        break
                    environment.practice_progress(
                        problem=problem,
                        cycle=cycle,
                        action=action_index + 1,
                        max_actions=args.max_steps_per_interaction,
                        steps=clock.steps,
                        budget=clock.budget,
                    )
                    method.observe_practice_action_budget(
                        remaining_actions=args.max_steps_per_interaction - action_index
                    )
                    try:
                        action = policy(state)
                    except HumanCubeBinResetRequested as request:
                        destination = request.destination
                        if destination not in human_skills:
                            raise ValueError(
                                f"Unexpected reset destination: {destination!r}"
                            ) from request
                        assert destination is not None
                        if not grant_human(destination=destination, request_cost=request.cost):
                            break
                        actions += 1
                        continue
                    except InteractionComplete as completion:
                        reason = (
                            "planner_stop" if completion.planner_stop else "interaction_complete"
                        )
                        before_count = clock.steps
                        before_cost = accounting.total_cost if accounting else 0
                        clock.stop_invoked()
                        stop_cost = accounting.total_cost - before_cost if accounting else None
                        StepFiles.event(
                            path=events,
                            event="stop",
                            cycle=cycle,
                            start_step=before_count,
                            end_step=clock.steps,
                            reason=reason,
                            cost=stop_cost,
                            success=True,
                        )
                        recorder.robot_action(
                            label="STOP",
                            start_step=before_count,
                            end_step=clock.steps,
                            success=True,
                            outcome=None,
                            cost=stop_cost,
                            cycle=cycle,
                        )
                        stopped = True
                        break
                    before_count = clock.steps
                    before_cost = accounting.total_cost if accounting else 0
                    robot_invocations += 1
                    clock.robot_active = True
                    interruption: str | None = None
                    completed = False
                    environment.drain_execution_timing(problem=problem)
                    try:
                        state = problem.take_action(action=action.action)
                        completed = True
                    except ControlStepLimitReached:
                        # `take_action` never reached its own bookkeeping, so the env
                        # still holds the pre-skill state and a cleared skill error.
                        interruption = "practice_step_budget"
                        state = environment.adopt_interrupted_skill(
                            problem=problem, action=action.action, reason=interruption
                        )
                    finally:
                        clock.robot_active = False
                        method.observe_execution_cost(
                            cost=accounting.total_cost - before_cost if accounting else 0,
                            steps=clock.steps - before_count,
                            complete=completed,
                        )
                    actions += 1
                    robot_skills += 1
                    skill_error = environment.last_skill_error(problem=problem)
                    robot_cost = accounting.total_cost - before_cost if accounting else None
                    StepFiles.event(
                        path=events,
                        event="robot",
                        cycle=cycle,
                        label=action.label,
                        action=action.action.tolist(),
                        start_step=before_count,
                        end_step=clock.steps,
                        interrupted=interruption is not None,
                        interruption=interruption,
                        cost=robot_cost,
                        skill_error=skill_error,
                        **StepFiles.trace(environment=environment, problem=problem),
                    )
                    StepFiles.timing(
                        path=timings,
                        environment=environment,
                        problem=problem,
                        event="robot",
                        cycle=cycle,
                        start_step=before_count,
                        end_step=clock.steps,
                        label=action.label,
                    )
                    # An adopted interruption is the skill's error, so it is the outcome.
                    recorder.robot_action(
                        label=action.label,
                        start_step=before_count,
                        end_step=clock.steps,
                        success=completed and skill_error is None,
                        outcome=skill_error,
                        cost=robot_cost,
                        cycle=cycle,
                    )
                    if interruption is not None:
                        budget_reached = True
                        break
                if early_stop:
                    endpoint = "three_perfect_evaluations"
                    break
                if budget_reached or clock.steps >= clock.budget:
                    endpoint = "practice_step_budget"
                    # The cycle the budget ended is learned from before the final
                    # measurement; otherwise a run of one cycle measures its prior.
                    if actions > 0:
                        learn(actions=actions, reason=endpoint)
                    break
                metrics.practice_session_ends.append(
                    PracticeSessionEnd(
                        cycle_index=cycle,
                        reason=reason,
                        actions_executed=actions,
                        action_limit=args.max_steps_per_interaction,
                    )
                )
                forced = StepPracticeRunner.session_endpoint(
                    actions=actions, stop_on_no_practice_action=stop_on_no_practice_action
                )
                if forced is not None:
                    endpoint = forced
                    break
                stop_streak = stop_streak + 1 if stopped and actions == 0 else 0
                # The always-ask wrapper's reset after a STOP-only session is that
                # session's own last act, so the session stays STOP-only for both streaks.
                owed = method.stop_only_session_reset() if stopped and actions == 0 else None
                if owed is not None and not grant_human(
                    destination=SAMPLED_START_RESET_DESTINATION,
                    request_cost=owed,
                    forced="stop_only_session",
                ):
                    endpoint = "practice_step_budget"
                    break
                if stop_streak >= StepPracticeRunner.REPEATED_STOP_LIMIT:
                    endpoint = "repeated_stop"
                    break
                # Human resets with no robot skill, STOP or not. A STOP-only session
                # (no actions) is `repeated_stop`'s: it neither extends nor breaks this.
                if robot_skills > 0:
                    no_robot_action_streak = 0
                elif actions > 0:
                    no_robot_action_streak += 1
                if no_robot_action_streak >= StepPracticeRunner.REPEATED_NO_ROBOT_ACTION_LIMIT:
                    endpoint = "repeated_no_robot_action"
                    break
                # A STOP's own charge is not a control step, and a STOP-only session
                # (no actions) is `repeated_stop`'s: it neither extends nor breaks this.
                if actions > 0:
                    progressed = (
                        clock.robot_steps > session_robot_steps
                        or clock.human_invocations > session_human_invocations
                    )
                    zero_step_streak = 0 if progressed else zero_step_streak + 1
                if zero_step_streak >= zero_step_limit:
                    endpoint = "repeated_zero_step"
                    break
                learn(actions=actions, reason=reason)
                cycle += 1
                due = (
                    measure_every_cycles is not None
                    and learning_updates % measure_every_cycles == 0
                )
                # Remembered before the evaluation uses the room.
                scene = (
                    environment.practice_scene(problem=problem) if measure_every_cycles else None
                )
                if checkpoint_cycles:
                    checkpoint(
                        completed_cycle=cycle - 1,
                        practice_scene=scene,
                        pending_evaluation=cycle - 1 if due else None,
                    )
                if due:
                    evaluate_and_restore(completed_cycle=cycle - 1, scene=scene)
                status()
            if endpoint == "running":
                endpoint = "practice_step_budget"
        except BaseException:
            endpoint = "failed"
            (output / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            problem.env.set_control_step_observers(before=None, after=None)
            clock.robot_active = False
            # Under `--measure-every-cycles`, a policy no update has changed since the last
            # cycle-end measurement is not measured again (each is a person's scene set-ups).
            unchanged = bool(
                measure_every_cycles
                and records
                and records[-1]["learning_updates"] == learning_updates
            )
            if (
                endpoint not in {"failed", "evaluate_only"}
                and not unchanged
                and (not records or records[-1]["practice_steps"] != clock.steps)
            ):
                measure()
            StepFiles.event(
                path=events, event="practice_end", practice_steps=clock.steps, reason=endpoint
            )
            if not evaluate_only:
                recorder.end_segment(outcome=endpoint, steps=clock.steps)
            status()
            if checkpoint_cycles:
                environment.run_saved(args=args, output=output, final=True)
            failed = endpoint == "failed"
            if failed:
                # A crash or Ctrl-C must not sit behind the evaluation queue: cancel what
                # has not started, terminate what has, and record both as cancelled.
                pool.abort()
            else:
                pool.finish()
            evaluation_records = []
            for record, future in futures:
                if future.cancelled() or (failed and not future.done()):
                    evaluation_records.append({
                        **record,
                        "complete": False,
                        "cancelled": True,
                        "error": "cancelled: practice ended in failure",
                    })
                    continue
                try:
                    result = future.result()
                    # x is the measurement's own practice step, unique and increasing,
                    # so mid-skill and final measurements never share a point and
                    # human invocations advance the axis like control steps do.
                    metrics.record_evaluation(
                        num_online_transitions=record["practice_steps"],
                        num_solved=result["num_solved"],
                        num_total=result["num_total"],
                        outcomes=tuple(
                            TaskOutcome(
                                goal=r["goal"],
                                task_index=r["task"],
                                solved=r["solved"],
                            )
                            for r in result["tasks"]
                        ),
                    )
                    evaluation_records.append({**record, **result})
                except Exception as exc:
                    evaluation_records.append({
                        **record,
                        "complete": False,
                        "cancelled": False,
                        "error": repr(exc),
                    })
            StepFiles.json(path=output / "step_evaluations.json", value=evaluation_records)
            StepFiles.json(path=output / "stats.json", value=metrics.model_dump(mode="json"))
            evaluation_failures = sum(not r["complete"] for r in evaluation_records)
            endpoint = (
                "failed"
                if endpoint == "failed"
                else "evaluation_failed"
                if evaluation_failures
                else "complete"
            )
            status()
        return metrics
