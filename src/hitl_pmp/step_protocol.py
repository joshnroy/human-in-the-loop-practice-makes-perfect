"""Practice measured in control steps, independently of learning sessions.

Any domain registered with a `StepProtocolEnvironment` (`cli_protocols.py`) runs here.

The training process publishes immutable deployment snapshots. Spawned workers own
their environments, policies and RNGs; they return measurements only to the logger.
"""

# ruff: noqa: SLF001
import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
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
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop, PracticeResetPolicy


class PracticeClock(BaseModel):
    """Only actual practice controller calls and human invocations advance this clock."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    budget: int = Field(gt=0)
    interval: int = Field(gt=0)
    human_steps: int = Field(default=1, gt=0)
    robot_steps: int = 0
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
        return self.robot_steps + self.counted_human_steps

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
    def event(*, path: Path, **record: Any) -> None:
        with path.open("a") as stream:
            stream.write(LogTiming.encode(record=record))


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
    ) -> dict[str, Any]:
        args = argparse.Namespace(**configuration)
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        raw = Path(snapshot).read_bytes()
        problem = environment.build_evaluation_problem(args=args)
        close_log = environment.attach_state_log(
            args=args, problem=problem, directory=directory, evaluation=True
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
                state = problem.reset_to_task(task=task)
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
                        previous = clock.steps
                        try:
                            state = problem.take_action(action=action.action)
                        except ControlStepLimitReached:
                            environment.interrupted(problem=problem, action=action.action)
                            break
                        StepFiles.event(
                            path=directory / "actions.jsonl",
                            task=index,
                            label=action.label,
                            start_step=previous,
                            end_step=clock.steps,
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
                    "seed": environment.task_seed(task=task),
                    "solved": clock.solved,
                    "steps": clock.steps,
                    "reason": reason,
                })
                StepFiles.json(path=directory / "progress.json", value={"tasks": results})
            result = {
                "complete": True,
                "tasks": results,
                "num_solved": sum(r["solved"] for r in results),
                "num_total": len(results),
                "policy_sha256": hashlib.sha256(raw).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
            }
            StepFiles.json(path=directory / "results.json", value=result)
            return result
        except BaseException:
            (directory / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            close_log()
            problem.env.close()


class StepProtocolValidation:
    """Everything that can refuse a run, checked before anything is built or opened."""

    # Any of these in the output directory means a run already lives there. The state
    # log is opened in mode "w" while the problem is built, so the check must come first
    # or rerunning into an old directory truncates the run it then refuses to replace.
    RUN_OUTPUTS: ClassVar[tuple[str, ...]] = (
        "step_events.jsonl",
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
        if getattr(args, "practice_cost_config", None) and args.human_skill_steps != 1:
            raise ValueError("Configure human durations in the cost file, not --human-skill-steps")
        output = Path(args.output_dir)
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
        StepProtocolValidation.check(args=args)

        # Method CLI factories remain authoritative for all algorithm configuration.
        def run_method(*, args: Any, method_factory: Any, **unused: Any) -> None:
            problem = environment.build_practice_problem(args=args)
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            close_log = environment.attach_state_log(
                args=args, problem=problem, directory=output, evaluation=False
            )
            context = environment.practice_context(args=args, problem=problem)
            try:
                StepPracticeRunner.run(
                    args=args,
                    method=method_factory(context),
                    problem=problem,
                    environment=environment,
                )
            finally:
                close_log()
                problem.env.close()

        # Named after the selected domain's CLI: method CLIs that support only some
        # domains (`--method pomdp`) dispatch on `env_cli.__name__`.
        step_cli = type(env_cli.__name__, (), {"run_method": staticmethod(run_method)})
        method_cli.run(args=args, env_cli=step_cli)


class StepPracticeRunner:
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
    ) -> Metrics:
        """Run persistent practice to the step budget. `StepProtocolValidation.check`
        must already have passed: by the time a problem exists, the state log beside
        these outputs has been opened, so the existing-run check cannot live here."""
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "snapshots").mkdir()
        configuration = vars(args).copy()
        events = output / "step_events.jsonl"
        metrics = Metrics()
        cycle = 0
        robot_invocations = 0
        learning_updates = 0
        human_skills = environment.human_skill_names()
        reset_counts = dict.fromkeys(human_skills, 0)
        records: list[dict[str, Any]] = []
        futures: list[tuple[dict[str, Any], Any]] = []
        started = time.monotonic()
        endpoint = "running"
        early_stop = False
        problem.hard_reset()
        accounting = StepPracticeRunner.configure_accounting(
            args=args, method=method, problem=problem, output=output
        )
        stop_on_no_practice_action = getattr(args, "stop_on_no_practice_action", False)
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

        def measure() -> None:
            raw = DeploymentSnapshot.encode(method=method)
            index = len(records)
            path = output / "snapshots" / f"{index:04d}.pickle"
            path.write_bytes(raw)
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
                "policy_sha256": hashlib.sha256(raw).hexdigest(),
                **(accounting.summary() if accounting else {}),
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
                ),
            ))
            StepFiles.json(path=output / "measurements.json", value=records)
            status()

        clock = PracticeClock(
            budget=args.practice_step_budget,
            interval=args.measurement_interval_steps,
            human_steps=args.human_skill_steps,
            on_measure=measure,
            on_progress=status,
            accounting=accounting,
        )
        problem.env.set_control_step_observers(
            before=clock.before_robot_step, after=clock.after_robot_step
        )
        StepFiles.json(
            path=output / "step_protocol.json",
            value={
                "practice_step_budget": clock.budget,
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
                "robot_duration_prior_steps": getattr(method, "robot_duration_prior_steps", None),
            },
        )
        StepFiles.json(
            path=output / "config_snapshot.json",
            value=ConfigSnapshot.collect(
                args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH")
            ).model_dump(mode="json"),
        )
        budget_reached = False
        try:
            measure()
            while clock.steps < clock.budget:
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
                StepFiles.event(
                    path=events, event="session_start", cycle=cycle, practice_steps=clock.steps
                )
                for action_index in range(args.max_steps_per_interaction):
                    status()
                    if clock.steps >= clock.budget or early_stop:
                        break
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
                                cost=request.cost,
                            )
                            break
                        before_count = clock.steps
                        before_cost = accounting.total_cost if accounting else 0
                        method.observe_environment_reset(state=state)
                        success = False
                        try:
                            problem.execute_movables_reset(destination=destination)
                            success = True
                        finally:
                            reset_counts[destination] += 1
                            clock.human_invoked(charge=charge)
                            cost = (
                                accounting.total_cost - before_cost if accounting else request.cost
                            )
                            metrics.record_human_intervention(cost=cost)
                            method.observe_execution_cost(
                                cost=cost, steps=charge.steps, complete=success
                            )
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
                            )
                        state = problem.get_current_state()
                        method.observe_help_granted(state=state)
                        actions += 1
                        continue
                    except InteractionComplete as completion:
                        reason = (
                            "planner_stop" if completion.planner_stop else "interaction_complete"
                        )
                        break
                    before_count = clock.steps
                    before_cost = accounting.total_cost if accounting else 0
                    robot_invocations += 1
                    clock.robot_active = True
                    interruption: str | None = None
                    completed = False
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
                        cost=accounting.total_cost - before_cost if accounting else None,
                        skill_error=environment.last_skill_error(problem=problem),
                    )
                    if interruption is not None:
                        budget_reached = True
                        break
                if early_stop:
                    endpoint = "three_perfect_evaluations"
                    break
                if budget_reached or clock.steps >= clock.budget:
                    endpoint = "practice_step_budget"
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
                cycle += 1
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
            if endpoint != "failed" and (
                not records or records[-1]["practice_steps"] != clock.steps
            ):
                measure()
            StepFiles.event(
                path=events, event="practice_end", practice_steps=clock.steps, reason=endpoint
            )
            status()
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
                                goal=environment.goal_description(),
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
