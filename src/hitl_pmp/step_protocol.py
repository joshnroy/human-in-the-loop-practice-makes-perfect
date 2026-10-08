"""Tossing3D practice measured in control steps, independently of learning sessions.

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
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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

    @property
    def steps(self) -> int:
        return self.robot_steps + self.counted_human_steps

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
        if self.steps // self.interval > previous // self.interval:
            self.on_measure()
        if self.steps % 100 == 0:
            self.on_progress()


class EvaluationStopping:
    @staticmethod
    def reached(*, records: list[dict[str, Any]], patience: int = 3) -> bool:
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
            perfect = record.get("num_solved") == record.get("num_total") == 10
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
    def run(*, snapshot: str, configuration: dict[str, Any], output: str) -> dict[str, Any]:
        # Import here to avoid a cycle with the domain's CLI composition root.
        from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
        from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
        from hitl_pmp.environments.tossing3d.state_log import StateLogHeader, StateLogWriter

        args = argparse.Namespace(**configuration)
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        raw = Path(snapshot).read_bytes()
        problem = Tossing3DCli.build_evaluation_problem(args=args)
        writer = StateLogWriter(
            output_path=directory / "states.jsonl",
            header=StateLogHeader(
                layout=problem.env.layout,
                variant=args.variant,
                scene_bg=args.scene_bg,
                canonical_seed=args.canonical_seed,
                seed=args.seed,
                test_env_seed_offset=args.test_env_seed_offset,
            ),
        )
        problem.env.attach_state_log_writer(writer=writer)
        results: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            problem.hard_reset()
            tasks = [problem.sample_test_task() for _ in range(args.num_test_tasks)]
            method = DeploymentSnapshot.restore(
                raw=raw,
                env=problem.env,
                provider=Tossing3DSkillProvider(env=problem.env, offer_human_reset=False),
            )
            for index, task in enumerate(tasks):
                state = problem.reset_to_task(task=task)
                policy = method.get_task_policy(task=task)
                backend = problem.env.backend()
                clock = EvaluationClock(
                    budget=args.evaluation_control_steps,
                    check_goal=backend.check_goals,
                    solved=backend.check_goals(),
                )
                reason = "goal" if clock.solved else "step_budget"
                backend.set_control_step_observers(before=clock.before, after=clock.after)
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
                            problem.env._log_skill_ticks(action=action.action)
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
                    backend.set_control_step_observers(before=None, after=None)
                if clock.solved:
                    reason = "goal"
                results.append({
                    "task": index,
                    "seed": int(task.initial_state.get(obj=problem.env.scene, feature_name="seed")),
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
            writer.close()
            problem.env.close()


class StepProtocolEntry:
    @staticmethod
    def run(*, args: argparse.Namespace, method_cli: Any) -> None:
        from hitl_pmp.core.method.skill_provider import DomainContext
        from hitl_pmp.environments.tossing3d.cli import Tossing3DCli as DomainCli
        from hitl_pmp.environments.tossing3d.skill_provider import (
            Tossing3DOracle,
            Tossing3DSkillProvider,
        )
        from hitl_pmp.environments.tossing3d.state_log import StateLogHeader, StateLogWriter

        # Method CLI factories remain authoritative for all algorithm configuration.
        class Tossing3DCli:
            @staticmethod
            def run_method(*, args: Any, method_factory: Any, **unused: Any) -> None:
                problem = DomainCli.build_practice_problem(args=args)
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                writer = StateLogWriter(
                    output_path=output / "tossing3d_state_log.jsonl",
                    header=StateLogHeader(
                        layout=problem.env.layout,
                        variant=args.variant,
                        scene_bg=args.scene_bg,
                        canonical_seed=args.canonical_seed,
                        seed=args.seed,
                        test_env_seed_offset=args.test_env_seed_offset,
                    ),
                )
                problem.env.attach_state_log_writer(writer=writer)
                context = DomainContext(
                    env=problem.env,
                    oracle=Tossing3DOracle(
                        env=problem.env, throw_standoff=args.oracle_throw_standoff
                    ),
                    skill_provider=Tossing3DSkillProvider(
                        env=problem.env,
                        human_reset_practice_cost=args.human_reset_practice_cost,
                        offer_human_reset=args.human_reset,
                    ),
                )
                try:
                    StepPracticeRunner.run(
                        args=args, method=method_factory(context), problem=problem
                    )
                finally:
                    writer.close()
                    problem.env.close()

        method_cli.run(args=args, env_cli=Tossing3DCli)


class StepPracticeRunner:
    @staticmethod
    def run(*, args: argparse.Namespace, method: Any, problem: Any) -> Metrics:
        if args.env != "tossing3d" or args.method not in {"ees", "pomdp"}:
            raise ValueError("Step protocol currently supports Tossing3D EES and PDDL")
        if args.practice_reset_policy != PracticeResetPolicy.NEVER:
            raise ValueError(
                "Step protocol requires persistent practice (--practice-reset-policy never)"
            )
        if args.practice_reset_interval is not None or args.output_dir is None:
            raise ValueError("Step protocol requires an output directory and no automatic resets")
        if not args.defer_rendering:
            raise ValueError("Use --defer-rendering; videos are reconstructed from recorded states")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "step_events.jsonl").exists():
            raise FileExistsError("Preserve the existing run; choose a new output directory")
        (output / "snapshots").mkdir()
        configuration = vars(args).copy()
        events = output / "step_events.jsonl"
        metrics = Metrics()
        cycle = 0
        robot_invocations = 0
        learning_updates = 0
        reset_counts = {"robot_side": 0, "opposite_side": 0}
        records: list[dict[str, Any]] = []
        futures: list[tuple[dict[str, Any], Any]] = []
        started = time.monotonic()
        endpoint = "running"
        early_stop = False
        problem.hard_reset()
        cost_path = getattr(args, "practice_cost_config", None)
        accounting = (
            PracticeAccounting(costs=PracticeCosts.load(path=cost_path)) if cost_path else None
        )
        if accounting is not None:
            if args.human_skill_steps != 1:
                raise ValueError(
                    "Configure human durations in the cost file, not --human-skill-steps"
                )
            from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge

            cost_bridge = Tossing3DAgenticBridge(env=problem.env, observation_mode="object_state")
            accounting.set_observation_provider(provider=cost_bridge.observe)
            method.configure_practice_accounting(accounting=accounting)
            StepFiles.json(
                path=output / "practice_costs.json", value=accounting.costs.model_dump(mode="json")
            )
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

        def status() -> None:
            nonlocal early_stop
            early_stop = EvaluationStopping.reached(
                records=EvaluationStopping.completed(futures=futures),
                patience=getattr(args, "stop_after_perfect_evaluations", 0),
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
                "practice_steps": clock.steps,
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
                    StepEvaluation.run,
                    snapshot=str(path),
                    configuration=configuration,
                    output=str(output / "evaluations" / f"{index:04d}"),
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
        backend = problem.env.backend()
        backend.set_control_step_observers(
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
                "evaluation_process": "spawn",
                "evaluation_feedback": False,
                "normalized_costs": accounting is not None,
                "stop_after_perfect_evaluations": getattr(
                    args, "stop_after_perfect_evaluations", 0
                ),
                "robot_duration_prior_steps": getattr(method, "robot_duration_prior_steps", None),
            },
        )
        StepFiles.json(
            path=output / "config_snapshot.json",
            value=ConfigSnapshot.collect(
                args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH")
            ).model_dump(mode="json"),
        )
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
                        if destination not in {"robot_side", "opposite_side"}:
                            raise ValueError(
                                f"Unexpected reset destination: {destination!r}"
                            ) from request
                        assert destination is not None
                        name = (
                            "reset_cube_and_bin_near"
                            if destination == "robot_side"
                            else "reset_cube_far"
                        )
                        charge = clock.before_human(skill=name)
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
                    interrupted = False
                    completed = False
                    try:
                        state = problem.take_action(action=action.action)
                        completed = True
                    except ControlStepLimitReached:
                        problem.env._log_skill_ticks(action=action.action)
                        interrupted = True
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
                        interrupted=interrupted,
                        cost=accounting.total_cost - before_cost if accounting else None,
                        skill_error=problem.env.last_skill_error(),
                    )
                    if interrupted:
                        break
                if early_stop:
                    endpoint = "three_perfect_evaluations"
                    break
                if clock.steps >= clock.budget:
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
                if actions == 0:
                    endpoint = "no_practice_action"
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
        except ControlStepLimitReached:
            endpoint = "remaining_budget_below_action_duration"
        except BaseException:
            endpoint = "failed"
            (output / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            backend.set_control_step_observers(before=None, after=None)
            clock.robot_active = False
            if endpoint != "failed" and (
                not records or records[-1]["practice_steps"] != clock.steps
            ):
                measure()
            StepFiles.event(
                path=events, event="practice_end", practice_steps=clock.steps, reason=endpoint
            )
            status()
            pool.shutdown(wait=True)
            evaluation_records = []
            for record, future in futures:
                try:
                    result = future.result()
                    metrics.record_evaluation(
                        num_online_transitions=record["robot_invocations"],
                        num_solved=result["num_solved"],
                        num_total=result["num_total"],
                        outcomes=tuple(
                            TaskOutcome(
                                goal="InBin(cube_0, bin_0)",
                                task_index=r["task"],
                                solved=r["solved"],
                            )
                            for r in result["tasks"]
                        ),
                    )
                    evaluation_records.append({**record, **result})
                except Exception as exc:
                    evaluation_records.append({**record, "complete": False, "error": repr(exc)})
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
