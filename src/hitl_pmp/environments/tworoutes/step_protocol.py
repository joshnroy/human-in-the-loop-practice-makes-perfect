"""The step protocol's hooks for Two Routes: one control step per robot skill."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import ATTEMPTS_LOG_FILENAME, TwoRoutesCli
from .environment import HUMAN_RESET_B, ROBOT_SKILL_NAMES, ROUTE_A, ROUTE_B, TwoRoutesEnvironment
from .problem import TwoRoutesProblem
from .skill_provider import TwoRoutesSkillProvider


class TwoRoutesStepProtocol(SimulatedStepProtocolHooks):
    @staticmethod
    def env(*, problem: Problem) -> TwoRoutesEnvironment:
        assert isinstance(problem.env, TwoRoutesEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> TwoRoutesProblem:
        return TwoRoutesCli.build_problem(args=args, practice=True)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> TwoRoutesProblem:
        problem = TwoRoutesCli.build_problem(args=args, practice=False)
        # Kept for `evaluation_progress`, which is told the checkpoint but not the run.
        problem.env._practice_output = Path(args.output_dir)
        return problem

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        assert isinstance(problem, TwoRoutesProblem)
        return TwoRoutesCli.context(args=args, problem=problem)

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> TwoRoutesSkillProvider:
        return TwoRoutesSkillProvider(
            env=TwoRoutesStepProtocol.env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def robot_invocations_at(*, output: Path, checkpoint: int) -> int:
        """How many robot actions practice had run when measurement `checkpoint` was
        taken. Read from the event log, which the runner writes before it spawns the
        evaluation."""
        for line in (output / "step_events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "measurement" and event.get("index") == checkpoint:
                return int(event["robot_invocations"])
        raise ValueError(f"No measurement {checkpoint} in {output / 'step_events.jsonl'}")

    @staticmethod
    def evaluation_progress(
        *, problem: Problem, index: int, label: str, task: int, num_tasks: int
    ) -> None:
        """Before each held-out task: the routes are as practised as they were when
        this evaluation's snapshot was taken."""
        del label, task, num_tasks
        env = TwoRoutesStepProtocol.env(problem=problem)
        output = env._practice_output
        if output is None:
            return
        env.adopt_practice_attempts(
            log=output / ATTEMPTS_LOG_FILENAME,
            robot_invocations=TwoRoutesStepProtocol.robot_invocations_at(
                output=output, checkpoint=index
            ),
        )

    @staticmethod
    def live_state(*, env: TwoRoutesEnvironment) -> dict[str, list[float]]:
        return {
            "robot": [float(v) for v in env.get_current_state()[env.robot]],
            "attempts": [float(env.attempts[ROUTE_A]), float(env.attempts[ROUTE_B])],
        }

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = TwoRoutesStepProtocol.env(problem=problem)
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(environment="tworoutes", method=args.method, run_seed=args.seed),
            state_provider=lambda: TwoRoutesStepProtocol.live_state(env=env),
        )

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {HUMAN_RESET_B: HUMAN_RESET_B}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        del task
        return 0

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        env = TwoRoutesStepProtocol.env(problem=problem)
        return lambda: task.goal.is_satisfied(state=env.get_current_state())

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        """A refused step changed nothing, so the scene is the one before it."""
        del action
        env = TwoRoutesStepProtocol.env(problem=problem)
        env.last_error = reason
        return env.get_current_state()

    @staticmethod
    def one_shot_failure_reason(*, action: Action) -> str | None:
        """A route attempt is the task's one attempt (`--evaluation-max-pour-attempts 1`)."""
        values = np.asarray(action, dtype=float)
        if values.shape != (1,) or not 0 <= values[0] < len(ROBOT_SKILL_NAMES):
            return None
        skill = ROBOT_SKILL_NAMES[int(round(values[0]))]
        return "route_failed" if skill in {ROUTE_A, ROUTE_B} else None

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return TwoRoutesStepProtocol.env(problem=problem).last_error

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        env = TwoRoutesStepProtocol.env(problem=problem)
        return {
            "attempts": dict(env.attempts),
            "success_probability": {
                skill: env.success_probability(skill=skill) for skill in (ROUTE_A, ROUTE_B)
            },
        }
