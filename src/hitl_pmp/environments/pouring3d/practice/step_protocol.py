"""Tick accounting, bounded interruption, and full native motion recording."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import FrozenPourCli
from .environment import FrozenPourEnvironment
from .problem import FrozenPourProblem
from .skill_provider import FrozenPourSkillProvider
from .skills import RESET_NAME
from .types import PourObjects


class FrozenPourStepProtocol(SimulatedStepProtocolHooks):
    @staticmethod
    def env(*, problem: Problem) -> FrozenPourEnvironment:
        assert isinstance(problem.env, FrozenPourEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> FrozenPourProblem:
        return FrozenPourCli.build_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> FrozenPourProblem:
        return FrozenPourCli.build_problem(args=args)

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        assert isinstance(problem, FrozenPourProblem)
        return FrozenPourCli.context(args=args, problem=problem)

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> FrozenPourSkillProvider:
        return FrozenPourSkillProvider(
            env=FrozenPourStepProtocol.env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def live_state(*, env: FrozenPourEnvironment) -> dict[str, list[float]]:
        c = env.controller()
        return {
            "qpos": c.data.qpos.tolist(),
            "qvel": c.data.qvel.tolist(),
            "time": [float(c.data.time)],
            "request": [float(env.requested_count)],
        }

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = FrozenPourStepProtocol.env(problem=problem)
        assert isinstance(problem, FrozenPourProblem)
        motion = args.record_motion == "all" or (
            args.record_motion == "evaluation" and "evaluations" in output_path.parts
        )
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="pouring3d",
                method=args.method,
                run_seed=args.seed,
                scene={
                    "payload_count": env.payload_count,
                    "requested_count": env.requested_count,
                    "surface_mu": env.surface_mu,
                    "scene_bg": env.scene_bg,
                    "motion": motion,
                    "fps": 10,
                    "practice_model_version": 2,
                    "requested_counts": list(
                        problem.tasks.requested_counts or (env.requested_count,)
                    ),
                },
            ),
            state_provider=lambda: FrozenPourStepProtocol.live_state(env=env),
            task_seed_provider=lambda task: FrozenPourStepProtocol.task_seed(task=task),
        )
        env.set_skill_sink(sink=writer.record_skill)
        if motion:
            env.set_motion_sink(
                sink=lambda **_: writer.record_tick(
                    state=FrozenPourStepProtocol.live_state(env=env)
                )
            )
        return writer

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {RESET_NAME: RESET_NAME}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(task.initial_state.get(obj=PourObjects.scene, feature_name="seed"))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        env = FrozenPourStepProtocol.env(problem=problem)
        return lambda: task.goal.is_satisfied(state=env.observe())

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return FrozenPourStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        info = FrozenPourStepProtocol.env(problem=problem).last_info
        if info.get("error") is not None:
            return str(info["error"])
        if info.get("success") is False:
            return f"{info.get('skill', 'Skill')}: observed success conditions not achieved"
        return None

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        return FrozenPourStepProtocol.env(problem=problem).last_info or None
