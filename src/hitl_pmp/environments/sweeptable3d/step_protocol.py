"""SweepTable3D's counted-control-step evaluation and recording hooks."""

import argparse
import functools
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

from .backend import SweepTable3DBackend
from .cli import SweepTable3DCli
from .environment import SweepTable3DEnvironment
from .objects import SweepTable3DObjects as O
from .problem import SweepTable3DProblem
from .recording import SweepTable3DReplayRenderer
from .skill_provider import SweepTable3DOracle, SweepTable3DSkillProvider
from .skill_trace import SkillTraces
from .skills import RESET_CUBES_NAME, RESET_WIPER_NAME


class SweepTable3DStepProtocol(SimulatedStepProtocolHooks):
    @staticmethod
    def env(*, problem: Problem) -> SweepTable3DEnvironment:
        assert isinstance(problem.env, SweepTable3DEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> SweepTable3DProblem:
        practice, _ = SweepTable3DCli.MOTION_SCOPES[getattr(args, "record_motion", "off")]
        return SweepTable3DCli.build_problem(
            args=args, human_seed=args.seed, motion_recording=practice
        )

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> SweepTable3DProblem:
        _, evaluation = SweepTable3DCli.MOTION_SCOPES[getattr(args, "record_motion", "off")]
        return SweepTable3DCli.build_problem(
            args=args, human_seed=args.seed + 1, motion_recording=evaluation
        )

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        env = SweepTable3DStepProtocol.env(problem=problem)
        return DomainContext(
            env=env,
            skill_provider=SweepTable3DSkillProvider(
                env=env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            ),
            oracle=SweepTable3DOracle(env=env),
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> SweepTable3DSkillProvider:
        return SweepTable3DSkillProvider(
            env=SweepTable3DStepProtocol.env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = SweepTable3DStepProtocol.env(problem=problem)
        scene: dict[str, Any] = {
            "scene_bg": env.scene_bg,
            "canonical_seed": env.canonical_seed,
            "reset_cubes_success_probability": env.reset_cubes_success_probability,
            "fixed_bin_offset": env.fixed_bin_offset,
            "test_env_seed_offset": args.test_env_seed_offset,
        }
        state: Callable[[], dict[str, list[float]]] = functools.partial(
            SweepTable3DStepProtocol.live_state, env=env
        )
        if env.motion_recording:
            state = env.backend().motion_state
            scene["motion"] = {
                "fps": SweepTable3DReplayRenderer.MOTION_FPS,
                "stride": SweepTable3DReplayRenderer.motion_stride(),
                "control_frequency": SweepTable3DBackend.CONTROL_FREQUENCY,
                "robot_joints": list(SweepTable3DBackend.ROBOT_JOINTS),
            }
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="sweeptable3d", method=args.method, run_seed=args.seed, scene=scene
            ),
            state_provider=state,
            task_seed_provider=lambda task: SweepTable3DStepProtocol.task_seed(task=task),
        )
        if env.motion_recording:
            env.attach_motion_sink(sink=writer, stride=scene["motion"]["stride"])
        return writer

    @staticmethod
    def live_state(*, env: SweepTable3DEnvironment) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in values]
            for name, values in env.backend().features().items()
        }

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {RESET_CUBES_NAME: RESET_CUBES_NAME, RESET_WIPER_NAME: RESET_WIPER_NAME}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(round(task.initial_state.get(obj=O.scene, feature_name="seed")))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        del task
        backend = SweepTable3DStepProtocol.env(problem=problem).backend()
        return lambda: backend.task_succeeded() and not backend.task_failed()

    @staticmethod
    def failure_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        del task
        env = SweepTable3DStepProtocol.env(problem=problem)
        return lambda: env.backend().task_failed()

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return SweepTable3DStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return SweepTable3DStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        trace = SkillTraces.last(env=SweepTable3DStepProtocol.env(problem=problem))
        return None if trace is None else trace.model_dump(mode="json")
