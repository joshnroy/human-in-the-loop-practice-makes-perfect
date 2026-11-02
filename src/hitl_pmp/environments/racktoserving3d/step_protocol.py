"""Rack-serving integration with the existing control-step practice runner."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import RackServingCli
from .environment import RackServingEnvironment
from .objects import RackServing3DObjects as Objects
from .problem import RackServingProblem
from .semantics import HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY
from .skill_provider import RackServingSkillProvider


class RackServingStepProtocol:
    """A spawn-importable adapter; all dynamics remain in the rack environment."""

    @staticmethod
    def env(*, problem: Problem) -> RackServingEnvironment:
        assert isinstance(problem.env, RackServingEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingCli.build_problem(args=args, mode="training")

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingCli.build_problem(args=args, mode="deployment")

    @staticmethod
    def evaluation_pool() -> None:
        """Use the runner's fresh spawned process for each evaluation."""
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        return RackServingCli.practice_context(
            args=args, env=RackServingStepProtocol.env(problem=problem)
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> RackServingSkillProvider:
        return RackServingSkillProvider(
            env=RackServingStepProtocol.env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        """Record live action-boundary states, not buffered per-control-step states."""
        env = RackServingStepProtocol.env(problem=problem)
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="racktoserving3d",
                method=args.method,
                run_seed=args.seed,
                scene={
                    **env.config.model_dump(mode="json"),
                    "scene_bg": env.scene_bg,
                    "canonical_seed": env.canonical_seed,
                    "skill_step_limit": env.skill_step_limit,
                    "test_env_seed_offset": args.test_env_seed_offset,
                },
            ),
            state_provider=lambda: RackServingStepProtocol.live_state(env=env),
            task_seed_provider=lambda task: RackServingStepProtocol.task_seed(task=task),
        )

    @staticmethod
    def live_state(*, env: RackServingEnvironment) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in values]
            for name, values in env.backend().features().items()
        }

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {
            HUMAN_BATCH_RETURN: HUMAN_BATCH_RETURN,
            HUMAN_TOOL_RECOVERY: HUMAN_TOOL_RECOVERY,
        }

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(round(task.initial_state.get(obj=Objects.scene, feature_name="seed")))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """Use the same live rack verdict as skill-step evaluation, not KINDER's shelf goal."""
        del task
        backend = RackServingStepProtocol.env(problem=problem).backend()
        return lambda: bool(backend.task_succeeded())

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        """Evaluation buffers no ticks; its next task resets the scene."""
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return RackServingStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return RackServingStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        """No rack-specific trace builder exists; shelf skill IDs are not interchangeable."""
        del problem
        return None
