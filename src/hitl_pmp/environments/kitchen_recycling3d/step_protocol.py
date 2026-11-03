"""Kitchen recycling adapter for the shared motor-step budget and evaluation protocol."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import KitchenRecyclingCli
from .domain import KitchenConfig
from .environment import KitchenRecyclingEnvironment
from .problem import KitchenRecyclingProblem
from .skill_provider import KitchenRecyclingSkillProvider


class KitchenRecyclingStepProtocol(SimulatedStepProtocolHooks):
    """Measure motor steps without sharing a simulator across training and deployment."""

    recovery_objects: ClassVar[tuple[str, ...]] = tuple(
        spec.name for spec in KitchenConfig().objects
    )

    @staticmethod
    def env(*, problem: Problem) -> KitchenRecyclingEnvironment:
        assert isinstance(problem.env, KitchenRecyclingEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> KitchenRecyclingProblem:
        problem = KitchenRecyclingCli.build_problem(args=args, phase="training")
        KitchenRecyclingStepProtocol.recovery_objects = tuple(
            spec.name for spec in problem.env.config.objects
        )
        return problem

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> KitchenRecyclingProblem:
        return KitchenRecyclingCli.build_problem(args=args, phase="deployment")

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        assert isinstance(problem, KitchenRecyclingProblem)
        return KitchenRecyclingCli.practice_context(args=args, problem=problem)

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> KitchenRecyclingSkillProvider:
        env = KitchenRecyclingStepProtocol.env(problem=problem)
        if env.phase != "deployment":
            raise ValueError("Evaluation requires a separate deployment environment")
        return KitchenRecyclingSkillProvider(env=env, offer_human_reset=False)

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = KitchenRecyclingStepProtocol.env(problem=problem)
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="kitchen_recycling3d",
                method=args.method,
                run_seed=args.seed,
                scene={
                    "config": env.config.model_dump(mode="json"),
                    "phase": env.phase,
                    "scene_bg": env.scene_bg,
                    "perception": "simulator_ground_truth",
                    "video": False,
                },
            ),
            state_provider=lambda: KitchenRecyclingStepProtocol.live_state(env=env),
            scene_description_provider=lambda: {
                "feature_names": {
                    obj.name: list(obj.type.feature_names) for obj in env.get_current_state().data
                }
            },
            task_seed_provider=lambda task: KitchenRecyclingStepProtocol.task_seed(task=task),
        )

    @staticmethod
    def live_state(*, env: KitchenRecyclingEnvironment) -> dict[str, list[float]]:
        return {
            obj.name: [float(value) for value in values]
            for obj, values in env.get_current_state().data.items()
        }

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {
            **{
                f"recover:{name}": f"recover:{name}"
                for name in KitchenRecyclingStepProtocol.recovery_objects
            },
            "relocate_bin_near": "relocate_bin_near",
            "restore_bin_target": "restore_bin_target",
        }

    @staticmethod
    def task_seed(*, task: Task) -> int:
        seed = getattr(task.initial_state, "seed", None)
        if seed is not None:
            return int(seed)
        scenes = [obj for obj in task.initial_state.data if "seed" in obj.type.feature_names]
        if len(scenes) != 1:
            raise ValueError("Kitchen task state must identify exactly one scene seed")
        return int(task.initial_state.get(obj=scenes[0], feature_name="seed"))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        del task
        return KitchenRecyclingStepProtocol.env(problem=problem).is_solved

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        KitchenRecyclingStepProtocol.adopt_interrupted_skill(
            problem=problem, action=action, reason="evaluation_control_steps"
        )

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return KitchenRecyclingStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return KitchenRecyclingStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        del problem
        return None
