"""Use the shared step accounting and learning methods without special cases."""

import argparse
from pathlib import Path

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.step_protocol import FrozenPourStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import KitchenCli
from .environment import HUMAN_RESET, KitchenEnvironment
from .skill_provider import KitchenSkillProvider
from .tasks import KitchenProblem


class KitchenStepProtocol(FrozenPourStepProtocol):
    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> KitchenProblem:
        return KitchenCli.build_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> KitchenProblem:
        return KitchenCli.build_problem(args=args, deployment=True)

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        assert isinstance(problem, KitchenProblem)
        return KitchenCli.context(args=args, problem=problem)

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> KitchenSkillProvider:
        assert isinstance(problem.env, KitchenEnvironment)
        return KitchenSkillProvider(env=problem.env, offer_human_reset=False)

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {HUMAN_RESET: HUMAN_RESET}

    @staticmethod
    def live_state(*, env: FrozenPourEnvironment) -> dict[str, list[float]]:
        assert isinstance(env, KitchenEnvironment)
        return {
            **FrozenPourStepProtocol.live_state(env=env),
            "hob": [float(env.controller().scene.cooking_observed())],
        }

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        assert isinstance(problem.env, KitchenEnvironment)
        env = problem.env
        motion = args.record_motion == "all" or (
            args.record_motion == "evaluation" and "evaluations" in output_path.parts
        )
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="pouring3d_kitchen",
                method=args.method,
                run_seed=args.seed,
                scene={
                    "geometry": env.geometry.model_dump(),
                    "requested_count": env.requested_count,
                    "surface_mu": env.surface_mu,
                    "scene_bg": env.scene_bg,
                    "deployment": env.deployment,
                    "human_interruption_cost": env.human_interruption_cost,
                    "motion": motion,
                    "fps": 10,
                    "kitchen_model_version": 2,
                },
            ),
            state_provider=lambda: KitchenStepProtocol.live_state(env=env),
            task_seed_provider=lambda task: KitchenStepProtocol.task_seed(task=task),
        )
        env.set_skill_sink(sink=writer.record_skill)
        if motion:
            env.set_motion_sink(
                sink=lambda **_: writer.record_tick(state=KitchenStepProtocol.live_state(env=env))
            )
        return writer
