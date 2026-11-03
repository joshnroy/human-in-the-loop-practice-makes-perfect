"""Tossing3D's step protocol over the kitchen: same budget, evaluation and logging."""

import argparse
from pathlib import Path

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.tossing3d.recording import Tossing3DRecording
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .backend import OPPOSITE_SIDE, ROBOT_SIDE
from .cli import KitchenRecyclingCli
from .environment import KitchenRecycling3DEnvironment
from .problem import KitchenRecyclingProblem

# The names a human reset is charged and logged under (the practice cost file's keys).
SAME_SIDE_RESET = "ResetCubeBinSameSide"
OPPOSITE_SIDE_RESET = "ResetCubeBinOppositeSide"


class KitchenRecyclingStepProtocol(Tossing3DStepProtocol):
    @staticmethod
    def kitchen_env(*, problem: Problem) -> KitchenRecycling3DEnvironment:
        assert isinstance(problem.env, KitchenRecycling3DEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> KitchenRecyclingProblem:
        return KitchenRecyclingCli.build_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> KitchenRecyclingProblem:
        return KitchenRecyclingCli.build_problem(args=args)

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        assert isinstance(problem, KitchenRecyclingProblem)
        return KitchenRecyclingCli.practice_context(args=args, problem=problem)

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> Tossing3DSkillProvider:
        env = KitchenRecyclingStepProtocol.kitchen_env(problem=problem)
        return Tossing3DSkillProvider(env=env, offer_human_reset=False)

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = KitchenRecyclingStepProtocol.kitchen_env(problem=problem)
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="kitchen_recycling3d",
                method=args.method,
                run_seed=args.seed,
                scene={
                    "config": env.kitchen_config.model_dump(mode="json"),
                    "scene_bg": env.scene_bg,
                    "seed": args.seed,
                    "test_env_seed_offset": args.test_env_seed_offset,
                },
            ),
            state_provider=lambda: Tossing3DRecording.state(env=env),
            scene_description_provider=lambda: Tossing3DRecording.describe(env=env),
            task_seed_provider=lambda task: Tossing3DRecording.task_seed(env=env, task=task),
        )
        env.attach_state_log_writer(writer=writer)
        return writer

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {ROBOT_SIDE: SAME_SIDE_RESET, OPPOSITE_SIDE: OPPOSITE_SIDE_RESET}
