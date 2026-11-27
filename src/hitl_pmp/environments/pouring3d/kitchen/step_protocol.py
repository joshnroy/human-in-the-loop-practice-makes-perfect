"""Use the shared step accounting and learning methods without special cases."""

import argparse
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.step_protocol import FrozenPourStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import KitchenCli
from .config import KitchenGeometry
from .environment import HUMAN_RESET, KitchenEnvironment
from .skill_provider import KitchenSkillProvider
from .tasks import KitchenProblem


class KitchenStepProtocol(FrozenPourStepProtocol):
    simulation_checkpoints = True

    @staticmethod
    def resolve_configuration(*, args: argparse.Namespace) -> None:
        """Freeze the geometry file's contents in a checkpointing run's arguments, so a
        resume is refused when the file changed. A run that neither checkpoints nor
        resumes records exactly the arguments it always did."""
        if not getattr(args, "checkpoint_cycles", False) and getattr(args, "resume", None) is None:
            return
        path = getattr(args, "kitchen_geometry", None)
        geometry = (
            KitchenGeometry()
            if path is None
            else KitchenGeometry.model_validate_json(path.read_text())
        )
        args.kitchen_geometry_resolved = geometry.model_dump_json()

    @staticmethod
    def checkpoint_practice_environment(*, problem: Problem) -> Any:
        assert isinstance(problem.env, KitchenEnvironment)
        return problem.env.simulation_checkpoint()

    @staticmethod
    def restore_checkpoint_environment(*, problem: Problem, checkpoint: Any) -> None:
        assert isinstance(problem.env, KitchenEnvironment)
        problem.env.restore_simulation_checkpoint(checkpoint=checkpoint)

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
    def isolate_evaluation_task(*, problem: Problem) -> None:
        """Pick refusals outlive tasks on one instance (`pick_plannable.py`): one
        recorded in an earlier task survives a reset that puts the dumpling back
        within tolerance, which a task in a fresh process would never have seen."""
        assert isinstance(problem.env, KitchenEnvironment)
        problem.env.forget_pick_refusals()

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
                    "skillet_mu": env.skillet_mu,
                    "scene_bg": env.scene_bg,
                    "deployment": env.deployment,
                    "human_interruption_cost": env.human_interruption_cost,
                    "human_reset_success_probability": env.human_reset_success_probability,
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
