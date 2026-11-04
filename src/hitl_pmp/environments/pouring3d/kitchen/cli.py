"""Opt-in kitchen variant; legacy pouring3d defaults and results are unchanged."""

import argparse
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourOracle
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.practice_loop import PracticeResetPolicy

from .config import KitchenGeometry
from .environment import KitchenEnvironment
from .skill_provider import KitchenSkillProvider
from .tasks import KitchenProblem, KitchenTasks


class KitchenCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        FrozenPourCli.add_arguments(parser=parser)
        parser.add_argument(
            "--kitchen-geometry",
            type=Path,
            help="KitchenGeometry JSON; unspecified real measurements stay null",
        )
        parser.set_defaults(
            human_reset_practice_cost=5,
            practice_reset_policy=PracticeResetPolicy.NEVER,
            surface_mu=KitchenEnvironment.model_fields["surface_mu"].default,
            # Pour tilt is fixed (KitchenSkillProvider.POUR_TILT_DEGREES), not a range.
            pour_tilt_bounds=None,
            pour_amplitude_bounds=KitchenSkillProvider.model_fields["amplitude_bounds"].default,
            pour_duration_bounds=KitchenSkillProvider.model_fields["duration_bounds"].default,
        )

    @staticmethod
    def build_problem(*, args: argparse.Namespace, deployment: bool = False) -> KitchenProblem:
        # Reuse argument validation only; this constructs no native scene.
        base = FrozenPourCli.build_problem(args=args)
        path = getattr(args, "kitchen_geometry", None)
        geometry = (
            KitchenGeometry.model_validate_json(path.read_text()) if path else KitchenGeometry()
        )
        env = KitchenEnvironment(
            **base.env.model_dump(exclude={"current_state", "last_info"}),
            geometry=geometry,
            deployment=deployment,
            human_interruption_cost=args.human_reset_practice_cost,
        )
        tasks = KitchenTasks(
            env=env,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
            requested_counts=base.tasks.requested_counts,
        )
        return KitchenProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def context(*, args: argparse.Namespace, problem: KitchenProblem) -> DomainContext:
        if args.pour_tilt_bounds is not None:
            raise ValueError(
                f"Kitchen Pour tilt is fixed at {KitchenSkillProvider.POUR_TILT_DEGREES} deg; "
                "omit --pour-tilt-bounds"
            )
        return DomainContext(
            env=problem.env,
            oracle=FrozenPourOracle(env=problem.env),
            test_task_goals=problem.tasks.test_task_goals(),
            skill_provider=KitchenSkillProvider(
                env=problem.env,
                offer_human_reset=args.human_reset,
                human_reset_practice_cost=args.human_reset_practice_cost,
                amplitude_bounds=tuple(args.pour_amplitude_bounds),
                duration_bounds=tuple(args.pour_duration_bounds),
            ),
        )

    @staticmethod
    def run_method(**kwargs: Any) -> None:
        del kwargs
        raise ValueError(
            "Kitchen practice requires --practice-step-budget: "
            "persistent world and explicit costed resets"
        )
