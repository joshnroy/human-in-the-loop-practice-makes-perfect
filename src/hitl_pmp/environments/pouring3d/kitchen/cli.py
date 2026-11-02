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
            human_reset_practice_cost=5, practice_reset_policy=PracticeResetPolicy.NEVER
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
        return DomainContext(
            env=problem.env,
            oracle=FrozenPourOracle(env=problem.env),
            skill_provider=KitchenSkillProvider(
                env=problem.env,
                offer_human_reset=args.human_reset,
                human_reset_practice_cost=args.human_reset_practice_cost,
                tilt_bounds=tuple(args.pour_tilt_bounds),
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
