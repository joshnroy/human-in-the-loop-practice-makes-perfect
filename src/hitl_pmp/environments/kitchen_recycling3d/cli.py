"""Simulation-only kitchen recycling composition root for the shared method runner."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .domain import KitchenConfig
from .environment import KitchenRecyclingEnvironment
from .problem import KitchenRecyclingProblem
from .skill_provider import KitchenRecyclingOracle, KitchenRecyclingSkillProvider
from .tasks import KitchenRecyclingTasks


class KitchenRecyclingCli:
    """Run existing methods against independent training and deployment simulators."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--kitchen-config",
            type=Path,
            default=None,
            help="KitchenConfig JSON, including the calibrated preview configuration. "
            "Omitted uses simulation defaults. Perception is simulator ground truth only.",
        )
        parser.add_argument(
            "--human-reset",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Offer recovery and bin relocation during training only. "
            "Deployment prohibits all human actions.",
        )
        parser.add_argument(
            "--no-scene-bg",
            dest="scene_bg",
            action="store_false",
            help="Use the plain simulation scene without decorative background assets.",
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=KitchenRecyclingTasks.model_fields["test_env_seed_offset"].default,
            help="Offset for the independent, fixed deployment task stream.",
        )
        parser.set_defaults(defer_rendering=True, scene_bg=True)

    @staticmethod
    def config(*, args: argparse.Namespace) -> KitchenConfig:
        """Freeze resolved JSON in the run arguments passed to evaluation workers."""
        resolved = getattr(args, "kitchen_config_resolved", None)
        if resolved is not None:
            return KitchenConfig.model_validate_json(resolved)
        path = getattr(args, "kitchen_config", None)
        config = (
            KitchenConfig()
            if path is None
            else KitchenConfig.model_validate_json(Path(path).read_text())
        )
        args.kitchen_config_resolved = config.model_dump_json()
        return config

    @staticmethod
    def build_problem(
        *, args: argparse.Namespace, phase: Literal["training", "deployment"]
    ) -> KitchenRecyclingProblem:
        if getattr(args, "record_full_loop", None) is not None:
            raise ValueError(
                "Kitchen recycling does not provide video rendering; use the step protocol "
                "for structured action and result recordings."
            )
        env = KitchenRecyclingEnvironment(
            config=KitchenRecyclingCli.config(args=args),
            phase=phase,
            seed=args.seed,
            scene_bg=args.scene_bg,
        )
        try:
            tasks = KitchenRecyclingTasks(
                env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
            )
            return KitchenRecyclingProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)
        except BaseException:
            env.close()
            raise

    @staticmethod
    def practice_context(
        *, args: argparse.Namespace, problem: KitchenRecyclingProblem
    ) -> DomainContext:
        return DomainContext(
            env=problem.env,
            skill_provider=KitchenRecyclingSkillProvider(
                env=problem.env, offer_human_reset=args.human_reset
            ),
            oracle=KitchenRecyclingOracle(env=problem.env),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        practice = KitchenRecyclingCli.build_problem(args=args, phase="training")
        try:
            evaluation = KitchenRecyclingCli.build_problem(args=args, phase="deployment")
            try:
                context = KitchenRecyclingCli.practice_context(args=args, problem=practice)
                MethodRunner.run(
                    args=args,
                    method=method_factory(context),
                    problem=practice,
                    evaluation_problem=evaluation,
                    num_cycles=num_cycles,
                    max_steps_per_interaction=max_steps_per_interaction,
                    renderer=None,
                    render_fps=1,
                    num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
                )
            finally:
                evaluation.env.close()
        finally:
            practice.env.close()
