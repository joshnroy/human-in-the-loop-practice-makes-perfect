"""`--env kitchen_recycling3d`: Tossing3D's skills, predicates and samplers on the kitchen.

The provider is `Tossing3DSkillProvider`'s skill table exactly (see `skill_provider.py`);
only the scene behind it differs.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.skill_oracle_policy import ORACLE_THROW_STANDOFF
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .domain import KitchenConfig
from .environment import KitchenRecycling3DEnvironment
from .problem import KitchenRecyclingProblem
from .skill_provider import KitchenRecyclingSkillProvider
from .tasks import KitchenRecyclingTasks

# A human reset, in robot-action equivalents: Tossing3D's.
HUMAN_RESET_PRACTICE_COST = 5.0


class KitchenRecyclingCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--kitchen-config",
            type=Path,
            default=None,
            help="KitchenConfig JSON (layout and the one object). Omitted uses defaults.",
        )
        parser.add_argument(
            "--human-reset",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Offer the human cube/bin resets to practice. Evaluation never does.",
        )
        parser.add_argument(
            "--human-reset-practice-cost",
            type=float,
            default=HUMAN_RESET_PRACTICE_COST,
            help="Cost of either human reset, in robot-action equivalents.",
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
            help="Offset added to --seed to derive the test scene-seed stream.",
        )
        # Tossing3D settings the shared step protocol reads; the kitchen fixes them.
        parser.set_defaults(
            defer_rendering=True,
            scene_bg=True,
            layout=Tossing3DLayout.BARRIER,
            variant="o1",
            canonical_seed=KitchenRecycling3DEnvironment.model_fields["canonical_seed"].default,
            oracle_throw_standoff=ORACLE_THROW_STANDOFF,
            skill_timing=True,
        )

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
    def build_problem(*, args: argparse.Namespace) -> KitchenRecyclingProblem:
        env = KitchenRecycling3DEnvironment(
            kitchen_config=KitchenRecyclingCli.config(args=args),
            scene_bg=args.scene_bg,
            skill_timing=getattr(args, "skill_timing", True),
            # A run that defers rendering draws nothing in-process (its videos are
            # replayed from recorded states, in a scene of their own), so its scenes go
            # without the visual-only furniture meshes. Anything else keeps them.
            visual_furniture=not getattr(args, "defer_rendering", False),
        )
        tasks = KitchenRecyclingTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return KitchenRecyclingProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def practice_context(
        *, args: argparse.Namespace, problem: KitchenRecyclingProblem
    ) -> DomainContext:
        return DomainContext(
            env=problem.env,
            skill_provider=KitchenRecyclingSkillProvider(
                env=problem.env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            ),
            oracle=Tossing3DOracle(env=problem.env, throw_standoff=args.oracle_throw_standoff),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        practice = KitchenRecyclingCli.build_problem(args=args)
        evaluation = KitchenRecyclingCli.build_problem(args=args)
        try:
            MethodRunner.run(
                args=args,
                method=method_factory(
                    KitchenRecyclingCli.practice_context(args=args, problem=practice)
                ),
                problem=practice,
                evaluation_problem=evaluation,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=None,
                render_fps=1,
                num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
            )
        finally:
            practice.env.close()
            evaluation.env.close()
