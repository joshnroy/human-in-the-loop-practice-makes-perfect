"""Plugs SweepOffShelf3D into the global runner as `--env sweepoffshelf3d`.

    python -m hitl_pmp.cli --env sweepoffshelf3d --method ees --left-behind 0 ...

Like Tossing3D it builds two independent Environment/Tasks/Problem triples, one for
practice and one for evaluation, so an evaluation episode's rebuild of the scene never
hands the practice arm a free reset. The practice environment's human leaves
`--left-behind` (k) cubes where they are on every `reset_cubes`.
"""

import argparse
from collections.abc import Callable

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .environment import SweepOffShelf3DEnvironment
from .objects import NUM_CUBES
from .problem import SweepOffShelf3DProblem
from .skill_provider import SweepOffShelf3DOracle, SweepOffShelf3DSkillProvider
from .tasks import SweepOffShelf3DTasks


class SweepOffShelf3DCli:
    """A static-method container, never instantiated."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        env_fields = SweepOffShelf3DEnvironment.model_fields
        parser.add_argument(
            "--left-behind",
            type=int,
            choices=range(NUM_CUBES + 1),
            default=env_fields["left_behind"].default,
            help="k: how many cubes the human's reset_cubes leaves where they are "
            "(k=0 returns all five to the shelf, k=5 returns none).",
        )
        parser.add_argument(
            "--no-scene-bg",
            dest="scene_bg",
            action="store_false",
            help="Use the plain scene instead of the MimicLabs lab (physics unchanged).",
        )
        parser.add_argument(
            "--canonical-seed", type=int, default=env_fields["canonical_seed"].default
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=SweepOffShelf3DTasks.model_fields["test_env_seed_offset"].default,
        )
        parser.add_argument(
            "--skill-step-limit",
            type=int,
            default=env_fields["skill_step_limit"].default,
            help="Control steps one skill may take before it is cut off as a failure.",
        )
        parser.add_argument(
            "--human-reset-practice-cost",
            type=float,
            default=SweepOffShelf3DSkillProvider.model_fields["human_reset_practice_cost"].default,
            help="Cost of each human skill (reset_cubes, reset_wiper) in robot-skill units.",
        )
        parser.add_argument(
            "--human-reset",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Offer the human skills to practice. Evaluation never offers them.",
        )
        parser.set_defaults(scene_bg=True, defer_rendering=True)

    @staticmethod
    def build_problem(*, args: argparse.Namespace, human_seed: int) -> SweepOffShelf3DProblem:
        env = SweepOffShelf3DEnvironment(
            scene_bg=args.scene_bg,
            canonical_seed=args.canonical_seed,
            left_behind=args.left_behind,
            human_seed=human_seed,
            skill_step_limit=args.skill_step_limit,
        )
        tasks = SweepOffShelf3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return SweepOffShelf3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        practice = SweepOffShelf3DCli.build_problem(args=args, human_seed=args.seed)
        evaluation = SweepOffShelf3DCli.build_problem(args=args, human_seed=args.seed + 1)
        context = DomainContext(
            env=practice.env,
            skill_provider=SweepOffShelf3DSkillProvider(
                env=practice.env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            ),
            oracle=SweepOffShelf3DOracle(env=practice.env),
        )
        try:
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
            practice.env.close()
            evaluation.env.close()
