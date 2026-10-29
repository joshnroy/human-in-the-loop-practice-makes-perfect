"""Composition root for `--env sweeptable3d`, with independent evaluation scenes."""

import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .environment import SweepTable3DEnvironment
from .problem import SweepTable3DProblem
from .skill_provider import SweepTable3DOracle, SweepTable3DSkillProvider
from .tasks import SweepTable3DTasks


class SweepTable3DCli:
    """Build table runs without importing or constructing a simulator at CLI parse time."""

    MOTION_SCOPES: ClassVar[dict[str, tuple[bool, bool]]] = {
        "off": (False, False),
        "evaluation": (False, True),
        "all": (True, True),
    }

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        env_fields = SweepTable3DEnvironment.model_fields
        parser.add_argument(
            "--reset-cubes-success-probability",
            type=lambda value: SweepTable3DCli.probability(value=value),
            default=env_fields["reset_cubes_success_probability"].default,
            help="Probability the human returns all five balls to the table; "
            "otherwise nothing moves.",
        )
        parser.add_argument(
            "--no-scene-bg",
            dest="scene_bg",
            action="store_false",
            help="Use the plain scene instead of the MimicLabs lab.",
        )
        parser.add_argument(
            "--canonical-seed", type=int, default=env_fields["canonical_seed"].default
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=SweepTable3DTasks.model_fields["test_env_seed_offset"].default,
        )
        parser.add_argument(
            "--fixed-bin-offset",
            type=lambda value: SweepTable3DCli.bin_offset(value=value),
            default=env_fields["fixed_bin_offset"].default,
            help="Fix the bin's y offset from table centre in [-0.4, 0.4] metres; "
            "default -0.18, fixed across scene seeds.",
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
            default=SweepTable3DSkillProvider.model_fields["human_reset_practice_cost"].default,
            help="Cost of each human skill in robot-skill units.",
        )
        parser.add_argument(
            "--human-reset",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Offer human skills during practice. Evaluation never offers them.",
        )
        parser.add_argument(
            "--record-motion",
            choices=list(SweepTable3DCli.MOTION_SCOPES),
            default="off",
            help="Step protocol only: record full motion for evaluation or all phases.",
        )
        parser.set_defaults(scene_bg=True, defer_rendering=True)

    @staticmethod
    def probability(*, value: str) -> float:
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise argparse.ArgumentTypeError("a probability must lie in [0, 1]")
        return probability

    @staticmethod
    def bin_offset(*, value: str) -> float:
        offset = float(value)
        if not -0.4 <= offset <= 0.4:
            raise argparse.ArgumentTypeError("a bin offset must lie in [-0.4, 0.4] metres")
        return offset

    @staticmethod
    def build_problem(
        *, args: argparse.Namespace, human_seed: int, motion_recording: bool = False
    ) -> SweepTable3DProblem:
        env = SweepTable3DEnvironment(
            scene_bg=args.scene_bg,
            canonical_seed=args.canonical_seed,
            reset_cubes_success_probability=args.reset_cubes_success_probability,
            human_seed=human_seed,
            skill_step_limit=args.skill_step_limit,
            motion_recording=motion_recording,
            fixed_bin_offset=getattr(args, "fixed_bin_offset", -0.18),
        )
        tasks = SweepTable3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return SweepTable3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        if getattr(args, "record_motion", "off") != "off":
            raise ValueError("--record-motion needs the step protocol (--practice-step-budget)")
        practice = SweepTable3DCli.build_problem(args=args, human_seed=args.seed)
        evaluation = SweepTable3DCli.build_problem(args=args, human_seed=args.seed + 1)
        try:
            context = DomainContext(
                env=practice.env,
                skill_provider=SweepTable3DSkillProvider(
                    env=practice.env,
                    human_reset_practice_cost=args.human_reset_practice_cost,
                    offer_human_reset=args.human_reset,
                ),
                oracle=SweepTable3DOracle(env=practice.env),
            )
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
