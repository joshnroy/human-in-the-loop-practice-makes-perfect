"""Compose independent practice/evaluation scenes for --env pouring3d."""

import argparse
from collections.abc import Callable

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .environment import FrozenPourEnvironment
from .problem import FrozenPourProblem
from .skill_provider import FrozenPourOracle, FrozenPourSkillProvider
from .tasks import FrozenPourTasks


class FrozenPourCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--payload-count", type=int, choices=(3, 5), default=3)
        parser.add_argument("--requested-count", type=int, default=2)
        parser.add_argument(
            "--surface-mu",
            type=float,
            default=0.1,
            help="Explicit uncalibrated contact assumption; default 0.1",
        )
        parser.add_argument("--canonical-seed", type=int, default=0)
        parser.add_argument("--test-env-seed-offset", type=int, default=10000)
        parser.add_argument("--skill-step-limit", type=int, default=2500)
        parser.add_argument("--human-reset-practice-cost", type=float, default=1)
        parser.add_argument("--human-reset", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--no-scene-bg", dest="scene_bg", action="store_false")
        parser.add_argument("--record-motion", choices=("off", "evaluation", "all"), default="off")
        parser.add_argument("--pour-tilt-bounds", type=float, nargs=2, default=(87, 95))
        parser.add_argument("--pour-amplitude-bounds", type=float, nargs=2, default=(0, 0.025))
        parser.add_argument("--pour-duration-bounds", type=float, nargs=2, default=(0, 30))
        parser.set_defaults(scene_bg=True, defer_rendering=True)

    @staticmethod
    def build_problem(*, args: argparse.Namespace) -> FrozenPourProblem:
        env = FrozenPourEnvironment(
            payload_count=args.payload_count,
            requested_count=args.requested_count,
            surface_mu=args.surface_mu,
            canonical_seed=args.canonical_seed,
            scene_bg=args.scene_bg,
            skill_step_limit=args.skill_step_limit,
        )
        tasks = FrozenPourTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return FrozenPourProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def context(*, args: argparse.Namespace, problem: FrozenPourProblem) -> DomainContext:
        return DomainContext(
            env=problem.env,
            oracle=FrozenPourOracle(env=problem.env),
            skill_provider=FrozenPourSkillProvider(
                env=problem.env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
                tilt_bounds=tuple(args.pour_tilt_bounds),
                amplitude_bounds=tuple(args.pour_amplitude_bounds),
                duration_bounds=tuple(args.pour_duration_bounds),
            ),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        if args.record_motion != "off":
            raise ValueError("--record-motion requires --practice-step-budget")
        practice = FrozenPourCli.build_problem(args=args)
        evaluation = FrozenPourCli.build_problem(args=args)
        try:
            MethodRunner.run(
                args=args,
                method=method_factory(FrozenPourCli.context(args=args, problem=practice)),
                problem=practice,
                evaluation_problem=evaluation,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=None,
                render_fps=10,
                num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
            )
        finally:
            practice.env.close()
            evaluation.env.close()
