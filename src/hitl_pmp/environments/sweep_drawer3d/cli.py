"""Compose independent Sweep practice and evaluation simulators from a frozen manifest."""

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.practice_loop import PracticeResetPolicy


class SweepDrawerCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sweep-manifest",
            type=Path,
            required=True,
            help="Frozen valid seeds, cost calibration and model settings JSON.",
        )
        parser.add_argument("--canonical-seed", type=int, default=0)
        parser.add_argument("--human-reset-practice-cost", type=float, default=1.0)
        parser.add_argument("--no-human-reset", action="store_true")

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        # Global CLI registration must remain usable without simulation extras.
        from .environment import SweepDrawerEnvironment
        from .problem import SweepDrawerProblem
        from .skill_provider import SweepDrawerOracle, SweepDrawerSkillProvider
        from .tasks import SweepDrawerTasks

        if args.practice_reset_policy != PracticeResetPolicy.NEVER:
            raise ValueError("Sweep learning requires --practice-reset-policy never")
        raw = args.sweep_manifest.read_bytes()
        manifest = json.loads(raw)
        if args.canonical_seed not in manifest["valid_practice_seeds"]:
            raise ValueError("Practice seed is absent from the frozen valid-start manifest")
        test_seeds = tuple(manifest["valid_evaluation_seeds"])
        if args.num_test_tasks > len(test_seeds):
            raise ValueError("Not enough prevalidated evaluation seeds in manifest")
        args.defer_rendering = True
        output = None if args.output_dir is None else Path(args.output_dir)
        practice = SweepDrawerEnvironment(
            canonical_seed=args.canonical_seed,
            output_dir=None if output is None else output / "practice",
        )
        evaluation = SweepDrawerEnvironment(
            canonical_seed=test_seeds[0],
            evaluation=True,
            output_dir=None if output is None else output / "evaluation",
        )
        horizon = int(manifest["deployment_horizon"])
        problem = SweepDrawerProblem(
            env=practice,
            tasks=SweepDrawerTasks(env=practice, test_seeds=test_seeds),
            human=UnconditionalHumanOracle,
            deployment_horizon=horizon,
        )
        eval_problem = SweepDrawerProblem(
            env=evaluation,
            tasks=SweepDrawerTasks(env=evaluation, test_seeds=test_seeds),
            human=UnconditionalHumanOracle,
            deployment_horizon=horizon,
        )
        provider = SweepDrawerSkillProvider(
            env=practice,
            human_reset_enabled=not args.no_human_reset,
            human_reset_practice_cost=args.human_reset_practice_cost,
            robot_practice_costs=manifest["robot_practice_costs"],
        )
        context = DomainContext(
            env=practice, skill_provider=provider, oracle=SweepDrawerOracle(env=practice)
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "sweep_preregistration.json").write_bytes(raw)
            (output / "sweep_manifest_sha256.txt").write_text(
                hashlib.sha256(raw).hexdigest() + "\n"
            )
        try:
            MethodRunner.run(
                args=args,
                method=method_factory(context),
                problem=problem,
                evaluation_problem=eval_problem,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=None,
                render_fps=10,
                num_render_checkpoints=0,
            )
        finally:
            practice.close()
            evaluation.close()
