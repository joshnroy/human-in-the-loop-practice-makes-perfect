"""Compose independent Sweep practice and evaluation simulators from a frozen manifest."""

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.practice_loop import PracticeResetPolicy


class SweepSimpleCli:
    @staticmethod
    def validate_manifest(*, manifest: dict[str, Any]) -> None:
        if manifest.get("status") != "FROZEN":
            raise ValueError("Simple launch requires an explicitly FROZEN readiness manifest")
        expected = {
            "environment": "sweep_simple3d",
            "num_cycles": 50,
            "max_steps_per_interaction": 20,
            "num_test_tasks": 10,
            "deployment_horizon": 10,
            "practice_reset_policy": "never",
        }
        for name, value in expected.items():
            if manifest.get(name) != value or isinstance(manifest.get(name), bool):
                raise ValueError(f"Unapproved Simple protocol {name}: expected {value!r}")
        model = manifest.get("pomdp", {})
        approved = {
            "goal_pursuit_horizon": 0,
            "pomdp_solver": "determinized_astar",
            "pomdp_max_search_iterations": 1000,
            "pomdp_search_depth": 20,
            "pomdp_inference_engine": "grid",
            "pomdp_grid_competence_bins": 25,
            "pomdp_grid_learning_rate_bins": 16,
            "pomdp_num_particles": 1024,
            "pomdp_linear_cost_lambda": 3e-6,
        }
        for name, value in approved.items():
            # Existing manifests serialize some numeric model settings as strings.
            actual = model.get(name, 0 if name == "goal_pursuit_horizon" else None)
            if isinstance(value, (int, float)):
                try:
                    matches = not isinstance(actual, bool) and float(actual) == value
                except (ValueError, TypeError):
                    matches = False
            else:
                matches = actual == value
            if not matches:
                raise ValueError(f"Unapproved Simple model {name}: expected {value!r}")
        evaluation = manifest.get("valid_evaluation_seeds", [])
        if len(evaluation) != 10 or len(set(evaluation)) != 10:
            raise ValueError("Simple requires exactly ten distinct held-out evaluation seeds")
        practice = manifest.get("valid_practice_seeds", [])
        if not practice or set(practice) & set(evaluation):
            raise ValueError("Simple practice and held-out evaluation seeds must be separate")

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
        raw = args.sweep_manifest.read_bytes()
        manifest = json.loads(raw)
        SweepSimpleCli.validate_manifest(manifest=manifest)
        if args.goal_pursuit_horizon not in (0, "0"):
            raise ValueError("Simple requires resolved --goal-pursuit-horizon 0 for both methods")
        if max_steps_per_interaction != 20 or (num_cycles, args.num_test_tasks) not in {
            (50, 10),
            (1, 1),  # Explicit launcher smoke is one cycle / one task.
        }:
            raise ValueError("Simple requires 50x20 practice / ten tasks, or one-cycle smoke")
        # Global CLI registration must remain usable without simulation extras.
        from .environment import SweepSimpleEnvironment
        from .problem import SweepSimpleProblem
        from .skill_provider import SweepSimpleOracle, SweepSimpleSkillProvider
        from .tasks import SweepSimpleTasks

        if args.practice_reset_policy != PracticeResetPolicy.NEVER:
            raise ValueError("Sweep learning requires --practice-reset-policy never")
        if args.canonical_seed not in manifest["valid_practice_seeds"]:
            raise ValueError("Practice seed is absent from the frozen valid-start manifest")
        test_seeds = tuple(manifest["valid_evaluation_seeds"])
        if args.num_test_tasks > len(test_seeds):
            raise ValueError("Not enough prevalidated evaluation seeds in manifest")
        args.defer_rendering = True
        output = None if args.output_dir is None else Path(args.output_dir)
        practice = SweepSimpleEnvironment(
            canonical_seed=args.canonical_seed,
            output_dir=None if output is None else output / "practice",
        )
        evaluation = SweepSimpleEnvironment(
            canonical_seed=test_seeds[0],
            evaluation=True,
            output_dir=None if output is None else output / "evaluation",
        )
        horizon = int(manifest["deployment_horizon"])
        problem = SweepSimpleProblem(
            env=practice,
            tasks=SweepSimpleTasks(env=practice, test_seeds=test_seeds),
            human=UnconditionalHumanOracle,
            deployment_horizon=horizon,
        )
        eval_problem = SweepSimpleProblem(
            env=evaluation,
            tasks=SweepSimpleTasks(env=evaluation, test_seeds=test_seeds),
            human=UnconditionalHumanOracle,
            deployment_horizon=horizon,
        )
        provider = SweepSimpleSkillProvider(
            env=practice,
            human_reset_enabled=not args.no_human_reset,
            human_reset_practice_cost=args.human_reset_practice_cost,
            robot_practice_costs=manifest["robot_practice_costs"],
            **(
                {"stock_parameter_bounds": manifest["stock_parameter_bounds"]}
                if "stock_parameter_bounds" in manifest
                else {}
            ),
        )
        provider.validate_trainable_support()
        context = DomainContext(
            env=practice, skill_provider=provider, oracle=SweepSimpleOracle(env=practice)
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
