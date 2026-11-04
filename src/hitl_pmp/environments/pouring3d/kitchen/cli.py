"""Opt-in kitchen variant; legacy pouring3d defaults and results are unchanged."""

import argparse
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.practice_loop import PracticeResetPolicy

from .config import KitchenGeometry
from .environment import HUMAN_RESET, KitchenEnvironment
from .skill_provider import KitchenPourOracle, KitchenSkillProvider
from .tasks import KitchenProblem, KitchenTasks

# The open-loop pour's flags: defined by the shared parser, meaningless to the kitchen.
OPEN_LOOP_POUR_FLAGS = ("pour_tilt_bounds", "pour_amplitude_bounds", "pour_duration_bounds")


class KitchenCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        FrozenPourCli.add_arguments(parser=parser)
        parser.add_argument(
            "--kitchen-geometry",
            type=Path,
            help="KitchenGeometry JSON; unspecified real measurements stay null",
        )
        parser.add_argument(
            "--pour-ramp-rate-bounds",
            type=float,
            nargs=2,
            default=list(KitchenSkillProvider.model_fields["ramp_rate_bounds"].default),
            help="The closed-loop Pour's learned tilt ramp rate, deg/s; default %(default)s",
        )
        parser.set_defaults(
            practice_reset_policy=PracticeResetPolicy.NEVER,
            surface_mu=KitchenEnvironment.model_fields["surface_mu"].default,
            **dict.fromkeys(OPEN_LOOP_POUR_FLAGS),
        )

    @staticmethod
    def human_interruption_cost(*, args: argparse.Namespace) -> float:
        """The environment's own interruption ledger records the price the learner is
        charged for a human reset, read from the same costs file, so the two agree."""
        costs = PracticeCosts.load(path=args.practice_cost_config)
        if HUMAN_RESET not in costs.human_skills:
            raise ValueError(f"{args.practice_cost_config} does not price {HUMAN_RESET!r}")
        charge = costs.human_skills[HUMAN_RESET].cost
        if charge.function is not None:
            raise ValueError(f"{HUMAN_RESET!r} needs a constant cost for the kitchen ledger")
        return costs.human_weight * charge.value

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
            human_interruption_cost=KitchenCli.human_interruption_cost(args=args),
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
        stale = [
            "--" + name.replace("_", "-")
            for name in OPEN_LOOP_POUR_FLAGS
            if getattr(args, name, None) is not None
        ]
        if stale:
            raise ValueError(
                "The kitchen Pour is closed-loop and learns only its tilt ramp rate; "
                f"{', '.join(stale)} do not apply: use --pour-ramp-rate-bounds"
            )
        return DomainContext(
            env=problem.env,
            oracle=KitchenPourOracle(env=problem.env),
            test_task_goals=problem.tasks.test_task_goals(),
            skill_provider=KitchenSkillProvider(
                env=problem.env,
                offer_human_reset=args.human_reset,
                ramp_rate_bounds=tuple(args.pour_ramp_rate_bounds),
            ),
        )

    @staticmethod
    def run_method(**kwargs: Any) -> None:
        del kwargs
        raise ValueError(
            "Kitchen practice requires --practice-step-budget: "
            "persistent world and explicit costed resets"
        )
