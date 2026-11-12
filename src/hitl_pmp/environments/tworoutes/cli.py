"""Compose the practice and evaluation scenes for --env tworoutes."""

import argparse
from collections.abc import Callable
from pathlib import Path

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

from .environment import TwoRoutesEnvironment
from .problem import TwoRoutesProblem
from .skill_provider import TwoRoutesOracle, TwoRoutesSkillProvider
from .tasks import TwoRoutesTasks

ATTEMPTS_LOG_FILENAME = "tworoutes_attempts.jsonl"


class TwoRoutesCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        fields = TwoRoutesEnvironment.model_fields
        for flag, field, text in (
            ("--route-a-start", "route_a_start", "Route A's success probability untrained"),
            ("--route-a-rate", "route_a_rate", "Route A's gain per practice attempt"),
            ("--route-b-start", "route_b_start", "Route B's success probability untrained"),
            ("--route-b-rate", "route_b_rate", "Route B's gain per practice attempt"),
            ("--competence-cap", "competence_cap", "The most either route's probability reaches"),
            ("--route-a-reward", "route_a_reward", "What finishing by Route A is worth"),
            ("--route-b-reward", "route_b_reward", "What finishing by Route B is worth"),
        ):
            parser.add_argument(flag, type=float, default=fields[field].default, help=text)
        parser.add_argument("--test-env-seed-offset", type=int, default=10000)
        parser.add_argument("--human-reset", action=argparse.BooleanOptionalAction, default=True)

    @staticmethod
    def build_problem(*, args: argparse.Namespace, practice: bool) -> TwoRoutesProblem:
        """Practice learns and logs its attempts; evaluation does neither, and draws
        its outcomes from a stream of its own."""
        output = getattr(args, "output_dir", None)
        env = TwoRoutesEnvironment(
            route_a_start=args.route_a_start,
            route_a_rate=args.route_a_rate,
            route_b_start=args.route_b_start,
            route_b_rate=args.route_b_rate,
            competence_cap=args.competence_cap,
            route_a_reward=args.route_a_reward,
            route_b_reward=args.route_b_reward,
            seed=args.seed + (0 if practice else args.test_env_seed_offset),
            learns=practice,
            attempts_log=(
                Path(output) / ATTEMPTS_LOG_FILENAME if practice and output is not None else None
            ),
        )
        return TwoRoutesProblem(
            env=env, tasks=TwoRoutesTasks(env=env), human=UnconditionalHumanOracle
        )

    @staticmethod
    def context(*, args: argparse.Namespace, problem: TwoRoutesProblem) -> DomainContext:
        return DomainContext(
            env=problem.env,
            oracle=TwoRoutesOracle(env=problem.env),
            skill_provider=TwoRoutesSkillProvider(
                env=problem.env, offer_human_reset=args.human_reset
            ),
            test_task_goals=(problem.tasks.goal(),),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """Never reached: the global CLI runs a domain that offers a human reset only
        under the step protocol, which composes the run itself (`step_protocol.py`)."""
        del args, method_factory, num_cycles, max_steps_per_interaction
        raise ValueError("--env tworoutes runs only under --practice-step-budget")
