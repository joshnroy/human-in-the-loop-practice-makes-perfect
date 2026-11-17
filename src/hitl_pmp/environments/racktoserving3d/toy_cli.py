"""Compose `--env racktoserving3d_toy`: the 3D Serving scene behind the chained toy's
interface (`toy_interface.py`). Every Serving flag applies; three are the toy's own."""

import argparse
from collections.abc import Callable
from typing import Literal

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom

from .cli import RackServingCli
from .problem import RackServingProblem
from .tasks import RackServingTasks
from .toy_environment import RackServingToyEnvironment
from .toy_interface import ServingToyPredicates
from .toy_skill_provider import RackServingToyOracle, RackServingToySkillProvider


class RackServingToyTasks(RackServingTasks):
    """The toy's one goal, `Served`: either route reaches it, and which one did is in
    the state (`ServedThree`/`ServedFive`)."""

    env: RackServingToyEnvironment

    def goal(self) -> Goal:
        return Goal(
            atoms=frozenset({
                GroundAtom(
                    predicate=ServingToyPredicates.get(name="Served"),
                    objects=(RackServingToyEnvironment.robot,),
                )
            })
        )


class RackServingToyCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        RackServingCli.add_arguments(parser=parser)
        fields = RackServingToyEnvironment.model_fields
        for flag, field, text in (
            (
                "--robot-reset-price",
                "robot_reset_price",
                "What one robot_reset costs, under a cost file that uses "
                "hitl_pmp.environments.tworoutes.costs:robot_step",
            ),
            ("--sweep3-reward", "sweep3_reward", "What serving three by sweep3 is worth"),
            ("--sweep5-reward", "sweep5_reward", "What serving five by sweep5 is worth"),
        ):
            parser.add_argument(flag, type=float, default=fields[field].default, help=text)

    @staticmethod
    def build_problem(
        *, args: argparse.Namespace, mode: Literal["training", "deployment"]
    ) -> RackServingProblem:
        """The Serving CLI's own problem, its environment and tasks swapped for the
        toy's: no simulator is built until the first reset."""
        plain = RackServingCli.build_problem(args=args, mode=mode)
        env = RackServingToyEnvironment(
            config=plain.env.config,
            canonical_seed=plain.env.canonical_seed,
            scene_bg=plain.env.scene_bg,
            skill_step_limit=plain.env.skill_step_limit,
            motion_recording=plain.env.motion_recording,
            robot_reset_price=args.robot_reset_price,
            sweep3_reward=args.sweep3_reward,
            sweep5_reward=args.sweep5_reward,
        )
        tasks = RackServingToyTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return RackServingProblem(env=env, tasks=tasks, human=plain.human)

    @staticmethod
    def practice_context(
        *, args: argparse.Namespace, env: RackServingToyEnvironment
    ) -> DomainContext:
        return DomainContext(
            env=env,
            skill_provider=RackServingToySkillProvider(env=env, offer_human_reset=args.human_reset),
            oracle=RackServingToyOracle(env=env),
            test_task_goals=(RackServingToyTasks(env=env).goal(),),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """Never reached: the toy's interface runs only under the step protocol, as
        the toy itself does (`TwoRoutesCli.run_method`)."""
        del args, method_factory, num_cycles, max_steps_per_interaction
        raise ValueError("--env racktoserving3d_toy runs only under --practice-step-budget")
