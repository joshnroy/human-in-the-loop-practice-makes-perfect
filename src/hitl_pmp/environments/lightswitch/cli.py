import argparse
from typing import ClassVar

from hitl_pmp.core.method.types import Policy

from .environment import LightSwitchEnvironment
from .oracle_policy import ORACLE_POLICY
from .problem import LightSwitchProblem
from .tasks import LightSwitchTasks


class LightSwitchCli:
    """Plugs Light Switch into the generic runner (see hitl_pmp/cli.py): exposes its
    configurable ClassVars as argparse flags and runs a chosen policy over sampled
    test tasks. A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    POLICIES: ClassVar[dict[str, Policy]] = {"oracle": ORACLE_POLICY}

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks are global flags added by hitl_pmp/cli.py,
        not here -- everything below is specific to this domain."""
        parser.add_argument(
            "--policy",
            choices=sorted(LightSwitchCli.POLICIES),
            default="oracle",
            help="Which policy to run.",
        )
        parser.add_argument(
            "--grid-size",
            type=int,
            default=LightSwitchEnvironment.grid_size,
            help="Number of cells in the Light Switch grid.",
        )
        parser.add_argument(
            "--light-on-tolerance",
            type=float,
            default=LightSwitchEnvironment.light_on_tolerance,
            help="Max |level - target| for the LightOn predicate to hold.",
        )
        parser.add_argument(
            "--same-position-tolerance",
            type=float,
            default=LightSwitchEnvironment.same_position_tolerance,
            help='Max |robot.x - light.x| to count as "at the light".',
        )
        parser.add_argument(
            "--canonical-light-target",
            type=float,
            default=LightSwitchEnvironment.canonical_light_target,
            help="light_target used by Environment.hard_reset (not task sampling).",
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=LightSwitchTasks.test_env_seed_offset,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        parser.add_argument(
            "--target-low",
            type=float,
            default=LightSwitchTasks.target_low,
            help="Lower bound of the light's sampled target (Uniform[target_low, target_high)).",
        )
        parser.add_argument(
            "--target-high",
            type=float,
            default=LightSwitchTasks.target_high,
            help="Upper bound of the light's sampled target.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace) -> tuple[int, int]:
        """Applies args as config, runs args.policy over args.num_test_tasks sampled
        test tasks, prints progress, and returns (num_solved, num_test_tasks)."""
        LightSwitchCli._apply_config(args=args)
        policy = LightSwitchCli.POLICIES[args.policy]
        # No hard_reset() here: run_task_episode below unconditionally overwrites
        # current_state from each task's own initial_state before doing anything
        # else, so a reset beforehand would never be observed.

        num_solved = 0
        for i in range(args.num_test_tasks):
            task = LightSwitchTasks.sample_test_task()
            solved = LightSwitchProblem.run_task_episode(task=task, policy=policy)
            num_solved += int(solved)
            print(f"task {i + 1}/{args.num_test_tasks}: {'solved' if solved else 'FAILED'}")

        print(
            f"success rate: {num_solved}/{args.num_test_tasks} "
            f"({num_solved / args.num_test_tasks:.0%})"
        )
        return num_solved, args.num_test_tasks

    @staticmethod
    def _apply_config(*, args: argparse.Namespace) -> None:
        LightSwitchEnvironment.grid_size = args.grid_size
        LightSwitchEnvironment.light_on_tolerance = args.light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = args.same_position_tolerance
        LightSwitchEnvironment.canonical_light_target = args.canonical_light_target
        LightSwitchTasks.test_env_seed_offset = args.test_env_seed_offset
        LightSwitchTasks.target_low = args.target_low
        LightSwitchTasks.target_high = args.target_high
        # Must run last: rederives train_rng/test_rng from seed + (possibly
        # just-updated) test_env_seed_offset.
        LightSwitchTasks.set_seed(seed=args.seed)
