"""Runs a Light Switch policy over sampled test tasks and reports a success rate --
the "Step 3: implement an oracle" step from the Problem Setting recipe, establishing
an upper bound on the metrics before any learning method exists.

Every LightSwitchEnvironment/LightSwitchTasks config value is exposed as a flag;
defaults are read live from those classes, not duplicated here. Run with --help to
see them all. --policy selects from POLICIES below -- currently just the oracle, but
the registry is the extension point for future baselines (a random-skills policy, a
pure-agent policy, ...) without changing how this script is invoked.
"""

import argparse

from hitl_pmp.core.method.types import Policy
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.oracle_policy import ORACLE_POLICY
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks

POLICIES: dict[str, Policy] = {"oracle": ORACLE_POLICY}


def _add_config_arguments(*, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-test-tasks",
        type=int,
        default=20,
        help="Number of sampled test tasks to run the policy on.",
    )
    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
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
        "--seed",
        type=int,
        default=LightSwitchTasks.seed,
        help="Base RNG seed for task sampling; test uses seed + test-env-seed-offset.",
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


def _apply_config(*, args: argparse.Namespace) -> None:
    LightSwitchEnvironment.grid_size = args.grid_size
    LightSwitchEnvironment.light_on_tolerance = args.light_on_tolerance
    LightSwitchEnvironment.same_position_tolerance = args.same_position_tolerance
    LightSwitchEnvironment.canonical_light_target = args.canonical_light_target
    LightSwitchTasks.test_env_seed_offset = args.test_env_seed_offset
    LightSwitchTasks.target_low = args.target_low
    LightSwitchTasks.target_high = args.target_high
    # Must run last: rederives train_rng/test_rng from seed + (possibly just-updated)
    # test_env_seed_offset.
    LightSwitchTasks.set_seed(seed=args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_config_arguments(parser=parser)
    args = parser.parse_args()
    _apply_config(args=args)

    policy = POLICIES[args.policy]
    LightSwitchEnvironment.hard_reset()

    num_solved = 0
    for i in range(args.num_test_tasks):
        task = LightSwitchTasks.sample_test_task()
        solved = LightSwitchProblem.run_task_episode(task=task, policy=policy)
        num_solved += int(solved)
        print(f"task {i + 1}/{args.num_test_tasks}: {'solved' if solved else 'FAILED'}")

    print(
        f"success rate: {num_solved}/{args.num_test_tasks} ({num_solved / args.num_test_tasks:.0%})"
    )


if __name__ == "__main__":
    main()
