"""The global CLI: pick an environment, configure it via a mix of global and
environment-specific flags, and run a chosen policy over sampled test tasks. Every
flag is named -- no positional arguments anywhere.

Global flags (--seed, --num-test-tasks, --env) are generic runner concepts meaningful
for any environment. Environment-specific flags depend on --env's value:
    python -m hitl_pmp.cli --seed 7 --num-test-tasks 5 --env lightswitch --grid-size 3

Every environment (and, once one exists, every method) plugs into this file by having
its own cli.py that exposes an add_arguments(*, parser)/run(*, args) pair (see
environments/lightswitch/cli.py's LightSwitchCli) and registering itself in
ENVIRONMENTS below -- this file itself has no domain-specific knowledge.
"""

import argparse

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

ENVIRONMENTS = {"lightswitch": LightSwitchCli}


def _add_global_arguments(*, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENTS),
        required=True,
        help="Which environment to run. Determines which other flags are valid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for task sampling. Meaning of derived streams (e.g. "
        "train/test) is up to the selected environment.",
    )
    parser.add_argument(
        "--num-test-tasks",
        type=int,
        default=20,
        help="Number of sampled test tasks to run the policy on.",
    )


def _parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_global_arguments(parser=parser)
    # --env determines which other flags are valid, so they can't all be registered
    # up front: parse what's known so far (ignoring the as-yet-unregistered
    # environment-specific flags), then add those and parse the full argv for real.
    known_args, _ = parser.parse_known_args(argv)
    ENVIRONMENTS[known_args.env].add_arguments(parser=parser)
    return parser.parse_args(argv)


def main(*, argv: list[str] | None = None) -> None:
    args = _parse_args(argv=argv)
    ENVIRONMENTS[args.env].run(args=args)


if __name__ == "__main__":
    main()
