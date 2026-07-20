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
from pathlib import Path

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

ENVIRONMENTS = {"lightswitch": LightSwitchCli}


class Cli:
    """The global CLI dispatcher. A static-method container, never instantiated,
    same as every other business-logic class in this project (including
    LightSwitchCli, which this delegates to once --env is known)."""

    @staticmethod
    def parse_positive_int(*, value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
        return parsed

    @staticmethod
    def add_global_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--env",
            choices=sorted(ENVIRONMENTS),
            required=True,
            default=argparse.SUPPRESS,
            help="Which environment to run. Determines which other flags are valid. "
            "Also reserves this name and --seed/--num-test-tasks/--output-dir: no "
            "environment's own add_arguments may redefine them (argparse itself will "
            "raise a clear conflicting-option-string error if one ever tries).",
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
            type=lambda value: Cli.parse_positive_int(value=value),
            default=20,
            help="Number of sampled test tasks to run the policy on. Must be >= 1.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="If set, additionally render one demo episode to <output-dir>/"
            "episode.mp4 and write run statistics to <output-dir>/stats.json. "
            "Disabled (nothing written) if omitted.",
        )

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
        # --env determines which other flags are valid, so they can't all be
        # registered up front. A plain two-pass parse_known_args here would exit
        # early on --help before ever reaching those environment-specific flags
        # (argparse's help action fires mid-scan, during the *first* pass) -- so
        # this discovery pass disables its own help handling and doesn't require
        # --env, purely to learn --env's value without ever printing anything or
        # exiting. The real parser built below always has the full flag set (global
        # + the selected environment's, if any), so it's the one that ever shows
        # --help or reports a genuine "missing --env" error.
        discovery = argparse.ArgumentParser(add_help=False)
        discovery.add_argument("--env", choices=sorted(ENVIRONMENTS))
        known_args, _ = discovery.parse_known_args(argv)

        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        Cli.add_global_arguments(parser=parser)
        if known_args.env is not None:
            ENVIRONMENTS[known_args.env].add_arguments(parser=parser)
        return parser.parse_args(argv)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        args = Cli.parse_args(argv=argv)
        ENVIRONMENTS[args.env].run(args=args)


if __name__ == "__main__":
    Cli.main()
