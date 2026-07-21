"""The global CLI: pick an environment, then either run one of its raw baseline
policies over sampled test tasks (--policy, an environment-specific flag) or drive
a registered core.Method through PracticeLoop (--method, a global flag here since a
Method is meant to work across environments, not tied to one). Every flag is named
-- no positional arguments anywhere.

Global flags (--seed, --num-test-tasks, --env, --method) are generic runner concepts
meaningful for any environment/method. Environment-specific flags depend on --env's
value, and a method's own flags (once --method is chosen) come from its own
add_arguments:
    python -m hitl_pmp.cli --seed 7 --num-test-tasks 5 --env lightswitch --grid-size 3

Every environment plugs into this file by having its own cli.py that exposes an
add_arguments(*, parser)/run(*, args) pair (see environments/lightswitch/cli.py's
LightSwitchCli) and registering itself in ENVIRONMENTS below; every method does the
same via methods/<name>/cli.py, registering in METHODS -- this file itself has no
domain- or method-specific knowledge. METHODS is empty for now: nothing implements
core.Method yet (see methods/README.md).
"""

import argparse
from pathlib import Path

from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

ENVIRONMENTS = {"lightswitch": LightSwitchCli}


class MethodCli:
    """The add_arguments(*, parser)/run(*, args) shape every methods/<name>/cli.py
    entry in METHODS must expose -- mirrors environments/<domain>/cli.py's
    (unnamed, structurally-typed) convention. Exists purely for METHODS' own type
    annotation; concrete method CLIs don't need to subclass this."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        raise NotImplementedError


METHODS: dict[str, type[MethodCli]] = {}


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
            "Also reserves this name and --seed/--num-test-tasks/--output-dir/--method: "
            "no environment's own add_arguments may redefine them (argparse itself will "
            "raise a clear conflicting-option-string error if one ever tries).",
        )
        parser.add_argument(
            "--method",
            choices=sorted(METHODS),
            default=None,
            help="If set, drives this core.Method through PracticeLoop instead of "
            "running one of --env's own raw baseline policies (see --policy, an "
            "environment-specific flag). Determines which method-specific flags are "
            "valid. Empty for now -- nothing implements core.Method yet.",
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
            "episode.mp4. Disabled (nothing written) if omitted.",
        )

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
        # --env/--method determine which other flags are valid, so they can't all
        # be registered up front. A plain two-pass parse_known_args here would exit
        # early on --help before ever reaching those specific flags (argparse's
        # help action fires mid-scan, during the *first* pass) -- so this discovery
        # pass disables its own help handling and doesn't require --env, purely to
        # learn --env/--method's values without ever printing anything or exiting.
        # The real parser built below always has the full flag set (global + the
        # selected environment's/method's), so it's the one that ever shows --help
        # or reports a genuine "missing --env" error.
        discovery = argparse.ArgumentParser(add_help=False)
        discovery.add_argument("--env", choices=sorted(ENVIRONMENTS))
        discovery.add_argument("--method", choices=sorted(METHODS), default=None)
        known_args, _ = discovery.parse_known_args(argv)

        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        Cli.add_global_arguments(parser=parser)
        if known_args.env is not None:
            ENVIRONMENTS[known_args.env].add_arguments(parser=parser)
        if known_args.method is not None:
            METHODS[known_args.method].add_arguments(parser=parser)
        return parser.parse_args(argv)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        args = Cli.parse_args(argv=argv)
        if args.method is not None:
            METHODS[args.method].run(args=args)
        else:
            ENVIRONMENTS[args.env].run(args=args)


if __name__ == "__main__":
    Cli.main()
