"""The global CLI: pick an environment, then drive a registered core.Method through
PracticeLoop (--method) -- the one execution harness every Method runs through,
whether it learns or not (a non-learning Method, e.g. a privileged oracle, is simply
--num-cycles 0 through the same loop; see practice_loop.py's own docstring). Every
flag is named -- no positional arguments anywhere.

Global flags (--seed, --num-test-tasks, --env, --method) are generic runner concepts
meaningful for any environment/method. Environment-specific flags depend on --env's
value, and a method's own flags come from its own add_arguments:
    python -m hitl_pmp.cli --env lightswitch --method skill-oracle --num-test-tasks 5

Every environment plugs into this file by having its own cli.py that exposes an
add_arguments(*, parser) pair (see environments/lightswitch/cli.py's LightSwitchCli)
and registering itself in ENVIRONMENTS below, purely for its own config flags
(--grid-size etc.) -- an environment is never run directly, only via a --method that
wires itself to it. Every method plugs in the same way via its own
add_arguments(*, parser)/run(*, args) pair, registering in METHODS -- this file
itself has no domain- or method-specific knowledge.
"""

import argparse
from pathlib import Path

from hitl_pmp.cli_protocols import EnvironmentCli, MethodCli
from hitl_pmp.environments.ballring.cli import BallRingCli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossingroom.cli import TossingRoomCli
from hitl_pmp.methods.oracle.cli import SkillOracleCli
from hitl_pmp.methods.practice_makes_perfect.cli import EesCli, RandomSkillsCli

# Registering tossing3d here costs nothing on a machine without KINDER: importing
# Tossing3DCli imports no simulator (see environments/tossing3d/kinder_backend.py), so
# --env lightswitch still works, --help still lists every flag, and only actually
# *running* --env tossing3d needs the optional dependency.
ENVIRONMENTS: dict[str, type[EnvironmentCli]] = {
    "ballring": BallRingCli,
    "lightswitch": LightSwitchCli,
    "tossing3d": Tossing3DCli,
    "tossingroom": TossingRoomCli,
}


METHODS: dict[str, type[MethodCli]] = {
    "skill-oracle": SkillOracleCli,
    "random-skills": RandomSkillsCli,
    "ees": EesCli,
}


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
            required=True,
            default=argparse.SUPPRESS,
            help="Which core.Method to drive through PracticeLoop. Determines which "
            "method-specific flags are valid.",
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
            default=10,
            help="Number of test tasks to evaluate the policy on. Must be >= 1. Drawn "
            "once and reused by every evaluation sweep. 10 is the paper's own value "
            "(active_sampler_learning.yaml's num_test_tasks).",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="If set, additionally write <output-dir>/stats.json (the run's "
            "Metrics) and render demo episodes to <output-dir>/episode*.mp4. "
            "Disabled (nothing written) if omitted.",
        )
        parser.add_argument(
            "--practice-reset-interval",
            type=lambda value: Cli.parse_positive_int(value=value),
            default=None,
            help="Put the environment back to the current practice task's initial "
            "state every N steps *within* an interaction period, without ending the "
            "cycle or retraining. Omitted (the default) means the only reset is the "
            "one at the top of each cycle, so reset frequency is whatever "
            "--max-steps-per-interaction happens to be. Set it to decouple how often "
            "the robot is rescued from how often the method refits.",
        )
        parser.add_argument(
            "--record-full-loop",
            type=Path,
            default=None,
            help="Record the ENTIRE outer loop to this .mp4: every practice period, "
            "every evaluation episode of every sweep, and every environment reset, in "
            "one continuous seekable timeline, with a status bar showing which phase "
            "the loop is in and a distinct marker per reset kind. Independent of "
            "--output-dir/--num-render-checkpoints, which record evaluation episodes "
            "only. Off by default; a recorded run takes exactly the same actions as an "
            "unrecorded one, it just renders far more of them, so keep the run small "
            "(a few cycles) if the point is to see the structure.",
        )
        parser.add_argument(
            "--num-render-checkpoints",
            type=lambda value: Cli.parse_positive_int(value=value),
            default=1,
            help="How many evaluation sweeps to record, spread evenly from before "
            "any practice through the end of training. 1 (default) records only "
            "the finished policy; a larger value produces a visible progression. "
            "Only meaningful with --output-dir and a learning --method.",
        )

    @staticmethod
    def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
        # --env/--method determine which other flags are valid, so they can't all
        # be registered up front. A plain two-pass parse_known_args here would exit
        # early on --help before ever reaching those specific flags (argparse's
        # help action fires mid-scan, during the *first* pass) -- so this discovery
        # pass disables its own help handling and doesn't require --env/--method,
        # purely to learn their values without ever printing anything or exiting.
        # The real parser built below always has the full flag set (global + the
        # selected environment's/method's), so it's the one that ever shows --help
        # or reports a genuine "missing --env"/"missing --method" error.
        discovery = argparse.ArgumentParser(add_help=False)
        discovery.add_argument("--env", choices=sorted(ENVIRONMENTS))
        discovery.add_argument("--method", choices=sorted(METHODS))
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
        # cli.py is the one place allowed to import both environments/ and methods/,
        # so it is where the selected env-CLI (its composition root) and method-CLI
        # are wired together -- the method-CLI drives `env_cli.run_method`, so a
        # method never imports a specific environment.
        METHODS[args.method].run(args=args, env_cli=ENVIRONMENTS[args.env])


if __name__ == "__main__":
    Cli.main()
