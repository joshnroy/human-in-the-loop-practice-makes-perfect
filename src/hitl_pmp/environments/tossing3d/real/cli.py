"""`--env tossing3d_real`: the Tossing3D practice loop on the real TidyBot.

    python -m hitl_pmp.cli --env tossing3d_real --method pomdp \\
        --preset tossing3d-canonical --robot real --robot-host <nuc> \\
        --practice-step-budget 6000 --num-test-tasks 3 --output-dir <dir>

The same domain, methods, preset (observation weight 0) and step protocol as
`--env tossing3d`. What changes is the backend -- the robot behind the skill server, or
`--robot fake` for a dry run with no robot -- the toss parameter set (the real swing's),
and the human, who is the person at this terminal (`TerminalPerson`). Simulated
execution of the same domain is `--env tossing3d`.

Budgets are in the sim's control steps: a real skill is charged its mean sim tick count
(`RealSkillStepCosts`). Evaluations run in this process, since each evaluation task is
a scene a person sets up and the robot cannot practise and be evaluated at once: the
untrained policy's (step 0) before any practice, and the final one; the interval
defaults past any budget, so no step-interval evaluation runs between them.
`--measure-every-cycles K` adds an evaluation after every K-th learning update, for a
learning curve. After every evaluation but the final one -- the step-0 one included --
the person puts practice's scene back. `--no-initial-measurement` skips step 0.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

from .backend import TerminalPerson
from .contract import DEFAULT_PORT, SkillServerClient
from .environment import RealTossing3DEnvironment
from .fake_robot import FakePerson, FakeTossingRobot
from .nuc_logs import NucLogs
from .session_log import RealSessionLog
from .tasks import RealTossing3DTasks

# Past any practice budget a real run could have, so only the final measurement runs.
END_ONLY_MEASUREMENT_INTERVAL = 10**9


class Tossing3DRealCli:
    """A static-method container, never instantiated."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        Tossing3DCli.add_arguments(parser=parser)
        parser.add_argument(
            "--robot",
            choices=("real", "fake"),
            default="real",
            help="The robot behind the skill-server contract: the TidyBot, or a fake with "
            "no physics and a person who places everything as asked.",
        )
        parser.add_argument("--robot-host", default=None, help="The skill server's host.")
        parser.add_argument("--robot-port", type=int, default=DEFAULT_PORT)
        parser.add_argument(
            "--robot-authkey-file",
            type=Path,
            default=None,
            help="File holding the skill server's RPC authkey; default $TIDYBOT_RPC_AUTHKEY.",
        )
        parser.add_argument(
            "--robot-confirm-each-skill",
            action="store_true",
            help="With --robot real, also ask the person for y before every skill (by "
            "default only before the drive home at a task's start, and at each placement).",
        )
        parser.add_argument(
            "--measure-every-cycles",
            type=Tossing3DRealCli.positive_int,
            default=None,
            metavar="K",
            help="Also evaluate after every K-th learning update (1: after every cycle), in "
            "this process, then ask the person to put practice's scene back. Off by "
            "default: only the final evaluation runs.",
        )
        NucLogs.add_arguments(parser=parser)
        parser.set_defaults(
            measurement_interval_steps=END_ONLY_MEASUREMENT_INTERVAL,
            checkpoint_cycles=True,
            scene_bg=False,
        )

    @staticmethod
    def positive_int(value: str) -> int:  # noqa: PLR0917 -- argparse's `type` callable
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
        return number

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        del args, method_factory, num_cycles, max_steps_per_interaction
        raise ValueError(
            "--env tossing3d_real runs only under the step protocol: pass "
            "--practice-step-budget (with --practice-reset-policy never)"
        )

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_problem(args=args, evaluation=False)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_problem(args=args, evaluation=True)

    @staticmethod
    def build_problem(*, args: argparse.Namespace, evaluation: bool) -> Tossing3DProblem:
        robot: FakeTossingRobot | SkillServerClient
        person: TerminalPerson
        if getattr(args, "output_dir", None) is not None:
            RealSessionLog.configure(output_dir=Path(args.output_dir))
        if args.robot == "fake":
            fake = FakeTossingRobot()
            robot, person = fake, FakePerson(robot=fake)
        else:
            if args.robot_host is None:
                raise ValueError("--robot real needs --robot-host")
            robot = SkillServerClient(
                host=args.robot_host,
                port=args.robot_port,
                authkey=SkillServerClient.read_authkey(path=args.robot_authkey_file),
            )
            person = TerminalPerson()
        env = RealTossing3DEnvironment(
            tossing_robot=robot,
            person=person,
            confirm_start=args.robot == "real",
            confirm_each_skill=args.robot == "real" and args.robot_confirm_each_skill,
            canonical_seed=args.canonical_seed,
            evaluation=evaluation,
        )
        tasks = RealTossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return Tossing3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)
