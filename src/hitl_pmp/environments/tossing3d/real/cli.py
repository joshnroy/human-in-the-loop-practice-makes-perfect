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
(`RealSkillStepCosts`). Evaluation runs once, at the end, in this process: the interval
defaults past any budget and the step-0 measurement is off, since each evaluation task
is a scene a person sets up and the robot cannot practise and be evaluated at once.
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
        parser.set_defaults(
            measurement_interval_steps=END_ONLY_MEASUREMENT_INTERVAL,
            initial_measurement=False,
            scene_bg=False,
        )

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
            canonical_seed=args.canonical_seed,
            evaluation=evaluation,
        )
        tasks = RealTossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return Tossing3DProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)
