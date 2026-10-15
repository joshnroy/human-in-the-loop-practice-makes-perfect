"""`--env tossing3d_real`: the Tossing3D practice loop on the real TidyBot.

    python -m hitl_pmp.cli --env tossing3d_real --method pomdp \\
        --robot-host <nuc> --practice-step-budget 60 --practice-reset-policy never \\
        --defer-rendering --num-test-tasks 3 --output-dir <dir>

Same domain, methods and step protocol as `--env tossing3d`; what changes is where the
world comes from (`--robot-plant`), the toss parameter set (the real swing's), and the
human (a person at this terminal, `RealHumanOracle`).

`--robot-plant` picks the robot behind the contract:

| plant | robot | person |
| --- | --- | --- |
| `robot` (default) | the robot side's client, imported from `--robot-client` | this terminal |
| `sim` | the KINDER simulation (`real/sim_plant.py`), the dry run | scripted |
| `fake` | `FakeTossingRobot`, no physics at all | scripted |

**Budgets are in skills.** One real skill is one control step (`real/backend.py`), and
a human reset costs `--human-skill-steps` (default 1). **Evaluation runs at the end
only** by default: `--measurement-interval-steps` defaults past any budget and the
step-0 measurement is off (`--no-initial-measurement`), since every evaluation task is
a scene a person sets up. Give an interval to add checkpoints; practice and evaluation
then share the room, and the practice scene is restored after each evaluation.
"""

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem

from .backend import RealWorld
from .environment import RealTossing3DEnvironment
from .fake_robot import FakeRobotPlacer, FakeTossingRobot
from .human import RealHumanOracle, TerminalScenePlacer
from .tasks import RealTossing3DTasks
from .types import RealSceneLayout

# Past any practice budget a real run could have, so no interval measurement happens.
END_ONLY_MEASUREMENT_INTERVAL = 10**9

REAL_EVENTS_FILENAME = "real_events.jsonl"


class RealWorldRegistry:
    """One `RealWorld` per process and robot: practice and evaluation share the room.

    Keyed on the plant and its address, so the evaluation problem the step protocol
    builds from its own copy of the arguments gets the very robot practice holds.
    """

    _worlds: ClassVar[dict[tuple[str, str | None, int | None, str], RealWorld]] = {}

    @staticmethod
    def world(*, args: argparse.Namespace) -> RealWorld:
        key = (args.robot_plant, args.robot_host, args.robot_port, args.robot_client)
        if key not in RealWorldRegistry._worlds:
            RealWorldRegistry._worlds[key] = RealWorldRegistry.build(args=args)
        return RealWorldRegistry._worlds[key]

    @staticmethod
    def build(*, args: argparse.Namespace) -> RealWorld:
        event_log = (
            None if args.output_dir is None else Path(args.output_dir) / REAL_EVENTS_FILENAME
        )
        if args.robot_plant == "fake":
            fake = FakeTossingRobot()
            return RealWorld(robot=fake, placer=FakeRobotPlacer(robot=fake), event_log=event_log)
        if args.robot_plant == "sim":
            from .sim_plant import SimPlantPlacer, SimPlantTossingRobot

            plant = SimPlantTossingRobot(scene_bg=args.scene_bg)
            return RealWorld(robot=plant, placer=SimPlantPlacer(robot=plant), event_log=event_log)
        if args.robot_host is None:
            raise ValueError("--robot-plant robot needs --robot-host")
        robot = RealWorldRegistry.client(
            spec=args.robot_client, host=args.robot_host, port=args.robot_port
        )
        return RealWorld(robot=robot, placer=TerminalScenePlacer(), event_log=event_log)

    @staticmethod
    def client(*, spec: str, host: str, port: int | None) -> Any:
        """The robot side's client, by import path: this package never depends on it."""
        module_name, _, attribute = spec.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(host=host) if port is None else factory(host=host, port=port)

    @staticmethod
    def clear() -> None:
        RealWorldRegistry._worlds.clear()


class Tossing3DRealCli:
    """A static-method container, never instantiated."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        Tossing3DCli.add_arguments(parser=parser)
        parser.add_argument(
            "--robot-plant",
            choices=("robot", "sim", "fake"),
            default="robot",
            help="What answers the robot contract: the real robot, the KINDER sim (a "
            "dry run), or a fake with no physics.",
        )
        parser.add_argument("--robot-host", default=None, help="The robot skill server.")
        parser.add_argument("--robot-port", type=int, default=None)
        parser.add_argument(
            "--robot-client",
            default="prpl_tidybot.skill_server:TossingRobotClient",
            help="module:factory of the robot side's client, called with host= (and "
            "port= when given). Imported only for --robot-plant robot.",
        )
        parser.add_argument(
            "--real-layout",
            type=Path,
            default=None,
            help="JSON overriding RealSceneLayout (drivable box, footprint, reset "
            "regions). The barrier and the bin are observed, not configured.",
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
        return Tossing3DRealCli.build_problem(args=args, owner="practice")

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_problem(args=args, owner="evaluation")

    @staticmethod
    def build_problem(*, args: argparse.Namespace, owner: str) -> Tossing3DProblem:
        layout = (
            RealSceneLayout()
            if args.real_layout is None
            else RealSceneLayout.model_validate(json.loads(Path(args.real_layout).read_text()))
        )
        env = RealTossing3DEnvironment(
            world=RealWorldRegistry.world(args=args),
            scene_layout=layout,
            owner=owner,
            canonical_seed=args.canonical_seed,
        )
        tasks = RealTossing3DTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return Tossing3DProblem(env=env, tasks=tasks, human=RealHumanOracle)
