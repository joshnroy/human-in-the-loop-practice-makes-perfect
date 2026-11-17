"""Isolated rack-to-serving CLI with training/deployment environments by construction."""

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from typing import Literal, get_args, get_origin

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.method_runner import MethodRunner

from .config import RackServingConfig
from .environment import RackServingEnvironment
from .problem import RackServingProblem
from .skill_provider import RackServingOracle, RackServingSkillProvider
from .tasks import RackServingTasks


class RackServingCli:
    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        for name, field in RackServingConfig.model_fields.items():
            if name == "mode":
                continue
            flag = "--" + name.replace("_", "-")
            choices = get_args(field.annotation) if get_origin(field.annotation) is Literal else ()
            if name == "custom_layout":
                parser.add_argument(
                    flag,
                    nargs=15,
                    type=float,
                    default=(),
                    help="start_layout custom: five (along row, up the rack, yaw) triples",
                )
            elif choices:
                parser.add_argument(flag, choices=choices, default=field.default)
            elif isinstance(field.default, tuple):
                parser.add_argument(flag, nargs=3, type=float, default=field.default)
            else:
                parser.add_argument(flag, type=float, default=field.default, help=field.description)
        parser.add_argument("--scene-bg", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--canonical-seed", type=int, default=0)
        parser.add_argument("--skill-step-limit", type=int, default=12000)
        parser.add_argument("--test-env-seed-offset", type=int, default=10000)
        parser.add_argument("--human-reset", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--record-motion",
            choices=list(SweepOffShelf3DCli.MOTION_SCOPES),
            default="off",
            help="Step protocol only: stream every control step's state, arm joints and "
            "every free body included, into the run recording so render_recording.py can "
            "draw full motion. 'evaluation' records evaluation episodes; 'all' also records "
            "practice. A pure observer: results are unchanged.",
        )
        parser.set_defaults(defer_rendering=True)

    @staticmethod
    def build_problem(
        *, args: argparse.Namespace, mode: Literal["training", "deployment"]
    ) -> RackServingProblem:
        values = {
            name: mode if name == "mode" else getattr(args, name)
            for name in RackServingConfig.model_fields
        }
        flat = values["custom_layout"]
        if len(flat) == 15:
            values["custom_layout"] = tuple(tuple(flat[i : i + 3]) for i in range(0, 15, 3))
        config = RackServingConfig.model_validate(values)
        practice, evaluation = SweepOffShelf3DCli.MOTION_SCOPES[
            getattr(args, "record_motion", "off")
        ]
        env = RackServingEnvironment(
            config=config,
            canonical_seed=args.canonical_seed,
            scene_bg=args.scene_bg,
            skill_step_limit=args.skill_step_limit,
            motion_recording=practice if mode == "training" else evaluation,
        )
        tasks = RackServingTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return RackServingProblem(
            env=env,
            tasks=tasks,
            human=UnconditionalHumanOracle if mode == "training" else None,
        )

    @staticmethod
    def practice_context(*, args: argparse.Namespace, env: RackServingEnvironment) -> DomainContext:
        return DomainContext(
            env=env,
            skill_provider=RackServingSkillProvider(env=env, offer_human_reset=args.human_reset),
            oracle=RackServingOracle(env=env),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        if getattr(args, "record_motion", "off") != "off":
            raise ValueError(
                "--record-motion needs the step protocol (--practice-step-budget): only it "
                "writes a run recording"
            )
        with ExitStack() as resources:
            practice = RackServingCli.build_problem(args=args, mode="training")
            resources.callback(practice.env.close)
            evaluation = RackServingCli.build_problem(args=args, mode="deployment")
            resources.callback(evaluation.env.close)
            context = RackServingCli.practice_context(args=args, env=practice.env)
            MethodRunner.run(
                args=args,
                method=method_factory(context),
                problem=practice,
                evaluation_problem=evaluation,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                renderer=None,
                render_fps=1,
                num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
            )
