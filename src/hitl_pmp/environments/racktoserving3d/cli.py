"""Compose `--env racktoserving3d`: the 3D Serving scene behind exactly the chained toy's
interface (`toy_interface.py`), with training/deployment environments by construction.

Every physical `RackServingConfig` field is a flag. The fields that only shaped the
retired planner interface (`PLANNER_FIELDS`) are not: the scene's internal driver
(`RackServingEnvironment.take_action`) still reads them, at their defaults."""

import argparse
from collections.abc import Callable
from typing import ClassVar, Literal, get_args, get_origin

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

from .config import RackServingConfig
from .problem import RackServingProblem
from .skill_provider import RackServingOracle, RackServingSkillProvider
from .tasks import RackServingToyTasks
from .toy_environment import RackServingToyEnvironment


class RackServingCli:
    PLANNER_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "two_route_rewards",
        "bulk_transfer_only",
        "sweep_skill",
        "split_gather_and_sweep",
        "served_three_reward",
        "served_five_reward",
        "sampler_features",
        "robot_batch_return",
    })

    @staticmethod
    def flagged_fields() -> list[str]:
        return [
            name
            for name in RackServingConfig.model_fields
            if name != "mode" and name not in RackServingCli.PLANNER_FIELDS
        ]

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        for name in RackServingCli.flagged_fields():
            field = RackServingConfig.model_fields[name]
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
        # The toy's full rack: robot_reset puts each dumpling back on its slot, so the
        # start has slots.
        parser.set_defaults(start_layout="middle_three")
        fields = RackServingToyEnvironment.model_fields
        for flag, name, text in (
            (
                "--robot-reset-price",
                "robot_reset_price",
                "What one robot_reset costs, under a cost file that uses "
                "hitl_pmp.environments.tworoutes.costs:robot_step",
            ),
            ("--sweep3-reward", "sweep3_reward", "What serving three by sweep3 is worth"),
            ("--sweep5-reward", "sweep5_reward", "What serving five by sweep5 is worth"),
        ):
            parser.add_argument(flag, type=float, default=fields[name].default, help=text)
        for flag, text in (
            (
                "--gather-success-half-width",
                "Opt-in: gather succeeds only if every dumpling not on the plate lies "
                "within this many metres of the plate's centre line (default: the blade "
                "band alone)",
            ),
            (
                "--gather-success-depth",
                "With --gather-success-half-width: and no farther than this many metres "
                "behind the rack's front edge (default: no depth bound)",
            ),
        ):
            parser.add_argument(flag, type=float, default=None, help=text)
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
    def config(
        *, args: argparse.Namespace, mode: Literal["training", "deployment"]
    ) -> RackServingConfig:
        """The flags' values, `mode`, and the robot's batch return switched on: its
        validation is robot_reset's (a start layout with slots, welded food)."""
        values = {name: getattr(args, name) for name in RackServingCli.flagged_fields()}
        flat = values["custom_layout"]
        if len(flat) == 15:
            values["custom_layout"] = tuple(tuple(flat[i : i + 3]) for i in range(0, 15, 3))
        return RackServingConfig.model_validate({
            **values,
            "mode": mode,
            "robot_batch_return": True,
        })

    @staticmethod
    def build_problem(
        *, args: argparse.Namespace, mode: Literal["training", "deployment"]
    ) -> RackServingProblem:
        practice, evaluation = SweepOffShelf3DCli.MOTION_SCOPES[
            getattr(args, "record_motion", "off")
        ]
        env = RackServingToyEnvironment(
            config=RackServingCli.config(args=args, mode=mode),
            canonical_seed=args.canonical_seed,
            scene_bg=args.scene_bg,
            skill_step_limit=args.skill_step_limit,
            motion_recording=practice if mode == "training" else evaluation,
            robot_reset_price=args.robot_reset_price,
            sweep3_reward=args.sweep3_reward,
            sweep5_reward=args.sweep5_reward,
            gather_success_half_width=args.gather_success_half_width,
            gather_success_depth=args.gather_success_depth,
        )
        tasks = RackServingToyTasks(
            env=env, seed=args.seed, test_env_seed_offset=args.test_env_seed_offset
        )
        return RackServingProblem(
            env=env,
            tasks=tasks,
            human=UnconditionalHumanOracle if mode == "training" else None,
        )

    @staticmethod
    def practice_context(
        *, args: argparse.Namespace, env: RackServingToyEnvironment
    ) -> DomainContext:
        return DomainContext(
            env=env,
            skill_provider=RackServingSkillProvider(env=env, offer_human_reset=args.human_reset),
            oracle=RackServingOracle(env=env),
            test_task_goals=(RackServingToyTasks(env=env).goal(),),
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """Never reached: Serving runs only under the step protocol, as the toy does
        (`TwoRoutesCli.run_method`)."""
        del args, method_factory, num_cycles, max_steps_per_interaction
        raise ValueError("--env racktoserving3d runs only under --practice-step-budget")
