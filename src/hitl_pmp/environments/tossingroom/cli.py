import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import TossingRoomEnvironment
from .problem import TossingRoomProblem
from .renderer import TossingRoomRenderer
from .skill_provider import TossingRoomOracle, TossingRoomSkillProvider
from .tasks import TossingRoomGoalType, TossingRoomTasks


class TossingRoomCli:
    """Plugs Tossing Room into the generic runner (see hitl_pmp/cli.py): exposes its
    configurable values as argparse flags, then run_method (below) is this domain's
    composition root -- the one place that constructs TossingRoomEnvironment/
    TossingRoomTasks/TossingRoomProblem from those flags before driving a method
    through PracticeLoop (via method_runner.py's MethodRunner). Mirrors
    LightSwitchCli. A static-method container, never instantiated, same as every
    other business-logic class in this project.

    Unlike LightSwitchCli, there is no apply_config step: every value this domain
    exposes flows through constructor arguments in run_method (nothing stays a
    ClassVar for predicates.py's sake -- see TossingRoomEnvironment's docstring)."""

    render_fps: ClassVar[int] = 2  # slow -- solves are only a handful of actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added
        by hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        fields = TossingRoomEnvironment.model_fields
        parser.add_argument(
            "--num-rooms", type=int, default=fields["num_rooms"].default, help="Number of rooms."
        )
        parser.add_argument(
            "--start-room",
            type=int,
            default=fields["start_room"].default,
            help="Robot's start room and the limitless trash/recycling pile's room.",
        )
        parser.add_argument(
            "--recycling-bin-room",
            type=int,
            default=fields["recycling_bin_room"].default,
            help="Room holding the recycling bin (behind the ledge by default).",
        )
        parser.add_argument(
            "--trash-bin-room",
            type=int,
            default=fields["trash_bin_room"].default,
            help="Room holding the trash bin.",
        )
        parser.add_argument(
            "--button-room",
            type=int,
            default=fields["button_room"].default,
            help="Room holding the empty/incinerate button.",
        )
        parser.add_argument(
            "--blocked-right-from",
            type=int,
            default=fields["blocked_right_from"].default,
            help="The one-way ledge: stepping RIGHT from this room to the next is blocked.",
        )
        parser.add_argument(
            "--throw-tolerance",
            type=float,
            default=fields["throw_tolerance"].default,
            help="Max |force - target| for a Throw to land in the bin.",
        )
        parser.add_argument(
            "--canonical-target-force",
            type=float,
            default=fields["canonical_target_force"].default,
            help="Per-item target force used by Environment.hard_reset (not task sampling).",
        )
        task_fields = TossingRoomTasks.model_fields
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=task_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        parser.add_argument(
            "--target-low",
            type=float,
            default=task_fields["target_low"].default,
            help="Lower bound of a task's sampled throw target (Uniform[low, high)).",
        )
        parser.add_argument(
            "--target-high",
            type=float,
            default=task_fields["target_high"].default,
            help="Upper bound of a task's sampled throw target.",
        )
        parser.add_argument(
            "--goal-type",
            choices=[goal_type.value for goal_type in TossingRoomGoalType],
            default=None,
            help="Pin every sampled task to one goal family (recycling/trash/empty). "
            "Omit to sample the default mix. Use e.g. 'recycling' for a deterministic demo.",
        )

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        """This domain's composition root -- builds the actual TossingRoomEnvironment/
        TossingRoomTasks/TossingRoomProblem from args, bundles this domain's
        SkillProvider/OraclePolicyProvider into a DomainContext, calls
        method_factory(context), then delegates the domain-agnostic rest (driving
        through PracticeLoop, printing, video-writing) to method_runner.py's
        MethodRunner. Mirrors LightSwitchCli.run_method."""
        env = TossingRoomEnvironment(
            num_rooms=args.num_rooms,
            start_room=args.start_room,
            recycling_bin_room=args.recycling_bin_room,
            trash_bin_room=args.trash_bin_room,
            button_room=args.button_room,
            blocked_right_from=args.blocked_right_from,
            throw_tolerance=args.throw_tolerance,
            canonical_target_force=args.canonical_target_force,
        )
        forced_goal_type = (
            TossingRoomGoalType(args.goal_type) if args.goal_type is not None else None
        )
        tasks = TossingRoomTasks(
            env=env,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
            target_low=args.target_low,
            target_high=args.target_high,
            forced_goal_type=forced_goal_type,
        )
        problem = TossingRoomProblem(env=env, tasks=tasks)
        context = DomainContext(
            env=env,
            skill_provider=TossingRoomSkillProvider(env=env),
            oracle=TossingRoomOracle(env=env),
        )

        renderer: type[Renderer] | None = (
            TossingRoomRenderer if args.output_dir is not None else None
        )
        MethodRunner.run(
            args=args,
            method=method_factory(context),
            problem=problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=TossingRoomCli.render_fps,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )
