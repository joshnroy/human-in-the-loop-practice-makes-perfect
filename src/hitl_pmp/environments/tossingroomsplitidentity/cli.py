import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import TossingRoomSplitIdentityEnvironment
from .problem import TossingRoomSplitIdentityProblem
from .renderer import TossingRoomSplitIdentityRenderer
from .skill_provider import TossingRoomSplitIdentityOracle, TossingRoomSplitIdentitySkillProvider
from .tasks import TossingRoomSplitIdentityGoalType, TossingRoomSplitIdentityTasks


class TossingRoomSplitIdentityCli:
    """Plugs Tossing Room (split throws) into the generic runner (see hitl_pmp/cli.py):
    exposes its configurable values as argparse flags, then run_method (below) is this
    domain's composition root.

    The flag set is `TossingRoomSplitCli`'s **minus the required-force relation and its
    two per-task cause ranges, plus a single target-force range** -- the one place the
    two arms genuinely differ. Every layout flag keeps the same name and the same
    default, so the two domains can be run with the same command line and any difference
    in results is the throw representation rather than a differently configured world.

    A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    render_fps: ClassVar[int] = 2  # slow -- solves are only a handful of actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added
        by hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        fields = TossingRoomSplitIdentityEnvironment.model_fields
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
        # There is no --button-room: each bin's empty/incinerate button sits in that
        # bin's own room, so --recycling-bin-room/--trash-bin-room place the buttons too.
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
            help="Max |force - item.target_force| for a throw to land in the bin.",
        )
        # No relation flags on the ENVIRONMENT: under the identity representation the
        # dynamics have no relation to configure -- the required force IS
        # item.target_force. The causal arm's five constants still exist, but as
        # TossingRoomSplitIdentityTasks fields, because here they shape only the task
        # DISTRIBUTION (they are what makes this arm's marginal match the causal arm's).
        # They are deliberately not exposed as flags: changing them would silently break
        # that match, which is the one thing this domain must not let a caller do by
        # accident.
        parser.add_argument(
            "--canonical-target-force",
            type=float,
            default=fields["canonical_target_force"].default,
            help="Per-item target force used by Environment.hard_reset (not task sampling).",
        )
        task_fields = TossingRoomSplitIdentityTasks.model_fields
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=task_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        # Same names, same defaults as the causal arm's: this arm draws the identical two
        # causes per task and merely resolves them into the target before putting it in
        # the State. Keeping the flags identical is what lets both arms be run from one
        # command line. See TossingRoomSplitIdentityTasks.
        parser.add_argument(
            "--distance-low",
            type=float,
            default=task_fields["distance_low"].default,
            help="Lower bound of a task's sampled bin throw distance (Uniform[low, high)).",
        )
        parser.add_argument(
            "--distance-high",
            type=float,
            default=task_fields["distance_high"].default,
            help="Upper bound of a task's sampled bin throw distance.",
        )
        parser.add_argument(
            "--weight-low",
            type=float,
            default=task_fields["weight_low"].default,
            help="Lower bound of a task's sampled item weight (Uniform[low, high)).",
        )
        parser.add_argument(
            "--weight-high",
            type=float,
            default=task_fields["weight_high"].default,
            help="Upper bound of a task's sampled item weight.",
        )
        parser.add_argument(
            "--goal-type",
            choices=[goal_type.value for goal_type in TossingRoomSplitIdentityGoalType],
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
        """This domain's composition root -- builds the actual environment/tasks/problem
        from args, bundles this domain's SkillProvider/OraclePolicyProvider into a
        DomainContext, calls method_factory(context), then delegates the domain-agnostic
        rest (driving through PracticeLoop, printing, video-writing) to
        method_runner.py's MethodRunner."""
        env = TossingRoomSplitIdentityEnvironment(
            num_rooms=args.num_rooms,
            start_room=args.start_room,
            recycling_bin_room=args.recycling_bin_room,
            trash_bin_room=args.trash_bin_room,
            blocked_right_from=args.blocked_right_from,
            throw_tolerance=args.throw_tolerance,
            canonical_target_force=args.canonical_target_force,
        )
        forced_goal_type = (
            TossingRoomSplitIdentityGoalType(args.goal_type) if args.goal_type is not None else None
        )
        tasks = TossingRoomSplitIdentityTasks(
            env=env,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
            distance_low=args.distance_low,
            distance_high=args.distance_high,
            weight_low=args.weight_low,
            weight_high=args.weight_high,
            forced_goal_type=forced_goal_type,
            # The global --num-test-tasks: this domain's test set has a *fixed*
            # goal-family composition (14 TRASH / 14 RECYCLING / 2 EMPTY at 30 tasks), so
            # Tasks has to know how many test tasks the harness will draw in order to
            # divide them up. See TossingRoomSplitIdentityTasks.test_goal_type_counts.
            num_test_tasks=args.num_test_tasks,
        )
        problem = TossingRoomSplitIdentityProblem(env=env, tasks=tasks)
        context = DomainContext(
            env=env,
            skill_provider=TossingRoomSplitIdentitySkillProvider(env=env),
            oracle=TossingRoomSplitIdentityOracle(env=env),
        )

        renderer: type[Renderer] | None = (
            TossingRoomSplitIdentityRenderer if args.output_dir is not None else None
        )
        MethodRunner.run(
            args=args,
            method=method_factory(context),
            problem=problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=TossingRoomSplitIdentityCli.render_fps,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )
