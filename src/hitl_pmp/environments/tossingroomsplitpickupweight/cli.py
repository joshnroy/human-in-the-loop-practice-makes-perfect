import argparse
from collections.abc import Callable
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.renderer.renderer import Renderer
from hitl_pmp.method_runner import MethodRunner

from .environment import TossingRoomSplitPickupWeightEnvironment
from .problem import TossingRoomSplitPickupWeightProblem
from .renderer import TossingRoomSplitPickupWeightRenderer
from .skill_provider import (
    TossingRoomSplitPickupWeightOracle,
    TossingRoomSplitPickupWeightSkillProvider,
)
from .tasks import TossingRoomSplitPickupWeightGoalType, TossingRoomSplitPickupWeightTasks


class TossingRoomSplitPickupWeightCli:
    """Plugs Tossing Room (split throws) into the generic runner (see hitl_pmp/cli.py):
    exposes its configurable values as argparse flags, then run_method (below) is this
    domain's composition root.

    The flag set is deliberately identical to `TossingRoomCli`'s, including the
    defaults, so the two domains can be run with the same command line and any
    difference in results is the skill split rather than a differently configured
    world. A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    render_fps: ClassVar[int] = 2  # slow -- solves are only a handful of actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags added
        by hitl_pmp/cli.py, not here -- everything below is specific to this domain."""
        fields = TossingRoomSplitPickupWeightEnvironment.model_fields
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
            "--two-way-ledge",
            action="store_true",
            help="Make the ledge traversable RIGHTWARD too, so the domain has no "
            "irreversible action at all. THE DOMAIN GETS EASIER: EMPTY's solve drops "
            "10 -> 9, the evaluation horizon 12 -> 11, and RECYCLING stops being "
            "one-shot -- so a two-way number is never directly comparable to a one-way "
            "one. Default off.",
        )
        parser.add_argument(
            "--throw-tolerance",
            type=float,
            default=fields["throw_tolerance"].default,
            help="Max |force - required_force| for a throw to land in the bin.",
        )
        # The unobserved required-force relation, in reference form: a reference_weight
        # item into a bin reference_distance away needs reference_force, and each
        # coefficient is the force per unit away from that reference. Same flags and same
        # defaults as Tossing Room's, deliberately -- the two domains must stay the same
        # learning problem under two skill decompositions.
        parser.add_argument(
            "--reference-force",
            type=float,
            default=fields["reference_force"].default,
            help="Force the reference throw needs, in the (unobserved) required-force relation.",
        )
        parser.add_argument(
            "--reference-distance",
            type=float,
            default=fields["reference_distance"].default,
            help="Bin throw distance at which --reference-force is exactly right.",
        )
        parser.add_argument(
            "--reference-weight",
            type=float,
            default=fields["reference_weight"].default,
            help="Item weight at which --reference-force is exactly right.",
        )
        parser.add_argument(
            "--distance-coefficient",
            type=float,
            default=fields["distance_coefficient"].default,
            help="Extra required force per unit of bin throw_distance beyond the reference.",
        )
        parser.add_argument(
            "--weight-coefficient",
            type=float,
            default=fields["weight_coefficient"].default,
            help="Extra required force per unit of item weight beyond the reference.",
        )
        parser.add_argument(
            "--throw-distance",
            type=float,
            default=fields["throw_distance"].default,
            help="Every bin's throw distance, FIXED rather than sampled per task -- so "
            "required_force is a one-dimensional function of the item's weight.",
        )
        parser.add_argument(
            "--canonical-item-weight",
            type=float,
            default=fields["canonical_item_weight"].default,
            help="Placeholder item weight before any pickup, and the value "
            "Environment.hard_reset starts from (not task sampling).",
        )
        parser.add_argument(
            "--pickup-weight-low",
            type=float,
            default=fields["pickup_weight_low"].default,
            help="Lower bound of the weight drawn at each pickup (Uniform[low, high)).",
        )
        parser.add_argument(
            "--pickup-weight-high",
            type=float,
            default=fields["pickup_weight_high"].default,
            help="Upper bound of the weight drawn at each pickup.",
        )
        parser.add_argument(
            "--weight-schedule-length",
            type=int,
            default=None,
            help="How many pickups each task's pre-sampled weight array covers. Default: "
            "sized from the run's own step budget. Running off the end raises -- the "
            "schedule never wraps.",
        )
        task_fields = TossingRoomSplitPickupWeightTasks.model_fields
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=task_fields["test_env_seed_offset"].default,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        parser.add_argument(
            "--goal-type",
            choices=[goal_type.value for goal_type in TossingRoomSplitPickupWeightGoalType],
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
        method_runner.py's MethodRunner.

        It builds that triple **twice**: one for practice and a wholly separate one for
        evaluation. Every evaluation episode opens with `reset_to_task`, a privileged
        state-write, so sharing the triple hands the practice environment
        `--num-test-tasks` free resets per sweep -- 30 of them by default. That is
        invisible under the per-period reset and fatal to any reset-free practice arm.
        See PracticeLoop's own "separate evaluation environment" section.

        Doing this is only sound because this domain's environment consumes **no**
        randomness (`take_action` is pure, and the only RNG lives in Tasks' train/test
        streams, which are derived independently from the same seed). A second instance
        therefore cannot shift what practice draws, and the split is a no-op on results
        -- pinned by the byte-identity of `stats.json`. Domains whose environment does
        draw (`ballring`'s `_noise_rng`) must not be wired this way without a
        re-baseline.

        That claim needs restating for THIS domain, because a weight is now drawn per
        pickup rather than per task: the draw is a lookup into an array pre-sampled by
        `build_initial_state` from a seed carried in the State, so `take_action` still
        consumes no randomness and a second instance still derives byte-identical
        arrays. See `TossingRoomSplitPickupWeightEnvironment.weight_schedule`."""
        practice_problem = TossingRoomSplitPickupWeightCli.build_problem(
            args=args,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
        )
        # Same args, same seed, independent objects: its Tasks derives the same test
        # stream, so it yields exactly the test tasks the practice Tasks would have.
        evaluation_problem = TossingRoomSplitPickupWeightCli.build_problem(
            args=args,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
        )
        # The Method is wired to the *practice* environment, deliberately: its env
        # reference is for structural config (skills, predicates, layout), and the two
        # instances are configured identically, so evaluation reads the same world.
        context = DomainContext(
            env=practice_problem.env,
            skill_provider=TossingRoomSplitPickupWeightSkillProvider(env=practice_problem.env),
            oracle=TossingRoomSplitPickupWeightOracle(env=practice_problem.env),
        )

        renderer: type[Renderer] | None = (
            TossingRoomSplitPickupWeightRenderer if args.output_dir is not None else None
        )
        MethodRunner.run(
            args=args,
            method=method_factory(context),
            problem=practice_problem,
            evaluation_problem=evaluation_problem,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            renderer=renderer,
            render_fps=TossingRoomSplitPickupWeightCli.render_fps,
            num_render_checkpoints=getattr(args, "num_render_checkpoints", 1),
        )

    @staticmethod
    def build_problem(
        *,
        args: argparse.Namespace,
        num_cycles: int = 0,
        max_steps_per_interaction: int = 0,
    ) -> TossingRoomSplitPickupWeightProblem:
        """One fully-independent Environment + Tasks + Problem triple from args.
        Called twice by run_method (practice and evaluation) -- factored out so the two
        cannot drift apart in configuration, which would silently make the evaluation
        set a different set of tasks.

        `num_cycles`/`max_steps_per_interaction` are the run's step budget, and are here
        only to size the pre-sampled weight schedule -- see `weight_schedule_length`
        below. They default to 0 so a caller that only wants a Problem (a test, a
        one-off) still gets one, at the Environment's own default length."""
        env = TossingRoomSplitPickupWeightEnvironment(
            num_rooms=args.num_rooms,
            start_room=args.start_room,
            recycling_bin_room=args.recycling_bin_room,
            trash_bin_room=args.trash_bin_room,
            blocked_right_from=args.blocked_right_from,
            two_way_ledge=args.two_way_ledge,
            throw_tolerance=args.throw_tolerance,
            reference_force=args.reference_force,
            reference_distance=args.reference_distance,
            reference_weight=args.reference_weight,
            distance_coefficient=args.distance_coefficient,
            weight_coefficient=args.weight_coefficient,
            throw_distance=args.throw_distance,
            canonical_item_weight=args.canonical_item_weight,
            pickup_weight_low=args.pickup_weight_low,
            pickup_weight_high=args.pickup_weight_high,
            weight_schedule_length=TossingRoomSplitPickupWeightCli.weight_schedule_length(
                args=args,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
            ),
            # Wired from --seed, deliberately. Under --practice-reset-policy never the
            # hard_reset state is the only one practice ever sees, so leaving this at a
            # constant would hand every seed of a sweep the same practice weights -- the
            # BallRingEnvironment --noise-seed trap, which defaults to 0 and makes every
            # arm of a sweep identical in the one place it must not be.
            canonical_weight_seed=args.seed,
        )
        forced_goal_type = (
            TossingRoomSplitPickupWeightGoalType(args.goal_type)
            if args.goal_type is not None
            else None
        )
        tasks = TossingRoomSplitPickupWeightTasks(
            env=env,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
            forced_goal_type=forced_goal_type,
            # The global --num-test-tasks: this domain's test set has a *fixed*
            # goal-family composition (14 TRASH / 14 RECYCLING / 2 EMPTY at 30 tasks), so
            # Tasks has to know how many test tasks the harness will draw in order to
            # divide them up. See TossingRoomSplitPickupWeightTasks.test_goal_type_counts.
            num_test_tasks=args.num_test_tasks,
        )
        return TossingRoomSplitPickupWeightProblem(env=env, tasks=tasks)

    @staticmethod
    def weight_schedule_length(
        *, args: argparse.Namespace, num_cycles: int, max_steps_per_interaction: int
    ) -> int:
        """How many pickups each task's pre-sampled weight array has to cover.

        Every pickup costs one environment step, so the run's own step budget is an
        exact upper bound on how many weights any single task can consume. Under the
        default per-period reset the cursor rewinds every period, so
        `max_steps_per_interaction` would do -- but under `--practice-reset-policy never`
        nothing ever rewinds it and the whole run walks one array, so the bound has to be
        `num_cycles * max_steps_per_interaction`. The larger bound is taken
        unconditionally: sizing off the reset policy would make the arrays, and therefore
        the weights, differ between the two arms of exactly the A/B this domain exists
        for.

        A floor of the Environment's own default keeps a degenerate budget (a 0-cycle
        oracle run, whose evaluation episodes still pick things up) from producing a
        zero-length array. An explicit --weight-schedule-length overrides all of it."""
        override = getattr(args, "weight_schedule_length", None)
        if override is not None:
            return int(override)
        floor = int(
            TossingRoomSplitPickupWeightEnvironment.model_fields["weight_schedule_length"].default
        )
        return max(floor, num_cycles * max_steps_per_interaction)
