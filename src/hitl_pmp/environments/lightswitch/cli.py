import argparse
from typing import ClassVar

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter
from hitl_pmp.practice_loop import PracticeLoop

from .environment import LightSwitchEnvironment
from .problem import LightSwitchProblem
from .renderer import LightSwitchRenderer
from .tasks import LightSwitchTasks


class LightSwitchCli:
    """Plugs Light Switch into the generic runner (see hitl_pmp/cli.py): exposes
    its configurable ClassVars as argparse flags, applied by whichever --method
    is chosen (e.g. methods/oracle/cli.py's SkillOracleCli) before driving that
    method through PracticeLoop -- the one execution harness every core.Method
    runs through, so there's no separate run() loop here anymore. A
    static-method container, never instantiated, same as every other
    business-logic class in this project."""

    render_fps: ClassVar[int] = 2  # slow -- episodes are only a few actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--method/--output-dir are global flags
        added by hitl_pmp/cli.py, not here -- everything below is specific to
        this domain."""
        parser.add_argument(
            "--grid-size",
            type=int,
            default=LightSwitchEnvironment.grid_size,
            help="Number of cells in the Light Switch grid.",
        )
        parser.add_argument(
            "--light-on-tolerance",
            type=float,
            default=LightSwitchEnvironment.light_on_tolerance,
            help="Max |level - target| for the LightOn predicate to hold.",
        )
        parser.add_argument(
            "--same-position-tolerance",
            type=float,
            default=LightSwitchEnvironment.same_position_tolerance,
            help='Max |robot.x - light.x| to count as "at the light".',
        )
        parser.add_argument(
            "--canonical-light-target",
            type=float,
            default=LightSwitchEnvironment.canonical_light_target,
            help="light_target used by Environment.hard_reset (not task sampling).",
        )
        parser.add_argument(
            "--test-env-seed-offset",
            type=int,
            default=LightSwitchTasks.test_env_seed_offset,
            help="Offset added to --seed to derive the test RNG stream.",
        )
        parser.add_argument(
            "--target-low",
            type=float,
            default=LightSwitchTasks.target_low,
            help="Lower bound of the light's sampled target (Uniform[target_low, target_high)).",
        )
        parser.add_argument(
            "--target-high",
            type=float,
            default=LightSwitchTasks.target_high,
            help="Upper bound of the light's sampled target.",
        )

    @staticmethod
    def apply_config(*, args: argparse.Namespace) -> None:
        LightSwitchEnvironment.grid_size = args.grid_size
        LightSwitchEnvironment.light_on_tolerance = args.light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = args.same_position_tolerance
        LightSwitchEnvironment.canonical_light_target = args.canonical_light_target
        LightSwitchTasks.test_env_seed_offset = args.test_env_seed_offset
        LightSwitchTasks.target_low = args.target_low
        LightSwitchTasks.target_high = args.target_high
        # Must run last: rederives train_rng/test_rng from seed + (possibly
        # just-updated) test_env_seed_offset.
        LightSwitchTasks.set_seed(seed=args.seed)

    @staticmethod
    def run_method(*, args: argparse.Namespace, method: type[Method]) -> None:
        """Shared by every Light-Switch method-CLI (methods/oracle/cli.py's
        SkillOracleCli, and eventually a Random Skills one): applies config, wires Problem.env/
        Problem.tasks (needed since PracticeLoop only
        ever calls the problem argument's *inherited* facade methods, which read
        Problem.env/Problem.tasks off the base class by name -- see
        practice_loop.py's own docstring), drives method through PracticeLoop,
        prints a success-rate summary, and writes episode.mp4 if --output-dir is
        set. Metrics.reset() first, so a run's own evaluations() don't leak from
        (or into) whatever an earlier run in the same process left behind."""
        LightSwitchCli.apply_config(args=args)
        Problem.env = LightSwitchEnvironment
        Problem.tasks = LightSwitchTasks
        Metrics.reset()

        renderer: type[Renderer] | None = (
            LightSwitchRenderer if args.output_dir is not None else None
        )
        frames = PracticeLoop.run(
            problem=LightSwitchProblem,
            method=method,
            metrics=Metrics,
            num_cycles=0,  # an oracle never practices/learns -- one evaluation sweep only
            max_steps_per_interaction=0,  # unused: never reached with num_cycles=0
            num_test_tasks=args.num_test_tasks,
            renderer=renderer,
        )
        _num_online_transitions, num_solved, num_total = Metrics.evaluations[0]
        print(f"success rate: {num_solved}/{num_total} ({num_solved / num_total:.0%})")

        if args.output_dir is not None:
            VideoWriter.write(
                frames=frames,
                output_path=args.output_dir / "episode.mp4",
                fps=LightSwitchCli.render_fps,
            )
