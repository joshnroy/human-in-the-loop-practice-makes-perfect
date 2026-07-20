import argparse
from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter

from .action_oracle_policy import ACTION_ORACLE_POLICY
from .environment import LightSwitchEnvironment
from .problem import LightSwitchProblem
from .renderer import LightSwitchRenderer
from .skill_oracle_policy import SKILL_ORACLE_POLICY
from .tasks import LightSwitchTasks


class LightSwitchCli:
    """Plugs Light Switch into the generic runner (see hitl_pmp/cli.py): exposes its
    configurable ClassVars as argparse flags and runs a chosen policy over sampled
    test tasks. A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    POLICIES: ClassVar[dict[str, Policy]] = {
        "action-oracle": ACTION_ORACLE_POLICY,
        "skill-oracle": SKILL_ORACLE_POLICY,
    }
    render_fps: ClassVar[int] = 2  # slow -- episodes are only a few actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks are global flags added by hitl_pmp/cli.py,
        not here -- everything below is specific to this domain."""
        parser.add_argument(
            "--policy",
            choices=sorted(LightSwitchCli.POLICIES),
            default="action-oracle",
            help="Which policy to run.",
        )
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
    def run(*, args: argparse.Namespace) -> tuple[int, int]:
        """Applies args as config, runs args.policy over args.num_test_tasks sampled
        test tasks via the one LightSwitchProblem.run_task_episode codepath, prints
        progress, and returns (num_solved, num_test_tasks). If args.output_dir is
        set, that same codepath also records the first task's episode (passing
        LightSwitchRenderer through -- every run is optionally recordable this way,
        not via a separate rendering-only path) and writes it to episode.mp4, plus
        episode.gif too if args.gif is also set; a no-op otherwise. These raw oracle
        policies aren't core.Method/core.Metrics-driven, so there's no stats.json
        here -- that's methods/practice_makes_perfect/cli.py's RandomSkillsCli
        (registered under --method, not --policy)."""
        LightSwitchCli.apply_config(args=args)
        policy = LightSwitchCli.POLICIES[args.policy]
        # No hard_reset() here: run_task_episode below unconditionally overwrites
        # current_state from each task's own initial_state before doing anything
        # else, so a reset beforehand would never be observed.

        num_solved = 0
        recorded_frames: list[np.ndarray] = []
        for i in range(args.num_test_tasks):
            task = LightSwitchTasks.sample_test_task()
            renderer: type[Renderer] | None = (
                LightSwitchRenderer if (i == 0 and args.output_dir is not None) else None
            )
            solved, frames = LightSwitchProblem.run_task_episode(
                task=task, policy=policy, renderer=renderer
            )
            if renderer is not None:
                recorded_frames = frames
            num_solved += int(solved)
            print(f"task {i + 1}/{args.num_test_tasks}: {'solved' if solved else 'FAILED'}")

        print(
            f"success rate: {num_solved}/{args.num_test_tasks} "
            f"({num_solved / args.num_test_tasks:.0%})"
        )

        if args.output_dir is not None:
            video_path = args.output_dir / "episode.mp4"
            VideoWriter.write(
                frames=recorded_frames, output_path=video_path, fps=LightSwitchCli.render_fps
            )
            if args.gif:
                VideoWriter.write_gif(
                    video_path=video_path,
                    gif_path=args.output_dir / "episode.gif",
                    fps=LightSwitchCli.render_fps,
                )

        return num_solved, args.num_test_tasks

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
