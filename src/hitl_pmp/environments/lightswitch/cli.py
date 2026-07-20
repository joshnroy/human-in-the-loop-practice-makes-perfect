import argparse
import json
from pathlib import Path
from typing import ClassVar

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.renderer.renderer import EpisodeRenderer, VideoWriter

from .environment import LightSwitchEnvironment
from .oracle_policy import ORACLE_POLICY
from .problem import LightSwitchProblem
from .renderer import LightSwitchRenderer
from .tasks import LightSwitchTasks


class LightSwitchCli:
    """Plugs Light Switch into the generic runner (see hitl_pmp/cli.py): exposes its
    configurable ClassVars as argparse flags and runs a chosen policy over sampled
    test tasks. A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    POLICIES: ClassVar[dict[str, Policy]] = {"oracle": ORACLE_POLICY}
    render_fps: ClassVar[int] = 2  # slow -- episodes are only a few actions long

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks are global flags added by hitl_pmp/cli.py,
        not here -- everything below is specific to this domain."""
        parser.add_argument(
            "--policy",
            choices=sorted(LightSwitchCli.POLICIES),
            default="oracle",
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
        test tasks, prints progress, and returns (num_solved, num_test_tasks). If
        args.output_dir is set, also renders one demo episode to episode.mp4 and
        writes stats.json there -- both no-ops otherwise."""
        LightSwitchCli._apply_config(args=args)
        policy = LightSwitchCli.POLICIES[args.policy]
        # No hard_reset() here: run_task_episode below unconditionally overwrites
        # current_state from each task's own initial_state before doing anything
        # else, so a reset beforehand would never be observed.

        num_solved = 0
        per_task_solved: list[bool] = []
        for i in range(args.num_test_tasks):
            task = LightSwitchTasks.sample_test_task()
            solved = LightSwitchProblem.run_task_episode(task=task, policy=policy)
            num_solved += int(solved)
            per_task_solved.append(solved)
            print(f"task {i + 1}/{args.num_test_tasks}: {'solved' if solved else 'FAILED'}")

        print(
            f"success rate: {num_solved}/{args.num_test_tasks} "
            f"({num_solved / args.num_test_tasks:.0%})"
        )

        if args.output_dir is not None:
            # Stats first: they're already fully computed and cheap to write, and
            # shouldn't be lost if the (heavier, ffmpeg-dependent) render step below
            # fails for an unrelated reason.
            LightSwitchCli._write_stats(
                output_dir=args.output_dir,
                policy_name=args.policy,
                num_test_tasks=args.num_test_tasks,
                num_solved=num_solved,
                per_task_solved=per_task_solved,
            )
            LightSwitchCli._render_demo_episode(policy=policy, output_dir=args.output_dir)

        return num_solved, args.num_test_tasks

    @staticmethod
    def _render_demo_episode(*, policy: Policy, output_dir: Path) -> None:
        """One representative episode for visual inspection -- not one video per
        swept test task, which would be wasteful to encode for little added value."""
        task = LightSwitchTasks.sample_test_task()
        frames = EpisodeRenderer.record(
            problem=LightSwitchProblem,
            renderer=LightSwitchRenderer,
            task=task,
            policy=policy,
            max_steps=LightSwitchProblem.max_episode_steps(),
        )
        VideoWriter.write(
            frames=frames, output_path=output_dir / "episode.mp4", fps=LightSwitchCli.render_fps
        )

    @staticmethod
    def _write_stats(
        *,
        output_dir: Path,
        policy_name: str,
        num_test_tasks: int,
        num_solved: int,
        per_task_solved: list[bool],
    ) -> None:
        stats = {
            "env": "lightswitch",
            "policy": policy_name,
            "num_test_tasks": num_test_tasks,
            "num_solved": num_solved,
            "success_rate": num_solved / num_test_tasks,
            "per_task_solved": per_task_solved,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    @staticmethod
    def _apply_config(*, args: argparse.Namespace) -> None:
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
