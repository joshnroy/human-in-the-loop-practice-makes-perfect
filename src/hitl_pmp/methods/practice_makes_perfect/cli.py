import argparse
from typing import ClassVar

from hitl_pmp.core.metrics.metrics import MetricsWriter
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.metrics import LightSwitchMetrics
from hitl_pmp.environments.lightswitch.predicates import (
    ADJACENT,
    LIGHT_IN_CELL,
    LIGHT_OFF,
    LIGHT_ON,
    ROBOT_IN_CELL,
)
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks

from .practice_loop import PracticeLoop
from .random_skills_method import RandomSkillsMethod


class RandomSkillsCli:
    """Plugs RandomSkillsMethod into the generic runner (see hitl_pmp/cli.py, under
    --method random-skills): exposes PracticeLoop's own knobs as argparse flags and
    runs Random Skills via PracticeLoop against a LightSwitchProblem +
    LightSwitchMetrics. A static-method container, never instantiated, same as
    every other business-logic class in this project.

    TODO(scale): hardcodes LightSwitchEnvironment/LightSwitchCli.apply_config --
    this is this codebase's only environment so far, so there's no genuine
    env-agnostic way yet to wire a Method's skills/predicates/objects/
    compute_action/sample_params off whichever --env was selected. A second
    environment would need that generalized (e.g. each environment exposing a
    standard "method wiring" descriptor), rather than this direct import."""

    render_fps: ClassVar[int] = 2  # matches LightSwitchCli.render_fps

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """--env/--seed/--num-test-tasks/--output-dir/--gif are global flags added
        by hitl_pmp/cli.py, not here."""
        parser.add_argument(
            "--num-cycles",
            type=int,
            default=10,
            help="Number of PracticeLoop interaction-period + evaluation cycles to run.",
        )
        parser.add_argument(
            "--steps-per-cycle",
            type=int,
            default=20,
            help="Number of environment steps per interaction period.",
        )

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        LightSwitchCli.apply_config(args=args)
        RandomSkillsCli._wire_method()
        RandomSkillsMethod.reset_state(seed=args.seed)
        LightSwitchMetrics.reset()

        renderer: type[Renderer] | None = (
            LightSwitchRenderer if args.output_dir is not None else None
        )
        frames = PracticeLoop.run(
            problem=LightSwitchProblem,
            method=RandomSkillsMethod,
            metrics=LightSwitchMetrics,
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.steps_per_cycle,
            num_test_tasks=args.num_test_tasks,
            renderer=renderer,
        )

        for num_online_transitions, percent_solved in LightSwitchMetrics.task_training_curve():
            print(f"{num_online_transitions} online transitions: {percent_solved:.0%} solved")

        if args.output_dir is not None:
            MetricsWriter.write(
                metrics=LightSwitchMetrics, output_path=args.output_dir / "stats.json"
            )
            video_path = args.output_dir / "episode.mp4"
            VideoWriter.write(frames=frames, output_path=video_path, fps=RandomSkillsCli.render_fps)
            if args.gif:
                VideoWriter.write_gif(
                    video_path=video_path,
                    gif_path=args.output_dir / "episode.gif",
                    fps=RandomSkillsCli.render_fps,
                )

    @staticmethod
    def _wire_method() -> None:
        env = LightSwitchEnvironment
        Problem.env = env
        Problem.tasks = LightSwitchTasks
        RandomSkillsMethod.env = env
        RandomSkillsMethod.tasks = LightSwitchTasks
        RandomSkillsMethod.predicates = (
            ADJACENT,
            LIGHT_IN_CELL,
            LIGHT_OFF,
            LIGHT_ON,
            ROBOT_IN_CELL,
        )
        RandomSkillsMethod.skills = (
            LightSwitchSkills.MOVE_ROBOT,
            LightSwitchSkills.TURN_ON_LIGHT,
            LightSwitchSkills.TURN_OFF_LIGHT,
            LightSwitchSkills.JUMP_TO_LIGHT,
        )
        RandomSkillsMethod.objects = (env.robot, env.light, *env.get_cells())
        RandomSkillsMethod.compute_action = LightSwitchSkills.compute_action
        RandomSkillsMethod.sample_params = LightSwitchSkills.sample_params
