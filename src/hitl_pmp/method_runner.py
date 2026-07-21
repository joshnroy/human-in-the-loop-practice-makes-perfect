import argparse

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter
from hitl_pmp.practice_loop import PracticeLoop


class MethodRunner:
    """The domain- and method-agnostic tail of driving a core.Method through
    PracticeLoop from the CLI: resets Metrics, runs PracticeLoop, prints a
    success-rate summary, and writes episode.mp4 if --output-dir is set. Every
    domain's own <domain>Cli.run_method wires Problem.env/Problem.tasks (the one
    genuinely domain-specific step, plus applying that domain's own config
    flags) and every method-CLI decides num_cycles/max_steps_per_interaction
    (an oracle passes 0/0 since it never practices; a learning method would
    read these from its own CLI flags) -- both then delegate here, so this
    logic is written once regardless of how many environments/methods exist,
    instead of being copy-pasted into every domain's cli.py (which is what
    LightSwitchCli.run_method did before this was extracted -- see
    core/README.md's dependency-direction section for where this sits in the
    layering). A static-method container, never instantiated, same as every
    other business-logic class in this project."""

    @staticmethod
    def run(
        *,
        args: argparse.Namespace,
        method: type[Method],
        problem: type[Problem],
        num_cycles: int,
        max_steps_per_interaction: int,
        renderer: type[Renderer] | None,
        render_fps: int,
    ) -> None:
        Metrics.reset()
        frames = PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=Metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            num_test_tasks=args.num_test_tasks,
            renderer=renderer,
        )
        _num_online_transitions, num_solved, num_total = Metrics.evaluations[0]
        print(f"success rate: {num_solved}/{num_total} ({num_solved / num_total:.0%})")

        if args.output_dir is not None:
            VideoWriter.write(
                frames=frames,
                output_path=args.output_dir / "episode.mp4",
                fps=render_fps,
            )
