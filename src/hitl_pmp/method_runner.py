import argparse

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter
from hitl_pmp.practice_loop import PracticeLoop


class MethodRunner:
    """The domain- and method-agnostic tail of driving a core.Method through
    PracticeLoop from the CLI: constructs a fresh Metrics(), runs PracticeLoop,
    prints a success-rate summary, and writes episode.mp4 if --output-dir is set.
    Every domain's own <domain>Cli.run_method constructs problem/method (the
    genuinely domain-specific step -- building the actual Environment/Tasks/Method
    instances) and every method-CLI decides num_cycles/max_steps_per_interaction
    (an oracle passes 0/0 since it never practices; a learning method would read
    these from its own CLI flags) -- both then delegate here, so this logic is
    written once regardless of how many environments/methods exist, instead of
    being copy-pasted into every domain's cli.py. Constructing a fresh Metrics()
    per call (rather than taking one as a parameter) means there's no reset()
    step for a caller to remember -- unlike the old shared-ClassVar Metrics,
    there's nothing left over from a previous run to reset in the first place.
    Returns the constructed Metrics so a caller (or a test) can inspect what
    actually happened beyond the printed summary -- e.g. confirming num_cycles
    was genuinely forwarded to PracticeLoop, not hardcoded -- and so a future
    --output-dir/stats.json feature has something to serialize. A static-method
    container, never instantiated, same as every other business-logic class in
    this project."""

    @staticmethod
    def run(
        *,
        args: argparse.Namespace,
        method: Method,
        problem: Problem,
        num_cycles: int,
        max_steps_per_interaction: int,
        renderer: type[Renderer] | None,
        render_fps: int,
    ) -> Metrics:
        metrics = Metrics()
        frames = PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            num_test_tasks=args.num_test_tasks,
            renderer=renderer,
        )
        _num_online_transitions, num_solved, num_total = metrics.evaluations[0]
        print(f"success rate: {num_solved}/{num_total} ({num_solved / num_total:.0%})")

        if args.output_dir is not None:
            VideoWriter.write(
                frames=frames,
                output_path=args.output_dir / "episode.mp4",
                fps=render_fps,
            )
        return metrics
