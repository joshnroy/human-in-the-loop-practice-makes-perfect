import argparse

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter
from hitl_pmp.practice_loop import PracticeLoop


class MethodRunner:
    """The domain- and method-agnostic tail of driving a core.Method through
    PracticeLoop from the CLI: constructs a fresh Metrics(), runs PracticeLoop,
    prints a success-rate summary, and writes episode.mp4 and stats.json if
    --output-dir is set. Every domain's own <domain>Cli.run_method constructs
    problem/method (the genuinely domain-specific step -- building the actual
    Environment/Tasks/Method instances) and every method-CLI decides
    num_cycles/max_steps_per_interaction (an oracle passes 0/0 since it never
    practices; a learning method would read these from its own CLI flags) --
    both then delegate here, so this logic is written once regardless of how
    many environments/methods exist, instead of being copy-pasted into every
    domain's cli.py. Constructing a fresh Metrics() per call (rather than
    taking one as a parameter) means there's no reset() step for a caller to
    remember -- unlike the old shared-ClassVar Metrics, there's nothing left
    over from a previous run to reset in the first place. Returns the
    constructed Metrics so a caller (or a test) can inspect what actually
    happened beyond the printed summary -- e.g. confirming num_cycles was
    genuinely forwarded to PracticeLoop, not hardcoded. stats.json holds only
    Metrics' raw fields (evaluations/task_name) via model_dump_json -- any
    reader reconstructs a Metrics via Metrics.model_validate_json(...) and
    calls its own computation methods (task_training_curve(),
    percentage_success_overall_test(), etc.), so there's exactly one place
    those are computed, not two. A static-method container, never
    instantiated, same as every other business-logic class in this
    project."""

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
        num_render_checkpoints: int = 1,
    ) -> Metrics:
        metrics = Metrics()
        frames_by_transitions = PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=metrics,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
            num_test_tasks=args.num_test_tasks,
            renderer=renderer,
            num_render_checkpoints=num_render_checkpoints,
        )
        # The LAST evaluation, not the first: with num_cycles=0 (every non-learning
        # baseline) there is exactly one sweep so the two coincide, but for a
        # learning Method the first sweep runs *before* any practice, so reporting
        # it would always print the untrained score and hide the whole result.
        _num_online_transitions, num_solved, num_total = metrics.evaluations[-1]
        print(f"success rate: {num_solved}/{num_total} ({num_solved / num_total:.0%})")

        if args.output_dir is not None:
            for transitions, frames in sorted(frames_by_transitions.items()):
                # One clip per rendered checkpoint, named by the training progress
                # it depicts, so a set of them reads as a progression. The final
                # one is additionally written as plain episode.mp4 -- the
                # single-clip name callers and docs already refer to.
                VideoWriter.write(
                    frames=frames,
                    output_path=args.output_dir / f"episode_{transitions:06d}.mp4",
                    fps=render_fps,
                )
            if frames_by_transitions:
                VideoWriter.write(
                    frames=frames_by_transitions[max(frames_by_transitions)],
                    output_path=args.output_dir / "episode.mp4",
                    fps=render_fps,
                )
            (args.output_dir / "stats.json").write_text(metrics.model_dump_json(indent=2))
        return metrics
