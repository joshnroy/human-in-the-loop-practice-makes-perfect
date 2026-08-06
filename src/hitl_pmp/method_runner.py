import argparse
import os
import shutil
from pathlib import Path
from typing import ClassVar

import numpy as np

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream, VideoWriter
from hitl_pmp.practice_loop import PracticeLoop
from hitl_pmp.recording.loop_recorder import LoopRecorder


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

    # --record-full-loop covers a whole run (every practice step and every
    # evaluation episode of every sweep), so it is far longer than the one-episode
    # clips render_fps is tuned for and would be tedious to scrub at that rate.
    # Reset markers are held for a fixed number of *frames*, so they stay visible
    # for a readable fraction of a second here rather than an awkward pause.
    full_loop_fps: ClassVar[int] = 4

    @staticmethod
    def rendering_needed(*, args: argparse.Namespace) -> bool:
        """Whether anything in this run will consume rendered frames, which is what
        decides whether an environment hands its Renderer over at all. Shared so the
        condition cannot drift between domains as consumers are added -- it grew a
        second one (--record-full-loop) the moment there was more than --output-dir."""
        return (
            getattr(args, "output_dir", None) is not None
            or getattr(args, "record_full_loop", None) is not None
        )

    @staticmethod
    def run(
        *,
        args: argparse.Namespace,
        method: Method,
        problem: Problem,
        # Next to `problem`, which is the one it is the counterpart of -- see
        # PracticeLoop's "separate evaluation environment" section.
        evaluation_problem: Problem | None = None,
        num_cycles: int,
        max_steps_per_interaction: int,
        renderer: type[Renderer] | None,
        render_fps: int,
        num_render_checkpoints: int = 1,
    ) -> Metrics:
        metrics = Metrics()
        # Written as each sweep finishes rather than collected and written at the
        # end: holding every frame of every checkpoint until the run completes is
        # an unbounded buffer (checkpoints x episode length x frame bytes), and a
        # runaway one already OOM-killed a whole session on this project. Only the
        # path is remembered between sweeps, never the pixels.
        written_clips: dict[int, Path] = {}

        def write_clip(*, transitions: int, frames: list[np.ndarray]) -> None:
            # One clip per rendered checkpoint, named by the training progress it
            # depicts, so a set of them reads as a progression.
            if args.output_dir is None:
                return
            output_path = args.output_dir / f"episode_{transitions:06d}.mp4"
            VideoWriter.write(frames=frames, output_path=output_path, fps=render_fps)
            written_clips[transitions] = output_path

        # A run whose planner never succeeds and a run whose planner works but plans
        # badly produce the same stats.json otherwise -- 0/N either way. Recorded as
        # per-window deltas of the Method's own cumulative counters, so a Method keeps
        # two monotonic numbers and this decides the cadence. See
        # Metrics.record_planning_outcomes for exactly what one window covers.
        #
        # Seeded from the Method's *current* reading rather than from 0, so a Method
        # instance reused across two runs cannot silently dump run 1's whole total into
        # run 2's first bucket. Nothing does that today (every CLI builds a fresh
        # Method), and the failure would be a positive delta, so no validation would
        # catch it.
        failures_recorded, attempts_recorded = method.planning_outcomes()

        def record_planning_outcomes() -> None:
            nonlocal failures_recorded, attempts_recorded
            failures, attempts = method.planning_outcomes()
            metrics.record_planning_outcomes(
                num_failures=failures - failures_recorded,
                num_attempts=attempts - attempts_recorded,
            )
            failures_recorded, attempts_recorded = failures, attempts

        # The same treatment for practice, and for the same class of failure one level
        # down: a run whose samplers were never given enough labels and a run whose
        # samplers cannot fit the labels they have score identically. Cumulative
        # readings differenced per window, seeded from the Method's current reading for
        # the reused-instance reason above. See Metrics.record_practice_outcomes.
        practice_recorded = method.practice_outcomes()

        def record_practice_outcomes() -> None:
            nonlocal practice_recorded
            current = method.practice_outcomes()
            # Skills absent from the previous reading are new this window, so their
            # whole tally is the delta -- an empty tally is the right `previous`. The
            # reverse (a skill that disappears) would mean a counter went backwards,
            # which SkillPracticeTally.minus rejects rather than hides; it cannot
            # happen here because a skill that vanished from `current` is simply not
            # iterated, and nothing else consults the stale entry.
            metrics.record_practice_outcomes(
                outcomes={
                    name: tally.minus(previous=practice_recorded.get(name, SkillPracticeTally()))
                    for name, tally in current.items()
                }
            )
            practice_recorded = current

        def record_cycle_end() -> None:
            record_planning_outcomes()
            record_practice_outcomes()

        recorder = MethodRunner._build_recorder(
            args=args,
            problem=problem,
            renderer=renderer,
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
        )
        try:
            PracticeLoop.run(
                problem=problem,
                # None for every domain that has not been migrated, which is what
                # keeps their results byte-identical -- PracticeLoop then evaluates
                # on `problem`, exactly as before. See its own docstring.
                evaluation_problem=evaluation_problem,
                method=method,
                metrics=metrics,
                num_cycles=num_cycles,
                max_steps_per_interaction=max_steps_per_interaction,
                num_test_tasks=args.num_test_tasks,
                # Read off args rather than threaded through every environment-CLI's
                # run_method, exactly like num_render_checkpoints above it: both are
                # harness knobs owned by cli.py's global flags, not per-domain or
                # per-method configuration.
                practice_reset_interval=getattr(args, "practice_reset_interval", None),
                # Checkpoint clips are an --output-dir product; without one there is
                # nowhere to write them, so nothing is rendered for them either. The
                # recorder carries its own renderer, so --record-full-loop still
                # records with no --output-dir at all.
                renderer=renderer if getattr(args, "output_dir", None) is not None else None,
                num_render_checkpoints=num_render_checkpoints,
                on_checkpoint_frames=write_clip,
                on_cycle_end=record_cycle_end,
                recorder=recorder,
            )
            # Once more after the loop, so the final evaluation sweep is covered and
            # so a num_cycles=0 run (every non-learning baseline) still gets exactly
            # one entry rather than none. With num_cycles=N that leaves N+1 entries,
            # the same length as `evaluations` -- though offset from it, since
            # on_cycle_end fires before each sweep rather than after it. That offset
            # is spelled out in Metrics.record_planning_outcomes and pinned by
            # tests/test_method_runner.py.
            record_cycle_end()
        finally:
            # In a finally so a crashed run still leaves a playable video of
            # everything up to the crash -- which is exactly when someone wants to
            # watch what the loop was doing.
            if recorder is not None:
                recorder.close()
        if recorder is not None:
            print(
                f"full-loop recording: {args.record_full_loop} ({recorder.frames_written} frames)"
            )
        # The LAST evaluation, not the first: with num_cycles=0 (every non-learning
        # baseline) there is exactly one sweep so the two coincide, but for a
        # learning Method the first sweep runs *before* any practice, so reporting
        # it would always print the untrained score and hide the whole result.
        _num_online_transitions, num_solved, num_total = metrics.evaluations[-1]
        print(f"success rate: {num_solved}/{num_total} ({num_solved / num_total:.0%})")
        # Printed beside the score, because the whole point of counting planning
        # failures is that they should be visible *without* opening stats.json -- a
        # malformed-PDDL defect once produced a clean-looking 0/5 that took an hour to
        # diagnose. Only when the planner failed at all, so a healthy non-planning run
        # prints exactly what it printed before.
        num_failed, num_attempted = metrics.total_planning_outcomes()
        if num_failed:
            print(f"planning failures: {num_failed}/{num_attempted} planner calls found no plan")
        # One line per lifted skill, for the same reason: the question these counters
        # exist to answer -- was the sampler starved, or is it unable -- gets asked
        # while watching a run, not only afterwards. x/y throughout and never a bare
        # percentage, since a rate over three attempts and a rate over three hundred
        # support very different claims. Printed only by a Method that measures
        # practice, so a non-learning baseline's output is exactly what it was before.
        # The last two are printed apart rather than as one "fallback" number because
        # they call for opposite fixes and are otherwise indistinguishable here: a
        # `param_dim == 0` skill can never be improved by practice at all, while a
        # sampler that was consulted and could not discriminate points at the success
        # predicate. See SamplerConsultation.
        for skill_name, tally in metrics.total_practice_outcomes().items():
            print(
                f"practice {skill_name}: {tally.num_successes}/{tally.num_attempts} succeeded "
                f"({tally.num_informed_successes}/{tally.num_informed_attempts} informed, "
                f"{tally.num_random_successes}/{tally.num_random_attempts} epsilon-random, "
                f"{tally.num_unparameterized_successes}/{tally.num_unparameterized_attempts} "
                f"no sampler, "
                f"{tally.num_uninformative_successes()}/{tally.num_uninformative_attempts()} "
                f"uninformative)"
            )

        if args.output_dir is not None:
            if written_clips:
                # The final checkpoint is additionally published as plain
                # episode.mp4 -- the single-clip name callers and docs already
                # refer to. Copied from the file just written rather than
                # re-encoded from retained frames, which is the whole point: the
                # frames are long gone by now.
                shutil.copyfile(written_clips[max(written_clips)], args.output_dir / "episode.mp4")
            (args.output_dir / "stats.json").write_text(metrics.model_dump_json(indent=2))
            # Written alongside rather than inside stats.json: stats.json is the
            # serialized Metrics, every archived run conforms to that shape, and its
            # byte-stability is what verifies a change did not alter results -- a
            # commit SHA in it would break that on every commit. Same rule as
            # timing.json. Collected after the run so a crash mid-run costs no
            # results, and from `args` post-argparse so defaulted flags land too.
            snapshot = ConfigSnapshot.collect(
                args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH")
            )
            (args.output_dir / "config_snapshot.json").write_text(
                snapshot.model_dump_json(indent=2)
            )
        return metrics

    @staticmethod
    def _build_recorder(
        *,
        args: argparse.Namespace,
        problem: Problem,
        renderer: type[Renderer] | None,
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> LoopRecorder | None:
        """None unless --record-full-loop asked for one, which is what keeps every
        existing run byte-identical."""
        output_path = getattr(args, "record_full_loop", None)
        if output_path is None:
            return None
        if renderer is None:
            # Up front, before the run: a domain with no renderer.py (Ball-Ring)
            # would otherwise run to completion and silently write nothing.
            raise ValueError(
                "--record-full-loop needs a renderer, and the selected environment "
                "does not provide one (no renderer.py for this domain yet)."
            )
        return LoopRecorder(
            renderer=renderer,
            env=problem.env,
            video=VideoStream(output_path=output_path, fps=MethodRunner.full_loop_fps),
            num_cycles=num_cycles,
            max_steps_per_interaction=max_steps_per_interaction,
        )
