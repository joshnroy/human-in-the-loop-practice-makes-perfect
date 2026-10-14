"""Render a run's recording into a presentation video, playing back recorded states.

No physics is stepped: every frame is a recorded object-centric state restored into a
drawing scene, so the video shows what the run executed, not a re-simulation of it.
Practice plays at 5x, each human skill is a one-second labelled pause, and the final
evaluation checkpoint (or `--checkpoint N`) plays at 1x with each task's number and
outcome::

    scripts/with_env.sh python analysis/render_recording.py \\
        --run <run-dir> --output <run-dir>/replay.mp4

`--full-agentic` reads a full-agentic run's own `practice.jsonl` and
`evaluations/NNNN/task-XX.jsonl` instead of `recording.jsonl` files.
"""

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.recording.replay.renderer import ReplayRenderer
from hitl_pmp.recording.replay.replayer import RecordingReplayer
from hitl_pmp.recording.replay.types import ReplaySummary
from hitl_pmp.recording.run_record.full_agentic_log import FullAgenticRunLog
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import (
    RECORDING_FILENAME,
    RecordEntry,
    RecordingHeader,
)

Source = tuple[RecordingHeader, Iterator[RecordEntry]]


class RecordingVideo:
    @staticmethod
    def sources(*, run: Path, checkpoint: int | None, full_agentic: bool) -> list[Source]:
        """Practice, then one evaluation checkpoint (the last one, unless chosen)."""
        if full_agentic:
            header = FullAgenticRunLog.header(run=run)
            checkpoints = FullAgenticRunLog.checkpoints(run=run)
            chosen = checkpoint if checkpoint is not None else max(checkpoints, default=None)
            sources: list[Source] = [(header, FullAgenticRunLog.practice_entries(run=run))]
            if chosen is not None:
                sources.append((
                    header,
                    FullAgenticRunLog.evaluation_entries(run=run, checkpoint=chosen),
                ))
            return sources
        paths = [run / RECORDING_FILENAME]
        evaluations = sorted((run / "evaluations").glob(f"*/{RECORDING_FILENAME}"))
        if checkpoint is not None:
            paths.append(run / "evaluations" / f"{checkpoint:04d}" / RECORDING_FILENAME)
        elif evaluations:
            paths.append(evaluations[-1])
        return [
            (RunRecordReader.header(path=path), RunRecordReader.entries(path=path))
            for path in paths
        ]

    @staticmethod
    def default_renderer(*, header: RecordingHeader, realistic_background: bool) -> ReplayRenderer:
        if header.environment == "tossing3d":
            from hitl_pmp.environments.tossing3d.recording import Tossing3DReplayRenderer

            return Tossing3DReplayRenderer(
                scene=header.scene, scene_bg=True if realistic_background else None
            )
        raise NotImplementedError(f"no replay renderer exists yet for {header.environment!r}")

    @staticmethod
    def render(
        *,
        run: Path,
        output: Path,
        checkpoint: int | None,
        full_agentic: bool,
        renderer_factory: Callable[[RecordingHeader], ReplayRenderer],
    ) -> list[ReplaySummary]:
        summaries = []
        video: VideoStream | None = None
        renderer: ReplayRenderer | None = None
        scene: dict | None = None
        try:
            for header, entries in RecordingVideo.sources(
                run=run, checkpoint=checkpoint, full_agentic=full_agentic
            ):
                if renderer is None or header.scene != scene:
                    if renderer is not None:
                        renderer.close()
                    renderer, scene = renderer_factory(header), header.scene
                if video is None:
                    video = VideoStream(output_path=output, fps=renderer.fps())
                summaries.append(
                    RecordingReplayer(renderer=renderer).play(
                        header=header, entries=entries, video=video
                    )
                )
        finally:
            if video is not None:
                video.close()
            if renderer is not None:
                renderer.close()
        return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--full-agentic", action="store_true")
    parser.add_argument(
        "--plain-background",
        action="store_true",
        help="Keep the run's own scene; the lab2 room is the default.",
    )
    args = parser.parse_args()
    summaries = RecordingVideo.render(
        run=args.run,
        output=args.output,
        checkpoint=args.checkpoint,
        full_agentic=args.full_agentic,
        renderer_factory=lambda header: RecordingVideo.default_renderer(
            header=header, realistic_background=not args.plain_background
        ),
    )
    frames = sum(s.frames_written for s in summaries)
    shown = sum(s.ticks_shown for s in summaries)
    recorded = sum(s.ticks_recorded for s in summaries)
    print(f"wrote {args.output}: {frames} frames, {shown}/{recorded} recorded states drawn")


if __name__ == "__main__":
    main()
