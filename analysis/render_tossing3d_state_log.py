"""Re-renders a pre-recording Tossing3D `StateLogWriter` log into an mp4, with no live
replay of the run that produced it -- see `environments/tossing3d/state_log.py`'s
module docstring for why a flat `core.State` alone cannot do this.

The log is loaded as run-recording entries (`LegacyTossing3DStateLog`) and drawn by the
same `RecordingReplayer` that plays new `recording.jsonl` files
(`analysis/render_recording.py`), at 1x and in the run's own scene. Those logs carry no
phases or outcomes, so every skill is labelled as practice.

    scripts/with_env.sh python analysis/render_tossing3d_state_log.py \
        --state-log /tmp/run/state_log.jsonl --output-video /tmp/run/replayed.mp4
"""

import argparse
from pathlib import Path

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.tossing3d.recording import (
    LegacyTossing3DStateLog,
    Tossing3DReplayRenderer,
)
from hitl_pmp.recording.replay.replayer import RecordingReplayer


def render_state_log(*, state_log_path: Path, output_video_path: Path) -> int:
    """Replays one state log into a video, returning the frame count written."""
    header, entries = LegacyTossing3DStateLog.load(path=state_log_path)
    renderer = Tossing3DReplayRenderer(scene=header.scene, scene_bg=None)
    video = VideoStream(output_path=output_video_path, fps=renderer.fps())
    try:
        RecordingReplayer(renderer=renderer, practice_speedup=1).play(
            header=header, entries=entries, video=video
        )
    finally:
        video.close()
        renderer.close()
    return video.frames_written


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-log", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    args = parser.parse_args()
    frames_written = render_state_log(
        state_log_path=args.state_log, output_video_path=args.output_video
    )
    print(f"wrote {args.output_video} ({frames_written} frames)")


if __name__ == "__main__":
    _main()
