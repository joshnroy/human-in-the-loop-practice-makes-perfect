"""Re-renders a Tossing3D `StateLogWriter` log into an mp4, with no live replay of the
run that produced it -- see `environments/tossing3d/state_log.py`'s module docstring
for why a flat `core.State` alone cannot do this and what the log carries instead.

Rebuilds a fresh scene from the log's own header (same seed/config the run used), then
for every logged tick: reconstructs an `ObjectCentricState` from the plain dict,
`restore`s the live simulator into it, and renders one frame. A `SkillEvent` line only
labels the frames that follow it; it produces no frame of its own.

    scripts/with_env.sh python analysis/render_tossing3d_state_log.py \
        --state-log /tmp/run/state_log.jsonl --output-video /tmp/run/replayed.mp4
"""

import argparse
from pathlib import Path

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer
from hitl_pmp.environments.tossing3d.state_log import SkillEvent, StateLogReader, TickEvent


def render_state_log(*, state_log_path: Path, output_video_path: Path) -> int:
    """Replays one state log into a video, returning the frame count written."""
    reader = StateLogReader(path=state_log_path)
    if reader.header is None:
        raise ValueError(f"{state_log_path} has no header line -- not a state log")
    args = argparse.Namespace(
        layout=reader.header.layout,
        variant=reader.header.variant,
        scene_bg=reader.header.scene_bg,
        canonical_seed=reader.header.canonical_seed,
        seed=reader.header.seed,
        test_env_seed_offset=reader.header.test_env_seed_offset,
    )
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    backend = env.backend()
    problem.hard_reset()
    # A live scene, matching whatever train task the logged run's own reset_to_task
    # produced -- restore() below overwrites its pose anyway, so which task is sampled
    # here does not matter, only that a scene exists to restore into.
    problem.reset_to_task(task=problem.sample_train_task())
    video = VideoStream(output_path=output_video_path, fps=backend.render_fps())
    current_label = "initial state"
    try:
        for event in reader.events:
            if isinstance(event, SkillEvent):
                objects_desc = ", ".join(event.objects)
                current_label = f"{event.name}({objects_desc})"
                if event.params:
                    current_label += f", params={[round(p, 2) for p in event.params]}"
                continue
            if isinstance(event, TickEvent):
                state = env.restore_plain_snapshot(
                    plain={k: list(v) for k, v in event.state.items()}
                )
                frame = Tossing3DRenderer.render_frame(state=state, env=env, label=current_label)
                video.append(frame=frame)
    finally:
        video.close()
        env.close()
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
