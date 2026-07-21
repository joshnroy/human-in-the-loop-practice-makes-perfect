import abc
import subprocess
from collections.abc import Sequence
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np

from hitl_pmp.core.problem.environment.types import State


class Renderer(abc.ABC):
    """Renders a State as an RGB frame -- a static-method container, never
    instantiated, same pattern as every other core interface. A pure function of
    State: unlike Environment/HumanOracle/Tasks it doesn't belong to Problem, so it
    isn't nested under problem/."""

    @staticmethod
    @abc.abstractmethod
    def render_frame(*, state: State, label: str | None = None) -> np.ndarray:
        """Returns an HxWx3 uint8 RGB frame. label, when given, is the
        core.method.types.LabeledAction.label of whichever action/skill just
        produced this state -- overlaid on the frame so a rendered episode shows
        what happened at each step. None on an episode's first frame (no action has
        been taken yet to produce a label for)."""
        raise NotImplementedError


class VideoWriter:
    """Writes a sequence of RGB frames to a video file via imageio's bundled ffmpeg
    -- domain-agnostic. write_gif converts an already-written video to a gif:
    prefers imageio itself (its ffmpeg-backed video reader + Pillow-backed gif
    writer, both pure Python, so ffmpeg is still doing the underlying decode work,
    just wrapped by a library instead of a raw subprocess call), falling back to
    shelling out to the `ffmpeg` binary directly (located via imageio_ffmpeg's own
    bundled copy, not relying on ffmpeg being on PATH) only if that import/read
    ever fails. A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def write(*, frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # imageio's own stubs are stricter than what it actually accepts at runtime
        # (verified in tests against a real Sequence of ndarrays) -- mypy can't
        # match Sequence[np.ndarray] against the overload's list[_Buffer |
        # _SupportsArray[...] | ...] union.
        imageio.mimsave(output_path, frames, fps=fps)  # type: ignore[call-overload]

    @staticmethod
    def write_gif(*, video_path: Path, gif_path: Path, fps: int) -> None:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # imageio's stubs type get_reader's result as non-iterable, but it is
            # at runtime (same stub-vs-runtime mismatch noted above for mimsave).
            frames = [frame for frame in imageio.get_reader(video_path)]  # type: ignore[attr-defined]
            # imageio's Pillow-backed gif writer deprecated the fps kwarg in favor
            # of duration in milliseconds per frame (unlike the ffmpeg-backed mp4
            # writer above, which still takes fps) -- 1000 / fps converts between
            # the two.
            imageio.mimsave(gif_path, frames, duration=1000 / fps)  # type: ignore[call-overload]
        except Exception:
            # Deliberately broad: any reason the pure-Python path fails should
            # fall back to a direct ffmpeg invocation, not propagate.
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path), str(gif_path)],
                check=True,
                capture_output=True,
            )
