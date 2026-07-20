import abc
from collections.abc import Sequence
from pathlib import Path

import imageio
import numpy as np

from hitl_pmp.core.problem.environment.types import State


class Renderer(abc.ABC):
    """Renders a State as an RGB frame -- a static-method container, never
    instantiated, same pattern as every other core interface. A pure function of
    State: unlike Environment/HumanOracle/Tasks it doesn't belong to Problem, so it
    isn't nested under problem/."""

    @staticmethod
    @abc.abstractmethod
    def render_frame(*, state: State) -> np.ndarray:
        """Returns an HxWx3 uint8 RGB frame."""
        raise NotImplementedError


class VideoWriter:
    """Writes a sequence of RGB frames to a video file via imageio's bundled ffmpeg
    -- domain-agnostic. No native GIF support: convert an written video with an
    external `ffmpeg` invocation instead of building that into this codebase. A
    static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def write(*, frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # imageio's own stubs are stricter than what it actually accepts at runtime
        # (verified in tests against a real Sequence of ndarrays) -- mypy can't
        # match Sequence[np.ndarray] against the overload's list[_Buffer |
        # _SupportsArray[...] | ...] union.
        imageio.mimsave(output_path, frames, fps=fps)  # type: ignore[call-overload]
