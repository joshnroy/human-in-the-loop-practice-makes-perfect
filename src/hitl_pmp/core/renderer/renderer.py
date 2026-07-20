import abc
from collections.abc import Sequence
from pathlib import Path

import imageio
import numpy as np

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task


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
    """Writes a sequence of RGB frames to a video (.mp4, via imageio's bundled
    ffmpeg) or an animated image (.gif, via imageio's native Pillow-backed writer) --
    the format is chosen by output_path's suffix, entirely domain-agnostic. A
    static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def write(*, frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # imageio's own stubs are stricter than what it actually accepts at runtime
        # (verified: both calls below work correctly against a real Sequence of
        # ndarrays in tests) -- mypy can't match Sequence[np.ndarray] against the
        # overloads' list[_Buffer | _SupportsArray[...] | ...] union.
        if output_path.suffix == ".gif":
            # imageio's Pillow-backed GIF writer deprecated fps= in favor of
            # duration= (milliseconds per frame); the ffmpeg-backed video writers
            # below still take fps= directly.
            imageio.mimsave(output_path, frames, duration=1000 / fps)  # type: ignore[call-overload]
        else:
            imageio.mimsave(output_path, frames, fps=fps)  # type: ignore[call-overload]


class EpisodeRenderer:
    """Runs one task episode while capturing an RGB frame at each step -- mirrors a
    concrete Problem's run_task_episode loop (check the goal, else take_action)
    without touching that core interface, since most callers never need rendering.
    A static-method container, never instantiated."""

    @staticmethod
    def record(
        *,
        problem: type[Problem],
        renderer: type[Renderer],
        task: Task,
        policy: Policy,
        max_steps: int,
    ) -> list[np.ndarray]:
        problem.env.set_state(state=task.initial_state)
        state = problem.env.get_current_state()
        frames = [renderer.render_frame(state=state)]
        for _ in range(max_steps):
            if task.goal.is_satisfied(state=state):
                break
            state = problem.env.take_action(action=policy(state))
            frames.append(renderer.render_frame(state=state))
        return frames
