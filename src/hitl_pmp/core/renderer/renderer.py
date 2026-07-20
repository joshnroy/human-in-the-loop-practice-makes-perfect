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
