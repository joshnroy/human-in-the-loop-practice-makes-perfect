from pathlib import Path

import imageio
import numpy as np
import pytest

from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate, Task
from hitl_pmp.core.renderer.renderer import EpisodeRenderer, Renderer, VideoWriter

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)
_AT_LEAST_TWO = Predicate(
    name="at-least-two", types=(_BLOCK,), holds=lambda state, objects: state[objects[0]][0] >= 2.0
)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


def _goal_at_least_two() -> Goal:
    atom = GroundAtom(predicate=_AT_LEAST_TWO, objects=(_OBJ,))
    return Goal(atoms=frozenset({atom}))


class _DummyRenderer(Renderer):
    @staticmethod
    def render_frame(*, state: State) -> np.ndarray:
        value = int(np.clip(state[_OBJ][0], 0, 255))
        return np.full((2, 2, 3), value, dtype=np.uint8)


class _DummyEnv(Environment):
    @staticmethod
    def take_action(*, action: Action) -> State:
        Environment.current_state = _state(x=float(action[0]))
        return Environment.current_state

    @staticmethod
    def get_valid_actions() -> list[Action]:
        return []

    @staticmethod
    def hard_reset() -> None:
        Environment.set_state(state=_state(x=0.0))


class _DummyProblem(Problem):
    env = _DummyEnv

    @staticmethod
    def run_task_episode(*, task: Task, policy: Policy) -> bool:
        raise NotImplementedError


_increment_policy: Policy = lambda state: np.array([state[_OBJ][0] + 1.0])  # noqa: E731


def test_renderer_declares_expected_abstract_methods() -> None:
    assert Renderer.__abstractmethods__ == frozenset({"render_frame"})


def test_renderer_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Renderer()  # type: ignore[abstract]


def test_episode_renderer_captures_the_initial_frame_before_any_action() -> None:
    task = Task(initial_state=_state(x=0.0), goal=_goal_at_least_two())
    frames = EpisodeRenderer.record(
        problem=_DummyProblem,
        renderer=_DummyRenderer,
        task=task,
        policy=_increment_policy,
        max_steps=5,
    )
    assert frames[0].shape == (2, 2, 3)
    assert frames[0][0, 0, 0] == 0


def test_episode_renderer_stops_capturing_once_goal_is_satisfied() -> None:
    task = Task(initial_state=_state(x=0.0), goal=_goal_at_least_two())
    frames = EpisodeRenderer.record(
        problem=_DummyProblem,
        renderer=_DummyRenderer,
        task=task,
        policy=_increment_policy,
        max_steps=5,
    )
    # x=0 (initial) -> x=1 -> x=2 (satisfies goal, loop breaks before a 4th action)
    assert len(frames) == 3
    assert [frame[0, 0, 0] for frame in frames] == [0, 1, 2]


def test_episode_renderer_respects_max_steps() -> None:
    task = Task(initial_state=_state(x=0.0), goal=_goal_at_least_two())
    frames = EpisodeRenderer.record(
        problem=_DummyProblem,
        renderer=_DummyRenderer,
        task=task,
        policy=_increment_policy,
        max_steps=1,
    )
    # Only 1 action allowed: x=0 (initial) -> x=1 (goal still unsatisfied, loop ends)
    assert len(frames) == 2


def _make_solid_frames(*, size: int, count: int) -> list[np.ndarray]:
    """count frames, each a solid color that clearly differs frame-to-frame, so a
    round-trip decode can confirm real content survived (not just file existence)."""
    return [
        np.full((size, size, 3), int(255 * i / (count - 1)), dtype=np.uint8) for i in range(count)
    ]


def test_video_writer_mp4_preserves_frame_count_and_content(*, tmp_path: Path) -> None:
    frames = _make_solid_frames(size=32, count=4)
    output_path = tmp_path / "clip.mp4"
    VideoWriter.write(frames=frames, output_path=output_path, fps=5)

    decoded = [np.asarray(frame) for frame in imageio.mimread(output_path)]
    assert len(decoded) == len(frames)
    # h264 is lossy, so compare mean brightness (not exact pixels) between the first
    # and last decoded frame -- confirms real per-frame content survived the encode,
    # not just a repeated/truncated single frame.
    assert decoded[-1].mean() - decoded[0].mean() > 100


def test_video_writer_writes_a_single_frame_mp4(*, tmp_path: Path) -> None:
    """EpisodeRenderer.record can legitimately return just 1 frame (goal already
    satisfied at t=0, no actions taken) -- confirm this path is handled."""
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    output_path = tmp_path / "clip.mp4"
    VideoWriter.write(frames=frames, output_path=output_path, fps=5)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_video_writer_creates_missing_parent_directories(*, tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dir" / "clip.mp4"
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    VideoWriter.write(frames=frames, output_path=output_path, fps=5)
    assert output_path.exists()
