from pathlib import Path

import imageio
import numpy as np
import pytest

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.renderer.renderer import Renderer, VideoWriter

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


class _DummyRenderer(Renderer):
    @staticmethod
    def render_frame(*, state: State, label: str | None = None) -> np.ndarray:
        del label
        value = int(np.clip(state[_OBJ][0], 0, 255))
        return np.full((2, 2, 3), value, dtype=np.uint8)


def test_renderer_declares_expected_abstract_methods() -> None:
    assert Renderer.__abstractmethods__ == frozenset({"render_frame"})


def test_renderer_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Renderer()  # type: ignore[abstract]


def test_dummy_renderer_reflects_state_in_the_frame() -> None:
    state = State(data={_OBJ: np.array([7.0])})
    frame = _DummyRenderer.render_frame(state=state)
    assert frame.shape == (2, 2, 3)
    assert frame[0, 0, 0] == 7


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
    """A rendered episode can legitimately be just 1 frame (goal already satisfied
    at t=0, no actions taken) -- confirm this path is handled."""
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
