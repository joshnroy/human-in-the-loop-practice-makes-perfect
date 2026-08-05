from pathlib import Path

import imageio
import numpy as np
import pytest

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.renderer.renderer import Renderer, VideoStream, VideoWriter

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


class _DummyEnv(Environment):
    """render_frame's signature requires an env instance now -- unused here, this
    domain's renderer just doesn't happen to need anything from it."""

    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def hard_reset(self) -> None:
        raise NotImplementedError


class _DummyRenderer(Renderer):
    @staticmethod
    def render_frame(*, state: State, env: Environment, label: str | None = None) -> np.ndarray:
        del env, label
        value = int(np.clip(state[_OBJ][0], 0, 255))
        return np.full((2, 2, 3), value, dtype=np.uint8)


def test_renderer_declares_expected_abstract_methods() -> None:
    assert Renderer.__abstractmethods__ == frozenset({"render_frame"})


def test_renderer_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Renderer()  # type: ignore[abstract]


def test_dummy_renderer_reflects_state_in_the_frame() -> None:
    state = State(data={_OBJ: np.array([7.0])})
    frame = _DummyRenderer.render_frame(state=state, env=_DummyEnv())
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


def test_video_writer_write_gif_converts_an_existing_video(*, tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    VideoWriter.write(frames=_make_solid_frames(size=32, count=4), output_path=video_path, fps=5)
    gif_path = tmp_path / "clip.gif"

    VideoWriter.write_gif(video_path=video_path, gif_path=gif_path, fps=5)

    assert gif_path.exists()
    decoded = [np.asarray(frame) for frame in imageio.mimread(gif_path)]
    assert len(decoded) == 4


def test_video_writer_write_gif_creates_missing_parent_directories(*, tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    VideoWriter.write(frames=_make_solid_frames(size=8, count=2), output_path=video_path, fps=5)
    gif_path = tmp_path / "nested" / "dir" / "clip.gif"

    VideoWriter.write_gif(video_path=video_path, gif_path=gif_path, fps=5)

    assert gif_path.exists()


def test_video_writer_write_gif_falls_back_to_ffmpeg_subprocess_on_imageio_failure(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If imageio's own video-reading path fails for any reason, write_gif should
    still succeed by shelling out to ffmpeg directly, not propagate the failure."""
    video_path = tmp_path / "clip.mp4"
    VideoWriter.write(frames=_make_solid_frames(size=16, count=3), output_path=video_path, fps=5)
    gif_path = tmp_path / "clip.gif"

    def _broken_get_reader(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated imageio failure")

    monkeypatch.setattr(imageio, "get_reader", _broken_get_reader)

    VideoWriter.write_gif(video_path=video_path, gif_path=gif_path, fps=5)

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_video_stream_writes_every_appended_frame(*, tmp_path: Path) -> None:
    """The streaming counterpart of VideoWriter.write: frames go to disk as they
    are produced, so nothing has to hold a whole run's worth of them."""
    stream = VideoStream(output_path=tmp_path / "stream.mp4", fps=5)
    for frame in _make_solid_frames(size=32, count=4):
        stream.append(frame=frame)
    stream.close()

    decoded = [np.asarray(frame) for frame in imageio.mimread(tmp_path / "stream.mp4")]
    assert len(decoded) == 4
    assert decoded[-1].mean() - decoded[0].mean() > 100


def test_video_stream_counts_the_frames_it_wrote(*, tmp_path: Path) -> None:
    stream = VideoStream(output_path=tmp_path / "stream.mp4", fps=5)
    for frame in _make_solid_frames(size=16, count=3):
        stream.append(frame=frame)
    stream.close()
    assert stream.frames_written == 3


def test_video_stream_creates_missing_parent_directories(*, tmp_path: Path) -> None:
    stream = VideoStream(output_path=tmp_path / "nested" / "dir" / "stream.mp4", fps=5)
    stream.append(frame=np.zeros((16, 16, 3), dtype=np.uint8))
    stream.close()
    assert (tmp_path / "nested" / "dir" / "stream.mp4").exists()


def test_video_stream_writes_no_file_when_nothing_was_appended(*, tmp_path: Path) -> None:
    """A run that recorded nothing should leave no truncated, unplayable file
    behind -- the encoder is opened lazily, on the first frame."""
    stream = VideoStream(output_path=tmp_path / "stream.mp4", fps=5)
    stream.close()
    assert not (tmp_path / "stream.mp4").exists()


def test_video_stream_close_is_idempotent(*, tmp_path: Path) -> None:
    stream = VideoStream(output_path=tmp_path / "stream.mp4", fps=5)
    stream.append(frame=np.zeros((16, 16, 3), dtype=np.uint8))
    stream.close()
    stream.close()
    assert (tmp_path / "stream.mp4").exists()
