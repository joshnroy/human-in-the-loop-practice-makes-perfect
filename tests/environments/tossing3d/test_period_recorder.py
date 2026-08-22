"""A real (short) run through the actual `hitl_pmp.cli` entrypoint must produce one
video per practice period and one per evaluation sweep, with no extra flag -- see
`method_runner.py`'s `_build_period_recorder` for why this is gated on `--output-dir`
alone, the same way `stats.json`/the Tossing3D state log already are. `random-skills`,
not `skill-oracle`: this domain's only non-learning method never practices at all
(`--num-cycles` defaults to 0), and a practice video is the whole point of this file.
`random-skills` needs no planner/sampler, so a `--num-cycles 1`, two-step period stays
fast."""

import importlib.util
from pathlib import Path

import imageio
import pytest

from hitl_pmp.cli import Cli

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra (`kindergarden` + `kinder_models`); CI never installs it",
)


def _run(*, output_dir: Path) -> None:
    Cli.main(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "random-skills",
            "--seed",
            "5",
            "--num-test-tasks",
            "1",
            "--num-cycles",
            "1",
            "--max-steps-per-interaction",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )


def test_an_ordinary_cli_run_writes_one_video_per_period_and_per_sweep(*, tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _run(output_dir=output_dir)

    practice_video = output_dir / "period_videos" / "practice" / "cycle_0000.mp4"
    baseline_sweep_video = output_dir / "period_videos" / "evaluation" / "sweep_0000.mp4"
    trained_sweep_video = output_dir / "period_videos" / "evaluation" / "sweep_0001.mp4"
    for path in (practice_video, baseline_sweep_video, trained_sweep_video):
        assert path.exists(), f"an ordinary run must write {path} with no extra flag"
        assert path.stat().st_size > 0


def test_the_practice_video_carries_every_substep_frame_not_one_per_skill(
    *, tmp_path: Path
) -> None:
    """The property this whole mechanism exists for on Tossing3D: a practice
    period's video must show each skill's physics ticks, not a four-frame
    storyboard -- see `PeriodRecorder.record_practice_step`'s own docstring."""
    output_dir = tmp_path / "run"
    _run(output_dir=output_dir)

    practice_video = output_dir / "period_videos" / "practice" / "cycle_0000.mp4"
    frame_count = sum(1 for _ in imageio.get_reader(practice_video))
    # The period ran 2 steps plus a period-reset marker (reset_hold_frames=6 held
    # frames); one frame per step alone would be well under 10 -- this domain's
    # controllers run tens to hundreds of ticks each, so genuine substep capture
    # clears that bar by at least an order of magnitude.
    assert frame_count > 50, (
        f"practice video has only {frame_count} frames -- looks like one frame per "
        "skill rather than real substep capture"
    )
