"""`--defer-rendering` is a global flag: every environment that renders honours it, and it
changes nothing but whether video is written.

It used to be defined on `--method ees` alone and read by Tossing3D alone, so the CLI tests
that drive `random-skills` or `skill-oracle` had no way to skip video encoding they never
look at. stats.json byte-stability is how this repo verifies a change did not alter
results, so the flag is pinned against it directly.
"""

import argparse
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossingroom.cli import TossingRoomCli
from hitl_pmp.method_runner import MethodRunner

FAST_RUNS = {
    "lightswitch": ["--grid-size", "1", "--num-test-tasks", "2"],
    "tossingroom": [
        "--num-test-tasks",
        "2",
        "--num-cycles",
        "1",
        "--max-steps-per-interaction",
        "5",
    ],
}


class DeferRenderingHarness:
    """A static-method container, never instantiated."""

    @staticmethod
    def run(*, env_name: str, output_dir: Path, extra: list[str]) -> None:
        Cli.main(
            argv=[
                "--env",
                env_name,
                "--method",
                "random-skills",
                *FAST_RUNS[env_name],
                "--seed",
                "3",
                "--output-dir",
                str(output_dir),
                *extra,
            ]
        )


@pytest.mark.parametrize("method_name", ["random-skills", "skill-oracle", "ees", "pomdp"])
@pytest.mark.parametrize(("argv", "expected"), [([], False), (["--defer-rendering"], True)])
def test_every_method_parses_the_flag(*, method_name: str, argv: list[str], expected: bool) -> None:
    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", method_name, *argv])
    assert args.defer_rendering is expected


@pytest.mark.parametrize(
    ("env_name", "env_cli"),
    [("lightswitch", LightSwitchCli), ("tossingroom", TossingRoomCli), ("tossing3d", Tossing3DCli)],
)
@pytest.mark.parametrize(("argv", "renders"), [([], True), (["--defer-rendering"], False)])
def test_every_rendering_environment_builds_no_renderer_when_deferred(
    *,
    env_name: str,
    env_cli: type,
    argv: list[str],
    renders: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def _run(*, renderer: object, **_kwargs: object) -> None:
        seen.append(renderer)

    monkeypatch.setattr(MethodRunner, "run", staticmethod(_run))
    if env_cli is Tossing3DCli:
        monkeypatch.setattr(Tossing3DCli, "resolve_render_fps", staticmethod(lambda **_: 1))
    Cli.main(
        argv=[
            "--env",
            env_name,
            "--method",
            "skill-oracle",
            "--output-dir",
            str(tmp_path),
            *argv,
        ]
    )
    assert len(seen) == 1
    assert (seen[0] is not None) is renders


@pytest.mark.parametrize("env_name", sorted(FAST_RUNS))
def test_a_deferred_run_writes_no_video_and_an_identical_stats_json(
    *, env_name: str, tmp_path: Path
) -> None:
    rendered, deferred = tmp_path / "rendered", tmp_path / "deferred"
    DeferRenderingHarness.run(env_name=env_name, output_dir=rendered, extra=[])
    DeferRenderingHarness.run(env_name=env_name, output_dir=deferred, extra=["--defer-rendering"])

    assert list(rendered.rglob("*.mp4")), "the control run must render, or this proves nothing"
    assert not list(deferred.rglob("*.mp4"))
    assert (rendered / "stats.json").read_bytes() == (deferred / "stats.json").read_bytes()


def test_the_flag_is_recorded_in_the_resolved_namespace() -> None:
    args = Cli.parse_args(argv=["--env", "lightswitch", "--method", "random-skills"])
    assert isinstance(args, argparse.Namespace)
    assert "defer_rendering" in vars(args)
