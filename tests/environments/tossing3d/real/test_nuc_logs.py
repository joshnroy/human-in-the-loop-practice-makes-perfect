"""Copying the skill server's session logs from the NUC beside the run, at every
checkpoint, in the background. A local directory stands in for the NUC."""

import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real.nuc_logs import NucLogs
from hitl_pmp.environments.tossing3d.real.session_log import RealSessionLog

pytestmark = pytest.mark.xdist_group("tossing3d_real_nuc_logs")


def _remote(*, root: Path) -> Path:
    """`~/prpl-tidybot/logs/<date>/<time>/`, two sessions, the newer one in use."""
    for session in ("2026-10-20/10-00-00", "2026-10-21/09-30-00"):
        directory = root / session
        (directory / "localization_failures").mkdir(parents=True)
        (directory / "skill_calls.jsonl").write_text(json.dumps({"session": session}) + "\n")
        (directory / "skill_server.log").write_text("started\n")
    return root


@pytest.fixture
def session_log(*, tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    RealSessionLog.configure(output_dir=output)
    yield output
    RealSessionLog.close()


def test_the_newest_session_is_copied_beside_the_run(*, session_log: Path, tmp_path: Path) -> None:
    remote = _remote(root=tmp_path / "logs")
    NucLogs.pull(host=None, path=str(remote), output=session_log, wait=True)
    copied = session_log / "nuc_logs" / "2026-10-21" / "09-30-00"
    assert (copied / "skill_calls.jsonl").read_text() == (
        remote / "2026-10-21/09-30-00/skill_calls.jsonl"
    ).read_text()
    assert (copied / "localization_failures").is_dir()
    assert not (session_log / "nuc_logs" / "2026-10-20").exists()
    logged = (session_log / RealSessionLog.FILENAME).read_text()
    assert "nuc logs: pulled" in logged


def test_a_failed_pull_is_logged_and_never_raises(*, session_log: Path, tmp_path: Path) -> None:
    NucLogs.pull(host=None, path=str(tmp_path / "missing"), output=session_log, wait=True)
    logged = (session_log / RealSessionLog.FILENAME).read_text()
    assert "nuc logs: pull failed" in logged


def test_a_pull_does_not_wait_for_the_copy(*, session_log: Path, tmp_path: Path) -> None:
    remote = _remote(root=tmp_path / "logs")
    thread = NucLogs.pull(host=None, path=str(remote), output=session_log, wait=False)
    assert thread is not None and thread.daemon
    thread.join(timeout=60)
    assert (session_log / "nuc_logs" / "2026-10-21" / "09-30-00").is_dir()


def test_a_fake_robot_run_pulls_the_logs_at_its_checkpoints_and_at_the_end(
    *, tmp_path: Path
) -> None:
    remote = _remote(root=tmp_path / "logs")
    output = tmp_path / "run"
    Cli.main(
        argv=[
            *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
            *("--robot", "fake", "--num-test-tasks", "1", "--seed", "0"),
            *("--practice-step-budget", "800", "--max-steps-per-interaction", "4"),
            *("--measure-every-cycles", "1", "--output-dir", str(output)),
            *("--nuc-logs-host", "local", "--nuc-logs-path", str(remote)),
        ]
    )
    assert (output / "nuc_logs" / "2026-10-21" / "09-30-00" / "skill_calls.jsonl").exists()
    logged = (output / RealSessionLog.FILENAME).read_text()
    assert "nuc logs: pulled" in logged


def test_the_real_robot_pulls_from_the_nuc_by_default_and_a_fake_does_not() -> None:
    base = [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--practice-step-budget", "100", "--output-dir", "unused"),
    ]
    real = Cli.parse_args(argv=[*base, "--robot", "real", "--robot-host", "nuc"])
    assert NucLogs.source(args=real) == ("tidybot-nuc-prpl", "prpl-tidybot/logs")
    fake = Cli.parse_args(argv=[*base, "--robot", "fake"])
    assert NucLogs.source(args=fake) is None
    off = Cli.parse_args(argv=[*base, "--robot", "real", "--robot-host", "nuc", "--no-nuc-logs"])
    assert NucLogs.source(args=off) is None
