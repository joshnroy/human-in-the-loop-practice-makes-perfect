import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from scripts.scratch_watchdog import STOP_EXIT_CODE, STOP_RECORD, FreeSpace, ScratchWatchdog

GB = 10**9


def _source(*, readings: list[int]) -> Callable[[], int]:
    """Free bytes from a script; the last reading repeats once the script runs out."""
    remaining: Iterator[int] = iter(readings)
    last = readings[-1]

    def read() -> int:
        return next(remaining, last)

    return read


def _watchdog(*, tmp_path: Path, readings: list[int], grace_seconds: float = 5.0):
    return ScratchWatchdog(
        output_dir=tmp_path / "run",
        threshold_bytes=200 * GB,
        poll_seconds=0.05,
        grace_seconds=grace_seconds,
        free_space=_source(readings=readings),
    )


def _python(*, code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_a_child_that_finishes_with_space_to_spare_keeps_its_exit_code(*, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path=tmp_path, readings=[900 * GB])
    assert watchdog.run(command=_python(code="import sys; sys.exit(3)")) == 3
    assert not (tmp_path / "run" / STOP_RECORD).exists()


def test_low_space_terminates_the_child_and_writes_a_stop_record(*, tmp_path: Path) -> None:
    marker = tmp_path / "got_sigterm"
    child = (
        "import signal, sys, time, pathlib\n"
        "def stop(*_):\n"
        f"    pathlib.Path({str(marker)!r}).write_text('term')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    watchdog = _watchdog(tmp_path=tmp_path, readings=[900 * GB, 900 * GB, 150 * GB])
    started = time.monotonic()
    assert watchdog.run(command=_python(code=child)) == STOP_EXIT_CODE
    assert time.monotonic() - started < 10
    assert marker.read_text() == "term"
    record = json.loads((tmp_path / "run" / STOP_RECORD).read_text())
    assert record["free_bytes"] == 150 * GB
    assert record["threshold_bytes"] == 200 * GB
    assert record["signal"] == "SIGTERM"
    assert record["time_unix"] == pytest.approx(time.time(), abs=60)
    assert "time_utc" in record


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(*, tmp_path: Path) -> None:
    child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    watchdog = _watchdog(tmp_path=tmp_path, readings=[900 * GB, 100 * GB], grace_seconds=0.5)
    started = time.monotonic()
    assert watchdog.run(command=_python(code=child)) == STOP_EXIT_CODE
    assert time.monotonic() - started < 10
    record = json.loads((tmp_path / "run" / STOP_RECORD).read_text())
    assert record["signal"] == "SIGKILL"


def test_the_whole_process_group_is_signalled(*, tmp_path: Path) -> None:
    # A launcher shell's grandchild (the actual run) must not survive the stop.
    pid_file = tmp_path / "grandchild.pid"
    command = ["sh", "-c", f"sleep 60 & echo $! > {pid_file}; wait"]
    watchdog = _watchdog(tmp_path=tmp_path, readings=[900 * GB, 900 * GB, 900 * GB, 10 * GB])
    assert watchdog.run(command=command) == STOP_EXIT_CODE
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild survived the stop")


def test_space_already_below_threshold_never_starts_the_child(*, tmp_path: Path) -> None:
    marker = tmp_path / "started"
    watchdog = _watchdog(tmp_path=tmp_path, readings=[1 * GB])
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')"
    assert watchdog.run(command=_python(code=code)) == STOP_EXIT_CODE
    assert not marker.exists()
    record = json.loads((tmp_path / "run" / STOP_RECORD).read_text())
    assert record["signal"] is None
    assert record["free_bytes"] == 1 * GB


def test_free_space_from_statvfs(*, tmp_path: Path) -> None:
    stats = os.statvfs(tmp_path)
    expected = stats.f_bavail * stats.f_frsize
    assert FreeSpace(path=tmp_path).bytes_free() == pytest.approx(expected, rel=0.01)


def test_free_space_from_a_command_that_prints_bytes(*, tmp_path: Path) -> None:
    assert FreeSpace(path=tmp_path, command="echo '  123456789 '").bytes_free() == 123456789


def test_a_failing_free_space_command_raises(*, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="free-space command"):
        FreeSpace(path=tmp_path, command="echo not-a-number").bytes_free()


def test_a_failing_probe_mid_run_is_skipped_not_fatal(*, tmp_path: Path) -> None:
    # A quota command hiccup must not kill 48 runs; the next good reading decides.
    readings = iter([900 * GB, None, None])

    def flaky() -> int:
        value = next(readings, 900 * GB)
        if value is None:
            raise RuntimeError("free-space command failed")
        return value

    watchdog = ScratchWatchdog(
        output_dir=tmp_path / "run",
        threshold_bytes=200 * GB,
        poll_seconds=0.05,
        grace_seconds=1.0,
        free_space=flaky,
    )
    assert watchdog.run(command=_python(code="import time; time.sleep(0.5)")) == 0


def test_main_reads_a_free_space_command_and_the_default_threshold(*, tmp_path: Path) -> None:
    output = tmp_path / "run"
    argv = [
        "--path",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--free-cmd",
        f"echo {199 * GB}",
        "--poll-seconds",
        "0.05",
        "--",
        *_python(code="import time; time.sleep(60)"),
    ]
    assert ScratchWatchdog.main(argv=argv) == STOP_EXIT_CODE
    record = json.loads((output / STOP_RECORD).read_text())
    assert record["threshold_bytes"] == 200 * GB
    assert record["free_bytes"] == 199 * GB
