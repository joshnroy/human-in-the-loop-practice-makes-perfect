"""Run a command, and stop it before the disk it writes to fills.

    python scripts/scratch_watchdog.py --path /scratch/gpfs/... --output-dir <run dir> \
        [--threshold-gb 200] [--poll-seconds 60] [--grace-seconds 120] \
        [--free-cmd '<shell command printing free bytes>'] [--max-unreadable-polls 10] \
        -- <command> [args...]

Every `--poll-seconds` it reads the free bytes on `--path`. Below the threshold it sends
the command's whole process group SIGTERM, then SIGKILL after `--grace-seconds`, writes
`watchdog_stop.json` (time, free bytes, threshold, signal) into `--output-dir`, and exits
75 (`EX_TEMPFAIL`). If space is already below the threshold the command never starts.
Otherwise it exits with the command's own code (128 + N for a command killed by signal N).
A SIGTERM or SIGINT sent to the watchdog itself (e.g. by `scancel`) is forwarded.

**On Della the limit is a GPFS fileset quota, not the filesystem's free space**, and
`os.statvfs` reports the latter. Pass `--free-cmd` there: any shell command whose stdout
is one integer, the bytes left. For a fileset quota that is limit minus usage, e.g.

    --free-cmd "mmlsquota -j <fileset> --block-size 1 <device> | awk 'NR==3 {print $4-$3}'"

The columns of `mmlsquota` (and Della's own `checkquota`) must be checked on Della before
relying on this; the command is the user's to supply, and the watchdog only parses an
integer. A probe that fails mid-run is reported on stderr and skipped, so a transient
quota-command error cannot stop a run; but `--max-unreadable-polls` (default 10, about
10 minutes at 60 s) consecutive failures stop it exactly as low disk does, since a run
writing blind could fill the disk unseen. A probe that fails before launch is fatal.
`watchdog_stop.json`'s `reason` is `low_disk` or `free_space_unreadable` (with
`last_error` and `unreadable_polls`).

A process killed by SIGKILL, or one that does not handle SIGTERM (Python's default
action terminates without running `finally` blocks), leaves its own status files as they
were at the last write; `watchdog_stop.json` is what says it was stopped, and why.
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

from pydantic import BaseModel, Field

STOP_RECORD = "watchdog_stop.json"
STOP_EXIT_CODE = 75
DEFAULT_THRESHOLD_GB = 200.0
GB = 10**9


class ScratchWatchdog(BaseModel):
    output_dir: Path
    threshold_bytes: int
    poll_seconds: float = 60.0
    grace_seconds: float = 120.0
    max_unreadable_polls: int = Field(default=10, ge=1)
    free_space: Callable[[], int]

    def run(self, *, command: list[str]) -> int:
        free = self.free_space()
        if free < self.threshold_bytes:
            self._write_stop(reason="low_disk", free_bytes=free, sent=None, returncode=None)
            return STOP_EXIT_CODE
        child = subprocess.Popen(command, start_new_session=True)  # noqa: S603
        previous = {
            number: signal.signal(number, ScratchWatchdog._forwarder(child=child))
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        unreadable = 0
        try:
            while True:
                try:
                    returncode = child.wait(timeout=self.poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    return returncode if returncode >= 0 else 128 - returncode
                try:
                    free = self.free_space()
                except RuntimeError as error:
                    unreadable += 1
                    print(
                        f"scratch_watchdog: probe {unreadable}/{self.max_unreadable_polls} "
                        f"failed: {error}",
                        file=sys.stderr,
                    )
                    if unreadable < self.max_unreadable_polls:
                        continue
                    sent = self._stop(child=child)
                    self._write_stop(
                        reason="free_space_unreadable",
                        free_bytes=None,
                        sent=sent,
                        returncode=child.returncode,
                        last_error=str(error),
                        unreadable_polls=unreadable,
                    )
                    return STOP_EXIT_CODE
                unreadable = 0
                if free < self.threshold_bytes:
                    sent = self._stop(child=child)
                    self._write_stop(
                        reason="low_disk", free_bytes=free, sent=sent, returncode=child.returncode
                    )
                    return STOP_EXIT_CODE
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    def _stop(self, *, child: subprocess.Popen[bytes]) -> str:
        ScratchWatchdog._signal_group(child=child, number=signal.SIGTERM)
        try:
            child.wait(timeout=self.grace_seconds)
        except subprocess.TimeoutExpired:
            ScratchWatchdog._signal_group(child=child, number=signal.SIGKILL)
            child.wait()
            return "SIGKILL"
        # The leader exiting on SIGTERM does not mean every member of its group did.
        ScratchWatchdog._signal_group(child=child, number=signal.SIGKILL)
        return "SIGTERM"

    def _write_stop(
        self,
        *,
        reason: str,
        free_bytes: int | None,
        sent: str | None,
        returncode: int | None,
        last_error: str | None = None,
        unreadable_polls: int = 0,
    ) -> None:
        now = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "time_unix": now,
            "time_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "reason": reason,
            "free_bytes": free_bytes,
            "threshold_bytes": self.threshold_bytes,
            "signal": sent,
            "child_returncode": returncode,
            "last_error": last_error,
            "unreadable_polls": unreadable_polls,
        }
        (self.output_dir / STOP_RECORD).write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"scratch_watchdog: {reason} ({free_bytes} bytes free, threshold "
            f"{self.threshold_bytes}); stopped ({sent or 'never started'})",
            file=sys.stderr,
        )

    @staticmethod
    def _signal_group(*, child: subprocess.Popen[bytes], number: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, number)

    @staticmethod
    def _forwarder(*, child: subprocess.Popen[bytes]) -> Callable[[int, FrameType | None], None]:
        def forward(number: int, _frame: FrameType | None) -> None:  # noqa: PLR0917
            ScratchWatchdog._signal_group(child=child, number=signal.Signals(number))

        return forward

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--threshold-gb", type=float, default=DEFAULT_THRESHOLD_GB)
        parser.add_argument("--poll-seconds", type=float, default=60.0)
        parser.add_argument("--grace-seconds", type=float, default=120.0)
        parser.add_argument("--free-cmd", default=None)
        parser.add_argument("--max-unreadable-polls", type=int, default=10)
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(argv)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("give the command to run after --")
        source = FreeSpace(path=args.path, command=args.free_cmd)
        return ScratchWatchdog(
            output_dir=args.output_dir,
            threshold_bytes=int(args.threshold_gb * GB),
            poll_seconds=args.poll_seconds,
            grace_seconds=args.grace_seconds,
            max_unreadable_polls=args.max_unreadable_polls,
            free_space=source.bytes_free,
        ).run(command=command)


class FreeSpace(BaseModel):
    """Bytes free on `path`: from `statvfs`, or from a command that prints them."""

    path: Path
    command: str | None = None

    def bytes_free(self) -> int:
        if self.command is None:
            stats = os.statvfs(self.path)
            return stats.f_bavail * stats.f_frsize
        result = subprocess.run(  # noqa: S602 -- the user's own probe, run as they wrote it
            self.command, shell=True, capture_output=True, text=True, check=False, timeout=60
        )
        try:
            if result.returncode != 0:
                raise ValueError(result.stderr.strip())
            return int(result.stdout.strip())
        except ValueError as error:
            raise RuntimeError(
                f"free-space command {self.command!r} gave {result.stdout.strip()!r} "
                f"(exit {result.returncode}): {error}"
            ) from error


if __name__ == "__main__":
    sys.exit(ScratchWatchdog.main())
