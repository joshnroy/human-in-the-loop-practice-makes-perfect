"""The skill server's session logs, copied from the NUC beside the run.

The skill server writes each session under `~/prpl-tidybot/logs/<date>/<time>/`
(`skill_calls.jsonl`, `skill_server.log`, `ceiling.mp4`, `pick_*_wrist.mp4`,
`localization_failures/`). At every cycle checkpoint, and once more when the run ends,
the newest session there is copied to `<output-dir>/nuc_logs/<date>/<time>/` with
`rsync`. A pull runs in a background thread and is never waited for, so it cannot hold
up the robot; only the end-of-run pull waits, the robot then being idle. Its success or
failure goes to the session log, and a failure never fails the run.
"""

import argparse
import shlex
import subprocess
import threading
from pathlib import Path
from typing import ClassVar, Final

from .session_log import RealSessionLog

DEFAULT_HOST: Final = "tidybot-nuc-prpl"
DEFAULT_PATH: Final = "prpl-tidybot/logs"
# `--nuc-logs-host local`: the path is a directory on this machine (a dry run, a test).
LOCAL: Final = "local"
_SSH: Final = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
_FINAL_WAIT_S: Final = 300.0


class NucLogs:
    """A static-method container, never instantiated."""

    _running: ClassVar[threading.Thread | None] = None

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--nuc-logs-host",
            default=None,
            help=f"ssh host whose skill-server logs are copied beside the run at every "
            f"checkpoint (default {DEFAULT_HOST} with --robot real; none with --robot "
            f"fake); '{LOCAL}' reads --nuc-logs-path on this machine.",
        )
        parser.add_argument(
            "--nuc-logs-path",
            default=DEFAULT_PATH,
            help="The skill server's logs directory on that host, holding <date>/<time>/.",
        )
        parser.add_argument(
            "--no-nuc-logs", action="store_true", help="Never copy the skill server's logs."
        )

    @staticmethod
    def source(*, args: argparse.Namespace) -> tuple[str | None, str] | None:
        """`(host, path)` to pull from (`host` `None`: a local directory), or `None`."""
        if getattr(args, "no_nuc_logs", False):
            return None
        host = getattr(args, "nuc_logs_host", None)
        if host is None:
            if getattr(args, "robot", None) != "real":
                return None
            host = DEFAULT_HOST
        return (None if host == LOCAL else host), args.nuc_logs_path

    @staticmethod
    def pull_for_run(*, args: argparse.Namespace, output: Path, final: bool) -> None:
        source = NucLogs.source(args=args)
        if source is not None:
            NucLogs.pull(host=source[0], path=source[1], output=output, wait=final)

    @staticmethod
    def pull(
        *, host: str | None, path: str, output: Path, wait: bool = False
    ) -> threading.Thread | None:
        """Copy the newest session in the background. With `wait`, wait (bounded) for
        any pull still running and then for this one. Without it, a pull still running
        means this one is skipped: the next checkpoint's catches up."""
        running = NucLogs._running
        if running is not None and running.is_alive():
            if not wait:
                RealSessionLog.detail(line="nuc logs: previous pull still running; skipped")
                return None
            running.join(timeout=_FINAL_WAIT_S)
        thread = threading.Thread(
            target=NucLogs._copy, kwargs={"host": host, "path": path, "output": output}, daemon=True
        )
        NucLogs._running = thread
        thread.start()
        if wait:
            thread.join(timeout=_FINAL_WAIT_S)
        return thread

    @staticmethod
    def _copy(*, host: str | None, path: str, output: Path) -> None:
        try:
            session = NucLogs._newest_session(host=host, path=path)
            destination = output / "nuc_logs" / session
            destination.mkdir(parents=True, exist_ok=True)
            remote = f"{path.rstrip('/')}/{session}/"
            command = ["rsync", "-a"]
            if host is not None:
                command += ["-e", shlex.join(_SSH), f"{host}:{remote}"]
            else:
                command.append(remote)
            command.append(f"{destination}/")
            done = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"rsync exited {done.returncode}: {done.stderr.strip()}")
            RealSessionLog.detail(line=f"nuc logs: pulled {session} to {destination}")
        except Exception as error:  # noqa: BLE001 -- logged; a pull never fails the run
            RealSessionLog.detail(line=f"nuc logs: pull failed: {type(error).__name__}: {error}")

    @staticmethod
    def _newest_session(*, host: str | None, path: str) -> str:
        """`<date>/<time>` of the newest session: both sort as written."""
        if host is None:
            sessions = sorted(
                str(p.relative_to(path)) for p in Path(path).glob("*/*") if p.is_dir()
            )
        else:
            listing = subprocess.run(
                [*_SSH, host, f"cd {shlex.quote(path)} && ls -1d */*/"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if listing.returncode != 0:
                raise RuntimeError(f"listing {host}:{path} failed: {listing.stderr.strip()}")
            sessions = sorted(line.strip().rstrip("/") for line in listing.stdout.splitlines())
        if not sessions:
            raise FileNotFoundError(f"no <date>/<time> session under {host or 'local'}:{path}")
        return sessions[-1]
