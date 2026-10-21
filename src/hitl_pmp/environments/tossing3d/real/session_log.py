"""The real run's session log: every progress line and prompt, timestamped.

Lines go through `logging` to the terminal (INFO and up) and to
`<output-dir>/robot_session.log` (DEBUG and up, so it also keeps the placement
coordinates and each answer typed), with the same millisecond timestamp the skill
server's log can be lined up against. Each record is flushed as it is written: the
run's output usually goes through `tee`, and a buffered line is one the person never
sees while the robot is busy.
"""

import logging
import sys
import time
from pathlib import Path
from typing import ClassVar, Final

_FORMAT: Final = "%(asctime)s.%(msecs)03d %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"
# A prompt's own record goes to the file only: `input` shows it on the terminal.
_PROMPT: Final = "hitl_prompt"


class _Stdout(logging.StreamHandler):
    """Writes to whatever `sys.stdout` is at each record, never a stream captured at
    set-up (a test's capture, or a redirect made later)."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: PLR0917 -- logging's override
        self.setStream(sys.stdout)
        super().emit(record)


class _NotAPrompt(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: PLR0917 -- logging's override
        return not getattr(record, _PROMPT, False)


class RealSessionLog:
    """A static-method container, never instantiated."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger("hitl_pmp.tossing3d_real")
    FILENAME: ClassVar[str] = "robot_session.log"
    _handlers: ClassVar[list[logging.Handler]] = []
    _path: ClassVar[Path | None] = None

    @staticmethod
    def configure(*, output_dir: Path) -> None:
        """Send the session's lines to the terminal and `output_dir / FILENAME`. Again
        with the same directory is a no-op (practice and evaluation each configure)."""
        path = Path(output_dir) / RealSessionLog.FILENAME
        if RealSessionLog._path == path:
            return
        RealSessionLog.close()
        formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        terminal = _Stdout()
        terminal.setLevel(logging.INFO)
        terminal.addFilter(_NotAPrompt())
        # Opened at the first line, by which time the run has made its directory: a
        # problem built and never run leaves no file behind.
        file = logging.FileHandler(path, encoding="utf-8", delay=True)
        file.setLevel(logging.DEBUG)
        for handler in (terminal, file):
            handler.setFormatter(formatter)
            RealSessionLog.LOGGER.addHandler(handler)
        RealSessionLog._handlers = [terminal, file]
        RealSessionLog._path = path
        RealSessionLog.LOGGER.setLevel(logging.DEBUG)
        RealSessionLog.LOGGER.propagate = False

    @staticmethod
    def close() -> None:
        for handler in RealSessionLog._handlers:
            RealSessionLog.LOGGER.removeHandler(handler)
            handler.close()
        RealSessionLog._handlers = []
        RealSessionLog._path = None
        RealSessionLog.LOGGER.propagate = True

    @staticmethod
    def say(line: str) -> None:  # noqa: PLR0917 -- `TerminalPerson.write`'s callable
        if not RealSessionLog.LOGGER.handlers:
            # Not configured (no run directory): still shown, never dropped.
            print(f"{RealSessionLog.stamp()} {line}", flush=True)
            return
        RealSessionLog.LOGGER.info(line)

    @staticmethod
    def detail(*, line: str) -> None:
        """For the log file only."""
        RealSessionLog.LOGGER.debug(line)

    @staticmethod
    def ask(prompt: str) -> str:  # noqa: PLR0917 -- `TerminalPerson.read`'s callable
        """`input`, with the prompt stamped as a logged line is, and both the prompt and
        the answer kept in the log file."""
        RealSessionLog.LOGGER.info(prompt.rstrip(), extra={_PROMPT: True})
        answer = input(f"{RealSessionLog.stamp()} {prompt}")
        RealSessionLog.detail(line=f"answer: {answer!r}")
        return answer

    @staticmethod
    def stamp() -> str:
        now = time.time()
        return f"{time.strftime(_DATE_FORMAT, time.localtime(now))}.{int(now * 1000) % 1000:03d}"
