"""A wall-clock limit for one test, by SIGALRM, since pytest-timeout is not installed.

A pick refusal that walks every grasp at every stand once took about 40 s where a fast
one takes a few; without a limit that regression only shows as a slower suite. Marking a
test `@pytest.mark.time_limit(seconds=N)` makes it fail by name instead.
"""

import contextlib
import signal
from collections.abc import Generator
from types import FrameType


class WallClockLimitExceeded(Exception):
    """Raised inside the test body when its limit runs out."""


class WallClockLimit:
    """A static-method container, never instantiated."""

    @staticmethod
    @contextlib.contextmanager
    def armed(*, seconds: int) -> Generator[None, None, None]:
        """Raise `WallClockLimitExceeded` in the body if it runs past `seconds`."""

        def expire(signum: int, frame: FrameType | None) -> None:  # noqa: PLR0917
            del signum, frame
            raise WallClockLimitExceeded(f"test ran past its {seconds} s limit")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
