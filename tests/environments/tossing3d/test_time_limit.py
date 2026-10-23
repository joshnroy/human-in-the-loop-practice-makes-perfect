"""The per-test wall-clock limit that makes a slow pick refusal fail loudly."""

import time

import pytest

from .time_limit import WallClockLimit, WallClockLimitExceeded


def test_a_body_that_overruns_its_limit_raises() -> None:
    with pytest.raises(WallClockLimitExceeded, match="1 s"), WallClockLimit.armed(seconds=1):
        time.sleep(3)


def test_a_body_within_its_limit_runs_and_disarms_the_alarm() -> None:
    with WallClockLimit.armed(seconds=2):
        pass
    time.sleep(3)  # an alarm left armed would fire here


@pytest.mark.time_limit(seconds=1)
def test_the_marker_arms_the_limit() -> None:
    with pytest.raises(WallClockLimitExceeded):
        time.sleep(3)
