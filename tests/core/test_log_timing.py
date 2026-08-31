import json
from datetime import datetime

from hitl_pmp.core.log_timing import LogTiming


def test_event_timing_is_utc_and_monotonic() -> None:
    first = json.loads(LogTiming.encode(record={"event": "first"}))
    second = json.loads(LogTiming.encode(record={"event": "second"}))
    assert first["event"] == "first"
    assert second["elapsed_seconds"] >= first["elapsed_seconds"] >= 0
    assert datetime.fromisoformat(first["timestamp"]).utcoffset().total_seconds() == 0
