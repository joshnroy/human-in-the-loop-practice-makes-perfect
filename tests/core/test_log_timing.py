import json
from datetime import datetime

from hitl_pmp.core.log_timing import LogTiming


def test_event_timing_is_utc_and_monotonic() -> None:
    first = json.loads(LogTiming.encode(record={"event": "first"}))
    second = json.loads(LogTiming.encode(record={"event": "second"}))
    assert first["event"] == "first"
    assert second["elapsed_seconds"] >= first["elapsed_seconds"] >= 0
    assert datetime.fromisoformat(first["timestamp"]).utcoffset().total_seconds() == 0


def test_nested_non_finite_diagnostics_are_valid_json() -> None:
    record = json.loads(
        LogTiming.encode(
            record={"value": -float("inf"), "search": [{"bound": float("inf")}]}
        )
    )

    assert record["value"] == "-Infinity"
    assert record["search"] == [{"bound": "Infinity"}]
