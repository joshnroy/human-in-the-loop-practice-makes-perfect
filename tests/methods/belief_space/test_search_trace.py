"""Tests for lossless out-of-process expectimax trace writing."""

import gzip
import json
from pathlib import Path

import pytest

from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace


def test_search_trace_writes_complete_gzip_from_another_process(*, tmp_path: Path) -> None:
    path = tmp_path / "decision.jsonl.gz"
    trace = SearchTrace(path=path, retain_events=False)

    trace.record(event="node", node=0, environment_state={"phase": "ready"})
    trace.record(event="choice", node=0, action="STOP", value=0.5)

    assert not path.exists()
    trace.close()

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert [record["event"] for record in records] == ["intern", "node", "choice"]
    assert records[1]["environment_state"] == {"$ref": 0, "$kind": "state"}
    assert [event["event"] for event in trace.events] == ["node", "choice"]
    assert not list(tmp_path.glob("*.tmp"))


def test_search_trace_close_propagates_writer_failure(*, tmp_path: Path) -> None:
    trace = SearchTrace(path=tmp_path / "decision.jsonl.gz", retain_events=False)
    trace.record(event="invalid", node=1, value=float("nan"))

    with pytest.raises(RuntimeError, match="search trace writer failed"):
        trace.close()


def test_search_trace_close_is_idempotent(*, tmp_path: Path) -> None:
    trace = SearchTrace(path=tmp_path / "decision.jsonl.gz", retain_events=False)
    trace.record(event="choice", node=0, action="STOP")

    trace.close()
    trace.close()
