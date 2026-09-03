"""Tests for bounded out-of-process expectimax trace writing."""

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


def test_search_trace_omits_packed_particle_buffers_before_queueing() -> None:
    trace = SearchTrace()
    encoded_particles = "random-looking-particle-data" * 1_000

    trace.record(
        event="node",
        node=0,
        belief_state={
            "skill_beliefs": {
                "Toss": {
                    "particle_parameters": encoded_particles,
                    "particle_weights": encoded_particles,
                    "num_particles": 512,
                }
            }
        },
    )

    serialized = json.dumps(trace.events)
    assert encoded_particles not in serialized
    assert serialized.count('"$packed": "omitted"') == 2
    assert len(serialized) < 1_000
