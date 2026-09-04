"""Tests for compact expectimax search summaries."""

from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace


def test_search_trace_retains_compact_root_events() -> None:
    trace = SearchTrace()
    trace.record(event="stop_value", node=0, value=0.5)
    trace.record(
        event="search_summary",
        node=0,
        expanded_nodes=12,
        cache_requests=20,
        cache_hits=8,
        action_evaluations=15,
        chance_outcomes=30,
        nodes_by_horizon={2: 1, 1: 3, 0: 8},
    )

    assert [event["event"] for event in trace.events] == ["stop_value", "search_summary"]
    assert trace.events[1]["expanded_nodes"] == 12
    trace.close()
