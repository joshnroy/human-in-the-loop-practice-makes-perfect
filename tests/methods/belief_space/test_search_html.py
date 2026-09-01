import json
from pathlib import Path

from hitl_pmp.methods.belief_space.expectimax import solve_belief_space_expectimax
from hitl_pmp.methods.belief_space.search_html import SearchHtml
from hitl_pmp.methods.belief_space.tossing3d_model import (
    Tossing3DPracticeModel,
    Tossing3DSearchState,
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types import SearchTrace


def test_html_preserves_every_search_event_and_escapes_script_injection(*, tmp_path: Path) -> None:
    state = make_default_tossing3d_belief()
    trace = SearchTrace()
    solve_belief_space_expectimax(
        environment_state=Tossing3DSearchState(state=state, true_atoms=frozenset()),
        belief_state=state,
        summed_cost=0,
        horizon=3,
        model=Tossing3DPracticeModel(),
        num_samples=2,
        trace=trace,
    )
    trace.record(event="annotation", text="</script><script>alert(1)</script>")
    node = next(event for event in trace.events if event["event"] == "node")
    assert node["environment_state"]["atoms"] == []
    path = tmp_path / "trees" / "stop.html"
    SearchHtml.write(path=path, trace=trace, budget=10)
    html = path.read_text()
    payload = html.split('<script type="application/json" id="data">')[1].split("</script>")[0]
    assert json.loads(payload) == {"budget": 10, "events": trace.events}
    assert "</script><script>alert" not in html
    # Every edge, including a cached successor, resolves to a recorded node.
    keys = {
        json.dumps([e["environment_state"], e["belief_state"], e["summed_cost"], e["horizon"]])
        for e in trace.events
        if e["event"] == "node"
    }
    for event in trace.events:
        if event["event"] == "branch":
            assert (
                json.dumps([
                    event["successor"],
                    event["belief_state"],
                    event["summed_cost"],
                    event["horizon"],
                ])
                in keys
            )
