"""The one path that runs when money has already been spent and a decision is at risk.

`prpl_agent_utils` raises whenever the CLI exits without a parseable final `result`, and
**every `--max-budget-usd` stop takes that path** -- the stop reports its own cost and
then exits in a shape the package treats as fatal, discarding a decision already written.
`recover_from_stream_log` reads both halves back out of the log the package itself wrote.

Tested directly against synthetic logs rather than through a query, because it is a pure
file reader and because provoking a real budget stop costs money to observe. Nothing here
needs `prpl_agent_utils` installed, so it runs on CI."""

import json
import tempfile
from pathlib import Path

from hitl_pmp.methods.pure_agent.claude_code_backend import STREAM_LOG, ClaudeCodeAgentBackend


def assistant(*, text):
    return json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    })


def result(*, cost=0.05, subtype="success", include_result_field=True):
    """A CLI `result` message. `include_result_field=False` is the budget-stop shape:
    measured upstream, the stop emits a result with `subtype: "error_max_budget_usd"` and
    **no** `result` key, which is precisely what `_parse_stream` refuses."""
    message = {
        "type": "result",
        "is_error": subtype != "success",
        "num_turns": 1,
        "total_cost_usd": cost,
        "subtype": subtype,
    }
    if include_result_field:
        message["result"] = "ok"
    return json.dumps(message)


def backend_with_log(*, root, lines):
    sandbox = Path(root) / "sandbox"
    (sandbox / STREAM_LOG.parent).mkdir(parents=True)
    (sandbox / STREAM_LOG).write_text("\n".join(lines) + "\n")
    return ClaudeCodeAgentBackend(sandbox_dir=sandbox)


def test_a_budget_stopped_call_still_yields_the_decision_it_already_paid_for():
    """The whole reason the cap is usable. The agent emitted its one line of JSON and then
    the CLI stopped on its own cap; the answer is real and so is the cost."""
    with tempfile.TemporaryDirectory() as root:
        backend = backend_with_log(
            root=root,
            lines=[
                assistant(text='{"skill_index": 2, "params": [7.5]}'),
                result(cost=0.51, subtype="error_max_budget_usd", include_result_field=False),
            ],
        )
        text, metadata = backend.recover_from_stream_log()
    assert text == '{"skill_index": 2, "params": [7.5]}'
    assert metadata["total_cost_usd"] == 0.51
    assert metadata["stop_reason"] == "error_max_budget_usd"
    assert metadata["is_error"] is True


def test_the_log_spans_queries_so_the_last_pair_is_the_one_returned():
    """The log is APPENDED per query. Reading the last text and the last result
    independently would pair this call's answer with the previous call's cost, which is
    exactly the bug this pairing exists to prevent."""
    with tempfile.TemporaryDirectory() as root:
        backend = backend_with_log(
            root=root,
            lines=[
                assistant(text="FIRST"),
                result(cost=0.01),
                assistant(text="SECOND"),
                result(cost=0.02, subtype="error_max_budget_usd", include_result_field=False),
            ],
        )
        text, metadata = backend.recover_from_stream_log()
    assert text == "SECOND"
    assert metadata["total_cost_usd"] == 0.02


def test_a_stream_killed_before_its_result_reports_the_text_and_an_unknown_cost():
    """The previous call's cost must NOT be attributed to this one. An honestly unknown
    cost is counted by `num_calls_missing_cost`; a borrowed one is silently wrong."""
    with tempfile.TemporaryDirectory() as root:
        backend = backend_with_log(
            root=root,
            lines=[assistant(text="FIRST"), result(cost=0.01), assistant(text="SECOND")],
        )
        text, metadata = backend.recover_from_stream_log()
    assert text == "SECOND"
    assert metadata == {}


def test_no_log_at_all_is_an_unknown_cost_rather_than_a_free_call():
    with tempfile.TemporaryDirectory() as root:
        backend = ClaudeCodeAgentBackend(sandbox_dir=Path(root) / "never-created")
        assert backend.recover_from_stream_log() == ("", {})


def test_unparseable_lines_are_skipped_rather_than_fatal():
    """The log is written line by line by another process; a partial line at the end is
    normal and must not cost the records before it."""
    with tempfile.TemporaryDirectory() as root:
        backend = backend_with_log(
            root=root,
            lines=[
                "not json at all",
                assistant(text="ANSWER"),
                result(cost=0.03, subtype="error_max_budget_usd", include_result_field=False),
                '{"type": "assist',
            ],
        )
        text, metadata = backend.recover_from_stream_log()
    assert text == "ANSWER"
    assert metadata["total_cost_usd"] == 0.03


def test_a_cap_that_fires_before_any_answer_leaves_an_empty_reply_to_be_counted():
    """Honest, and visible: the caller records a malformed decision and takes a no-op."""
    with tempfile.TemporaryDirectory() as root:
        backend = backend_with_log(
            root=root,
            lines=[result(cost=0.5, subtype="error_max_budget_usd", include_result_field=False)],
        )
        text, metadata = backend.recover_from_stream_log()
    assert text == ""
    assert metadata["total_cost_usd"] == 0.5


def test_the_per_query_cap_is_on_by_default():
    """Off in `prpl-agent-utils` and off on the previous arm, which made 6 calls per
    trial. This one makes thousands."""
    assert ClaudeCodeAgentBackend.model_fields["max_budget_usd_per_query"].default > 0
