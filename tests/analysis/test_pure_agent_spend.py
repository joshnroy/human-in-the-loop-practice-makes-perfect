"""The spend report reads a journalled ledger back and totals it honestly.

The two properties worth pinning are both about not lying: a truncated last line must not
throw away the complete records before it, and a call whose cost the CLI never reported
must not silently read as free."""

import json
import tempfile
from pathlib import Path

from analysis.pure_agent.spend import SpendAnalysis
from hitl_pmp.methods.pure_agent.types import AgentCallKind, AgentPhase, PureAgentLedger


def record(*, phase="evaluation", kind="decision", cost=0.02, index=0):
    return {
        "phase": phase,
        "kind": kind,
        "cycle_index": 0,
        "episode_index": 1 if phase == "evaluation" else None,
        "step_index": index,
        "observation_digest": "a" * 64 if kind == "decision" else "",
        "reply_text": '{"skill_index": 0, "params": []}',
        "skill_index": 0 if kind == "decision" else None,
        "params": [],
        "parse_error": None,
        "seconds": 3.0,
        "total_cost_usd": cost,
        "num_turns": 1,
        "num_tool_calls": 0,
        "stop_reason": "success",
        "query_error": None,
    }


def write_ledger(*, root, lines):
    path = Path(root) / "agent_calls.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_truncated_final_line_does_not_discard_the_complete_records_before_it():
    """The file is journalled during a multi-hour run, so a killed process leaves half a
    line. Refusing the whole ledger for that would throw away exactly the spend record the
    journalling exists to preserve."""
    lines = [json.dumps(record(index=i)) for i in range(3)]
    lines.append('{"phase": "evaluation", "kind": "dec')  # killed mid-write
    with tempfile.TemporaryDirectory() as root:
        ledger = SpendAnalysis.load(path=write_ledger(root=root, lines=lines))
    assert ledger.num_calls() == 3
    assert abs(ledger.total_cost_usd() - 0.06) < 1e-9


def test_an_unreported_cost_is_counted_rather_than_read_as_free():
    lines = [
        json.dumps(record(index=0, cost=0.02)),
        json.dumps(record(index=1, cost=None)),
    ]
    with tempfile.TemporaryDirectory() as root:
        ledger = SpendAnalysis.load(path=write_ledger(root=root, lines=lines))
    assert ledger.num_calls() == 2
    assert ledger.num_calls_missing_cost() == 1
    assert abs(ledger.total_cost_usd() - 0.02) < 1e-9


def test_cells_are_split_by_phase_and_kind_because_they_scale_with_different_things():
    """`DECISION` scales with the horizon, `OPENING` with the number of episodes, and
    `DIGEST` with the number of cycles. One pooled mean answers no planning question."""
    lines = [
        json.dumps(record(phase="evaluation", kind="opening", cost=0.05)),
        json.dumps(record(phase="evaluation", kind="decision", cost=0.02)),
        json.dumps(record(phase="practice", kind="decision", cost=0.03)),
        json.dumps(record(phase="practice", kind="digest", cost=0.10)),
    ]
    with tempfile.TemporaryDirectory() as root:
        ledger = SpendAnalysis.load(path=write_ledger(root=root, lines=lines))
    cells = {(row["phase"], row["kind"]): row for row in SpendAnalysis.cell_rows(ledger=ledger)}
    assert set(cells) == {
        ("evaluation", "opening"),
        ("evaluation", "decision"),
        ("practice", "decision"),
        ("practice", "digest"),
    }
    assert cells[("practice", "digest")]["calls"] == 1
    assert ledger.num_decisions(phase=AgentPhase.PRACTICE) == 1
    assert ledger.num_calls(phase=AgentPhase.PRACTICE) == 2


def test_a_sweep_root_is_found_by_glob_in_the_layout_run_sweep_writes():
    with tempfile.TemporaryDirectory() as root:
        run = Path(root) / "pure-agent" / "3"
        run.mkdir(parents=True)
        write_ledger(root=run, lines=[json.dumps(record())])
        found = SpendAnalysis.find_ledgers(results_root=Path(root))
    assert set(found) == {"pure-agent/3"}


def test_the_figure_is_written_and_is_a_real_png():
    lines = [json.dumps(record(index=i, cost=0.01 * (i + 1))) for i in range(5)]
    with tempfile.TemporaryDirectory() as root:
        ledger = SpendAnalysis.load(path=write_ledger(root=root, lines=lines))
        figure = Path(root) / "nested" / "spend.png"
        SpendAnalysis.plot(ledgers={"pure-agent/0": ledger}, output_path=figure)
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_an_empty_ledger_reports_nothing_rather_than_dividing_by_zero():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "agent_calls.jsonl"
        path.write_text("")
        ledger = SpendAnalysis.load(path=path)
    assert ledger == PureAgentLedger()
    assert SpendAnalysis.cell_rows(ledger=ledger) == []
    assert ledger.num_decisions(phase=AgentPhase.EVALUATION) == 0
    assert AgentCallKind.DECISION is not None  # the enum is what the rows key on
