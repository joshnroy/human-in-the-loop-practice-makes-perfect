"""The LLM-planner log must read actual endpoints and reconcile every total it plots."""

import json
from pathlib import Path

import pytest

from analysis.llm_practice_planner import LlmPracticePlanner


class Fixtures:
    PICK = {"skill": {"name": "PickCube", "parameters": []}, "objects": []}
    RESET = {"skill": {"name": "ask_for_reset_cube_bin_only", "parameters": []}, "objects": []}

    @staticmethod
    def write(*, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @staticmethod
    def lines(*, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    @staticmethod
    def run(*, root: Path, spent: float = 0.10, damage_contacts: int = 3) -> None:
        Fixtures.write(
            path=root / "status.json",
            payload={
                "phase": "complete",
                "practice_steps": 900,
                "human_invocations": 3,
                "human_resets_by_destination": {"robot_side": 1, "opposite_side": 2},
                "damage_contacts": damage_contacts,
                "physical_cost": 1230.0,
            },
        )
        Fixtures.write(
            path=root / "step_evaluations.json",
            payload=[
                {"index": 0, "practice_steps": 0, "num_solved": 1, "num_total": 10},
                {"index": 1, "practice_steps": 500, "num_solved": 8, "num_total": 10},
                {"index": 2, "practice_steps": 900, "num_solved": 2, "num_total": 10},
            ],
        )
        Fixtures.write(
            path=root / "model_budget.json",
            payload={"limit": 20.0, "sessions": {"a": 0.06, "b": 0.04}, "spent": spent},
        )
        for index in range(5):
            Fixtures.write(
                path=root / "planner_calls" / f"{index:04d}" / "result.json",
                payload={"accepted": index < 4, "cost_usd": 0.01},
            )
        Fixtures.lines(
            path=root / "step_events.jsonl",
            rows=[
                {"event": "measurement", "index": 0, "practice_steps": 0},
                {"event": "human", "end_step": 1, "destination": "opposite_side"},
                {"event": "damage", "practice_step": 200, "impacts": [{}, {}]},
                {"event": "human", "end_step": 300, "destination": "robot_side"},
                {"event": "human", "end_step": 600, "destination": "opposite_side"},
                {"event": "damage", "practice_step": 700, "impacts": [{}]},
                {"event": "practice_end", "practice_steps": 900, "reason": "no_practice_action"},
            ],
        )
        Fixtures.lines(
            path=root / "pomdp_decisions.jsonl",
            rows=[
                {"event": "session_start", "cycle": 0},
                {
                    "event": "decision",
                    "cycle": 0,
                    "action": Fixtures.RESET,
                    "llm_reason": "Plenty of step budget remains, so reset to practise.",
                    "llm_cost_usd": 0.03,
                    "practice_steps_used": 0,
                },
                {
                    "event": "decision",
                    "cycle": 0,
                    "action": Fixtures.PICK,
                    "llm_reason": "Pick to set up the toss.",
                    "llm_cost_usd": 0.02,
                    "practice_steps_used": 1,
                },
                {
                    "event": "decision",
                    "cycle": 0,
                    "action": "STOP",
                    "llm_reason": "Not paying off.",
                    "llm_cost_usd": 0.01,
                    "practice_steps_used": 900,
                },
                {"event": "session_start", "cycle": 1},
                {
                    "event": "decision",
                    "cycle": 1,
                    "action": "STOP",
                    "llm_reason": "A reset costs 100, so stop now.",
                    "llm_cost_usd": 0.04,
                    "practice_steps_used": 900,
                },
            ],
        )


def test_load_run_reads_evaluations_resets_by_side_damage_and_endpoint(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path)
    run = LlmPracticePlanner.load_run(root=tmp_path)
    assert run["evaluations"] == [[0, 1, 10], [500, 8, 10], [900, 2, 10]]
    assert run["human_events"] == [[0, 0, 0], [1, 0, 1], [300, 1, 1], [600, 1, 2], [900, 1, 2]]
    assert run["damage_events"] == [[0, 0], [200, 2], [700, 3], [900, 3]]
    assert run["endpoint_steps"] == 900
    assert run["endpoint_reason"] == "no_practice_action"
    assert run["physical_cost"] == 1230.0


def test_spend_curve_closes_on_the_ledger_total_at_the_endpoint(*, tmp_path: Path) -> None:
    # One call is in the ledger but has no decision (a call that hit the budget).
    Fixtures.run(root=tmp_path, spent=0.13)
    run = LlmPracticePlanner.load_run(root=tmp_path)
    steps = [p[0] for p in run["spend_events"]]
    usd = [p[1] for p in run["spend_events"]]
    assert steps == [0, 0, 1, 900, 900, 900]
    assert usd == pytest.approx([0.0, 0.03, 0.05, 0.06, 0.10, 0.13])
    assert run["model_spent_usd"] == pytest.approx(0.13)
    assert run["model_calls"] == 5
    assert run["decisions"] == 4


def test_decisions_costing_more_than_the_ledger_raise(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path, spent=0.05)
    with pytest.raises(ValueError, match="ledger"):
        LlmPracticePlanner.load_run(root=tmp_path)


def test_damage_that_does_not_reconcile_with_status_raises(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path, damage_contacts=4)
    with pytest.raises(ValueError, match="damage"):
        LlmPracticePlanner.load_run(root=tmp_path)


def test_the_terminal_stop_is_recorded_with_whether_it_opened_a_session(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path)
    run = LlmPracticePlanner.load_run(root=tmp_path)
    assert run["final_decision"] == {
        "action": "STOP",
        "first_in_session": True,
        "practice_steps": 900,
        "reason": "A reset costs 100, so stop now.",
    }
    assert run["stop_decisions"] == 2


def test_reset_reasons_are_counted_by_what_they_cite(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path)
    run = LlmPracticePlanner.load_run(root=tmp_path)
    assert run["reset_reasons"] == {
        "total": 1,
        "practice": 1,
        "step_budget": 1,
        "actions_left": 0,
        "cost": 0,
        "human": 0,
    }


def test_per_decision_cost_quantiles(*, tmp_path: Path) -> None:
    Fixtures.run(root=tmp_path)
    run = LlmPracticePlanner.load_run(root=tmp_path)
    assert run["decision_cost_usd"]["median"] == pytest.approx(0.025)
    assert run["decision_cost_usd"]["min"] == pytest.approx(0.01)
    assert run["decision_cost_usd"]["max"] == pytest.approx(0.04)


def test_score_summary_reports_final_best_mean_of_last_three_and_largest_swing() -> None:
    summary = LlmPracticePlanner.score_summary(
        evaluations=[[0, 1, 10], [1, 8, 10], [2, 2, 10], [3, 9, 10], [4, 4, 10]]
    )
    assert summary == {"final": 4, "best": 9, "mean_last_three": 5.0, "largest_swing": 7}


def test_footer_names_the_environment_and_the_endpoint_markers() -> None:
    text = LlmPracticePlanner.footer()
    assert "Tossing3D" in text
    assert "Diamonds" in text
