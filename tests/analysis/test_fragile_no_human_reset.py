import json
from pathlib import Path

import pytest

from analysis.fragile_no_human_reset import FragileNoHumanReset


def _write(*, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _structured_run(*, root: Path, phase: str, with_practice_end: bool) -> None:
    _write(
        path=root / "status.json",
        payload={
            "phase": phase,
            "practice_steps": 590,
            "human_invocations": 0,
            "damage_contacts": 2,
            "evaluations_completed": 2,
        },
    )
    _write(
        path=root / "step_evaluations.json",
        payload=[
            {"index": 0, "practice_steps": 0, "num_solved": 1, "num_total": 10, "complete": True},
            {"index": 1, "practice_steps": 590, "num_solved": 1, "num_total": 10, "complete": True},
        ],
    )
    events = [
        {"event": "measurement", "index": 0, "practice_steps": 0},
        {"event": "robot", "cycle": 0, "label": "PickCube", "start_step": 0, "end_step": 117},
        {"event": "damage", "practice_step": 209, "impacts": [{"event_id": 1}, {"event_id": 2}]},
        {"event": "robot", "cycle": 0, "label": "Toss", "start_step": 117, "end_step": 590},
        {"event": "measurement", "index": 1, "practice_steps": 590},
    ]
    if with_practice_end:
        events.append({
            "event": "practice_end",
            "practice_steps": 590,
            "reason": "no_practice_action",
        })
    (root / "step_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_structured_loader_reads_evaluations_damage_and_endpoint(*, tmp_path: Path) -> None:
    _structured_run(root=tmp_path, phase="complete", with_practice_end=True)
    run = FragileNoHumanReset.load_structured_run(root=tmp_path)
    assert run["evaluations"] == [[0, 1, 10], [590, 1, 10]]
    assert run["damage_events"] == [[0, 0], [209, 2], [590, 2]]
    assert run["endpoint_steps"] == 590
    assert run["endpoint_reason"] == "no_practice_action"
    assert run["complete"] is True
    assert run["robot_invocations"] == 2
    assert run["human_invocations"] == 0


def test_structured_loader_marks_a_running_run_incomplete(*, tmp_path: Path) -> None:
    _structured_run(root=tmp_path, phase="running", with_practice_end=False)
    (tmp_path / "step_evaluations.json").unlink()
    _write(
        path=tmp_path / "measurements.json",
        payload=[{"index": 0, "practice_steps": 0}, {"index": 1, "practice_steps": 590}],
    )
    _write(
        path=tmp_path / "evaluations" / "0000" / "results.json",
        payload={"num_solved": 0, "num_total": 10, "complete": True},
    )
    # Evaluation 0001 has started but has no results yet: it must not appear as a point.
    (tmp_path / "evaluations" / "0001").mkdir(parents=True)
    run = FragileNoHumanReset.load_structured_run(root=tmp_path)
    assert run["evaluations"] == [[0, 0, 10]]
    assert run["complete"] is False
    assert run["endpoint_reason"] is None


def test_full_agentic_loader_reads_snapshots_and_practice_damage(*, tmp_path: Path) -> None:
    _write(
        path=tmp_path / "status.json",
        payload={
            "phase": "complete",
            "practice_steps": 300,
            "human_requests": 0,
            "damage_contacts": 3,
            "endpoint": "agent_submission",
        },
    )
    for index, steps, solved in ((0, 0, 0), (1, 300, 10)):
        _write(
            path=tmp_path / "measurements" / f"{index:04d}" / "snapshot.json",
            payload={"index": index, "practice_steps": steps, "damage_contacts": 3 if steps else 0},
        )
        _write(
            path=tmp_path / "evaluations" / f"{index:04d}" / "results.json",
            payload={"num_solved": solved, "num_total": 10, "complete": True},
        )
    records = [
        {
            "request": {"operation": "observe"},
            "response": {"result": {"practice_steps": 0, "damage_contacts": 0}},
        },
        {
            "request": {"operation": "step"},
            "response": {"result": {"practice_steps": 40, "damage_contacts": 1}},
        },
        {
            "request": {"operation": "step"},
            "response": {"result": {"practice_steps": 41, "damage_contacts": 1}},
        },
        {
            "request": {"operation": "step"},
            "response": {"result": {"practice_steps": 120, "damage_contacts": 3}},
        },
    ]
    (tmp_path / "practice.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    run = FragileNoHumanReset.load_full_agentic_run(root=tmp_path)
    assert run["evaluations"] == [[0, 0, 10], [300, 10, 10]]
    assert run["damage_events"] == [[0, 0], [40, 1], [120, 3], [300, 3]]
    assert run["endpoint_steps"] == 300
    assert run["endpoint_reason"] == "agent_submission"
    assert run["complete"] is True


def test_damage_endpoint_must_match_status(*, tmp_path: Path) -> None:
    _structured_run(root=tmp_path, phase="complete", with_practice_end=True)
    status = json.loads((tmp_path / "status.json").read_text())
    status["damage_contacts"] = 5
    _write(path=tmp_path / "status.json", payload=status)
    with pytest.raises(ValueError, match="damage"):
        FragileNoHumanReset.load_structured_run(root=tmp_path)


def test_plot_writes_both_figures(*, tmp_path: Path) -> None:
    arm = {
        "key": "ees",
        "method": "EES",
        "human_reset": False,
        "seed": 0,
        "complete": True,
        "endpoint_steps": 590,
        "endpoint_reason": "no_practice_action",
        "evaluations": [[0, 1, 10], [590, 1, 10]],
        "damage_events": [[0, 0], [209, 2], [590, 2]],
        "robot_invocations": 2,
        "human_invocations": 0,
    }
    comparator = dict(
        arm, human_reset=True, endpoint_steps=3400, endpoint_reason="practice_step_budget"
    )
    comparator["evaluations"] = [[0, 1, 10], [1700, 3, 10], [3400, 8, 10]]
    comparator["damage_events"] = [[0, 0], [500, 4], [3400, 4]]
    curves = {"snapshot": "test", "arms": [arm, comparator]}
    written = FragileNoHumanReset.plot(curves=curves, output_dir=tmp_path, stem="fig")
    assert sorted(p.name for p in written) == ["fig-performance.png", "fig-raw-ground-hits.png"]
    assert all(p.stat().st_size > 0 for p in written)
