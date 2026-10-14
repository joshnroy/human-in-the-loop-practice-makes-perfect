"""Curves must preserve observed policy checkpoints and real physical experience."""

import json
from pathlib import Path

from analysis.step_matched import StepMatchedAnalysis


class Fixtures:
    @staticmethod
    def run(*, root: Path) -> None:
        (root / "evaluations" / "0000").mkdir(parents=True)
        (root / "evaluations" / "0001").mkdir()
        metadata = [
            dict(
                index=i,
                practice_steps=i * 1700,
                robot_steps=i * 1699,
                human_invocations=i,
                human_resets_by_destination={"robot_side": i, "opposite_side": 0},
                policy_sha256=str(i),
            )
            for i in range(3)
        ]
        (root / "measurements.json").write_text(json.dumps(metadata))
        (root / "status.json").write_text(json.dumps(dict(phase="running", practice_steps=3800)))
        (root / "evaluations/0000/results.json").write_text(
            json.dumps(dict(index=0, complete=True, num_solved=0, num_total=10))
        )
        (root / "evaluations/0001/results.json").write_text(
            json.dumps(dict(index=1, complete=False, num_solved=3, num_total=5))
        )
        (root / "step_events.jsonl").write_text(
            "\n".join(
                json.dumps(x)
                for x in [
                    dict(event="human", end_step=220, destination="robot_side"),
                    dict(event="human", end_step=2000, destination="opposite_side"),
                    dict(event="robot", start_step=2000, end_step=2200),
                ]
            )
            + "\n"
        )


def test_pending_evaluations_are_not_zero_scores_or_future_learning(*, tmp_path):
    Fixtures.run(root=tmp_path)
    run = StepMatchedAnalysis.load_run(root=tmp_path, method="ees")
    assert [(x["practice_steps"], x["num_solved"]) for x in run["evaluations"]] == [(0, 0)]
    assert run["pending_indices"] == [1, 2]
    assert run["endpoint_steps"] == 3800
    assert run["human_resets"][-1] == dict(practice_steps=2000, robot_side=1, opposite_side=1)


def test_complete_checkpoint_keeps_snapshot_counter_not_current_training_counter(*, tmp_path):
    Fixtures.run(root=tmp_path)
    (tmp_path / "evaluations/0001/results.json").write_text(
        json.dumps(dict(index=1, complete=True, num_solved=6, num_total=10))
    )
    run = StepMatchedAnalysis.load_run(root=tmp_path, method="pddl")
    assert run["evaluations"][-1]["practice_steps"] == 1700
    assert run["evaluations"][-1]["num_solved"] == 6
    assert run["pending_indices"] == [2]


def test_full_agentic_final_revision_at_same_counter_remains_distinct(*, tmp_path):
    records = [
        dict(
            index=i,
            practice_steps=steps,
            sha256=str(i),
            human_by_side={"robot_side": 0, "opposite_side": i},
            num_solved=i,
            num_total=10,
        )
        for i, steps in enumerate([0, 1700, 1700])
    ]
    (tmp_path / "step_evaluations.json").write_text(json.dumps(records))
    (tmp_path / "status.json").write_text(json.dumps(dict(phase="complete", practice_steps=1700)))
    run = StepMatchedAnalysis.load_run(root=tmp_path, method="full-agentic")
    assert [x["practice_steps"] for x in run["evaluations"]] == [0, 1700, 1700]
    assert [x["policy_sha256"] for x in run["evaluations"]] == ["0", "1", "2"]
    assert run["pending_indices"] == []


def test_old_pilot_has_only_observed_endpoints():
    points = StepMatchedAnalysis.pilot_points(
        robot_steps=33877, human_requests=101, initial_solved=0, final_solved=10, tasks=10
    )
    assert [x["practice_steps"] for x in points] == [0, 33978]
    assert all(x["protocol"] == "old_pilot" for x in points)


def test_live_capture_uses_status_cutoff_for_later_appended_events(*, tmp_path):
    Fixtures.run(root=tmp_path)
    with (tmp_path / "step_events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event="human", end_step=3900, destination="opposite_side")) + "\n"
        )
    run = StepMatchedAnalysis.load_run(root=tmp_path, method="ees")
    assert run["human_resets"][-1]["opposite_side"] == 1
    assert all(p["practice_steps"] <= run["endpoint_steps"] for p in run["human_resets"])
