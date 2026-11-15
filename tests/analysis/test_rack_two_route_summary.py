"""Two-route Serving summary: a task is solved, and rewarded, by the skill that ended it."""

import json

import pytest

from analysis.rack_two_route_summary import RackTwoRouteSummary


@pytest.mark.parametrize(
    ("skill", "count", "solved", "reward"),
    [
        ("Sweep", 2, False, 0),
        ("Sweep", 3, True, 3),
        ("Sweep", 5, True, 3),
        ("GatherAndSweep", 4, False, 0),
        ("GatherAndSweep", 5, True, 5),
        (None, 5, False, 0),
    ],
)
def test_solved_and_reward_are_by_skill(*, skill, count, solved, reward):
    assert RackTwoRouteSummary.solved(skill=skill, count=count) is solved
    assert RackTwoRouteSummary.reward(skill=skill, count=count) == reward


def test_an_evaluation_is_scored_from_each_tasks_bulk_event(*, tmp_path):
    def event(*, task, skill, count):
        return {"task": task, "label": f"{skill}(robot)", "skill_trace": {"plate_count": count}}

    (tmp_path / "results.json").write_text(
        json.dumps({
            "tasks": [
                {"task": 0, "solved": True, "reason": "goal"},
                {"task": 1, "solved": False, "reason": "bulk_attempt_finished"},
                {"task": 2, "solved": True, "reason": "goal"},
            ],
            "deployment_policy": {"kind": "dp_mdp"},
        })
    )
    (tmp_path / "actions.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                event(task=0, skill="PickScraper", count=0),
                event(task=0, skill="Sweep", count=4),
                event(task=1, skill="GatherAndSweep", count=4),
                event(task=2, skill="GatherAndSweep", count=5),
            )
        )
        + "\n"
    )
    summary = RackTwoRouteSummary.evaluation(directory=tmp_path)
    assert [(t["bulk_skill"], t["solved"], t["reward"]) for t in summary["tasks"]] == [
        ("Sweep", True, 3),
        ("GatherAndSweep", False, 0),
        ("GatherAndSweep", True, 5),
    ]
    assert (summary["num_solved"], summary["total_reward"]) == (2, 8)


def test_a_run_is_rewarded_with_the_values_its_config_snapshot_names(*, tmp_path):
    evaluation = tmp_path / "evaluations" / "0"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(
        json.dumps({
            "tasks": [
                {"task": 0, "solved": True, "reason": "goal"},
                {"task": 1, "solved": True, "reason": "goal"},
            ],
            "deployment_policy": {"kind": "dp_mdp"},
        })
    )
    (evaluation / "actions.jsonl").write_text(
        json.dumps({"task": 0, "label": "Sweep(robot)", "skill_trace": {"plate_count": 3}})
        + "\n"
        + json.dumps({
            "task": 1,
            "label": "GatherAndSweep(robot)",
            "skill_trace": {"plate_count": 5},
        })
        + "\n"
    )
    (tmp_path / "config_snapshot.json").write_text(
        json.dumps({"args": {"served_three_reward": "5.0", "served_five_reward": "10.0"}})
    )
    summary = RackTwoRouteSummary.run(run=tmp_path)["evaluations"]["0"]
    assert [t["reward"] for t in summary["tasks"]] == [5.0, 10.0]
    assert (summary["total_reward"], summary["max_reward"]) == (15.0, 20.0)


def test_a_run_without_configured_rewards_keeps_three_and_five(*, tmp_path):
    assert RackTwoRouteSummary.rewards(run=tmp_path) == {"Sweep": 3.0, "GatherAndSweep": 5.0}
    (tmp_path / "config_snapshot.json").write_text(json.dumps({"args": {"seed": "0"}}))
    assert RackTwoRouteSummary.rewards(run=tmp_path) == {"Sweep": 3.0, "GatherAndSweep": 5.0}
