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
