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


def _practice_run(*, directory, status=None, dp_episodes=None):
    def robot(*, skill, interrupted=False):
        return {"event": "robot", "label": f"{skill}(robot)", "interrupted": interrupted}

    rows = [
        robot(skill="PickScraper"),
        robot(skill="Sweep"),
        {"event": "human", "destination": "rack"},
        robot(skill="GatherAndSweep"),
        robot(skill="GatherAndSweep"),
        robot(skill="GatherAndSweep", interrupted=True),
        {"event": "practice_end", "reason": "practice_episode_budget"},
    ]
    (directory / "step_events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if status is not None:
        (directory / "status.json").write_text(json.dumps(status))
    if dp_episodes is not None:
        (directory / "dp_allocations.jsonl").write_text(
            json.dumps({"event": "allocation", "episodes": 0})
            + "\n"
            + json.dumps({"event": "period_end", "episodes": dp_episodes})
            + "\n"
        )
    return RackTwoRouteSummary.practice(run=directory)["episodes"]


def test_practice_reports_the_episodes_completed_by_skill(*, tmp_path):
    """Every run, with or without an episode budget: a completed Sweep or GatherAndSweep
    is an episode; a setup skill, a person's reset and a cut try are not."""
    episodes = _practice_run(directory=tmp_path)
    assert episodes == {"completed": 3, "by_skill": {"Sweep": 1, "GatherAndSweep": 2}}


def test_a_run_with_an_episode_budget_reports_it_and_how_practice_ended(*, tmp_path):
    episodes = _practice_run(
        directory=tmp_path,
        status={
            "practice_episodes": 3,
            "practice_episode_budget": 5,
            "practice_episode_budget_reached": False,
            "practice_endpoint": "practice_step_budget",
        },
    )
    assert episodes["budget"] == 5
    assert episodes["budget_reached"] is False
    assert episodes["practice_endpoint"] == "practice_step_budget"
    assert episodes["loop_count"] == 3


@pytest.mark.parametrize(("dp_episodes", "agrees"), [(3, True), (4, False)])
def test_a_dp_runs_own_episode_count_is_reported_beside_the_loops(*, tmp_path, dp_episodes, agrees):
    episodes = _practice_run(directory=tmp_path, dp_episodes=dp_episodes)
    assert episodes["dp_count"] == dp_episodes
    assert episodes["dp_count_agrees"] is agrees
