"""Two Routes: rows are read off real run directories; the report pairs arms by seed."""

import json
import shutil
from pathlib import Path

import pytest

from analysis.tworoutes_three_methods import TwoRoutesExtract, TwoRoutesReport
from tests.environments.tworoutes.run_fixtures import TwoRoutesRun

pytestmark = pytest.mark.xdist_group("tworoutes_analysis")


@pytest.fixture(scope="module")
def results_root(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real EES run, placed where the launcher puts a cost-sweep cell's run."""
    scratch = tmp_path_factory.mktemp("run")
    output = TwoRoutesRun.run(directory=scratch, method="ees", budget=40, human_cost=7.0)
    root = tmp_path_factory.mktemp("results")
    shutil.copytree(output, root / "cost" / "c7" / "ees" / "0")
    return root


def test_a_run_becomes_one_row_with_reward_read_off_the_route_taken(*, results_root: Path) -> None:
    rows = TwoRoutesExtract.extract_root(root=results_root)
    assert len(rows) == 1
    row = rows[0]
    assert (row["sweep"], row["cell"], row["arm"], row["seed"]) == ("cost", "c7", "ees", 0)
    assert (row["human_cost"], row["budget"], row["route_b_rate"]) == (7.0, 40, 0.05)
    assert row["num_tasks"] == 8
    run = results_root / "cost" / "c7" / "ees" / "0"
    actions = TwoRoutesRun.evaluation_actions(output=run, index=1)
    solved = {
        t["task"]: t["solved"]
        for t in json.loads((run / "step_evaluations.json").read_text())[-1]["tasks"]
    }
    expected = sum(
        (3.0 if a["label"].startswith("RouteB") else 1.0) * solved[a["task"]] for a in actions
    )
    assert row["reward_sum"] == pytest.approx(expected)
    assert row["reward"] == pytest.approx(expected / 8)
    assert row["deployed_b"] + row["deployed_a"] == 8
    attempts = json.loads((run / "tworoutes_attempts.jsonl").read_text().splitlines()[-1])
    assert (row["attempts_a"], row["attempts_b"]) == (
        attempts["attempts"]["RouteA"],
        attempts["attempts"]["RouteB"],
    )
    humans = [e for e in TwoRoutesRun.events(output=run) if e["event"] == "human"]
    assert row["human_resets"] == len(humans)
    assert row["practice_cost"] == pytest.approx(row["robot_steps"] + 7.0 * len(humans))
    # What the routes would pay at the competence practice reached, without sampling.
    assert row["expected_reward"] == pytest.approx(
        (row["deployed_a"] * row["p_a"] + 3.0 * row["deployed_b"] * row["p_b"]) / 8
    )
    assert row["best_expected_reward"] == pytest.approx(max(row["p_a"], 3.0 * row["p_b"]))


def test_an_unfinished_run_is_reported_not_dropped(*, results_root: Path, tmp_path: Path) -> None:
    root = tmp_path / "results"
    shutil.copytree(results_root, root)
    broken = root / "cost" / "c7" / "dp" / "3"
    broken.mkdir(parents=True)
    (broken / "failure.txt").write_text("boom")
    rows = TwoRoutesExtract.extract_root(root=root)
    assert [(r["arm"], r["seed"], r["complete"]) for r in rows] == [
        ("dp", 3, False),
        ("ees", 0, True),
    ]


def _row(*, arm: str, seed: int, reward: float, cell: str = "c1", x: float = 1.0) -> dict:
    return {
        "sweep": "cost", "cell": cell, "x": x, "arm": arm, "seed": seed, "complete": True,
        "reward": reward, "expected_reward": reward, "best_expected_reward": reward,
        "reward_sum": reward * 20, "num_tasks": 20, "num_solved": 10,
        "deployed_a": 0, "deployed_b": 20, "attempts_a": 1, "attempts_b": 2,
        "human_resets": 2, "practice_cost": 5.0, "practice_steps": 5,
    }  # fmt: skip


def test_arms_are_compared_on_the_seeds_they_share() -> None:
    rows = [
        *(_row(arm="pomdp", seed=s, reward=2.0 + 0.01 * s) for s in range(10)),
        *(_row(arm="dp", seed=s, reward=1.0 + 0.02 * s) for s in range(10)),
        _row(arm="dp", seed=10, reward=9.0),  # no pomdp run at seed 10: left out
    ]
    test = TwoRoutesReport.paired_test(rows=rows, first="pomdp", second="dp", metric="reward")
    assert test["n"] == 10
    assert test["mean_difference"] == pytest.approx(1.0 - 0.045)
    assert test["wins"] == 10 and test["losses"] == 0
    # Ten of ten positive differences: the exact two-sided signed-rank p is 2 / 2**10.
    assert test["p_value"] == pytest.approx(2 / 1024)


def test_identical_arms_support_no_inference() -> None:
    rows = [
        *(_row(arm="pomdp", seed=s, reward=2.0) for s in range(10)),
        *(_row(arm="dp", seed=s, reward=2.0) for s in range(10)),
    ]
    test = TwoRoutesReport.paired_test(rows=rows, first="pomdp", second="dp", metric="reward")
    assert test["p_value"] is None and test["mean_difference"] == 0.0


def test_the_summary_counts_runs_and_tasks_as_fractions() -> None:
    rows = [_row(arm="pomdp", seed=s, reward=2.0) for s in range(3)]
    rows[0]["deployed_a"], rows[0]["deployed_b"] = 20, 0
    summary = TwoRoutesReport.summarize(rows=rows)
    assert summary == [
        {
            "sweep": "cost", "cell": "c1", "x": 1.0, "arm": "pomdp", "runs": 3,
            "reward": pytest.approx(2.0), "expected_reward": pytest.approx(2.0),
            "best_expected_reward": pytest.approx(2.0),
            "solved": "30/60", "runs_deploying_b": "2/3",
            "attempts_a": pytest.approx(1.0), "attempts_b": pytest.approx(2.0),
            "human_resets": pytest.approx(2.0), "practice_cost": pytest.approx(5.0),
        }
    ]  # fmt: skip


def test_the_report_writes_a_figure_per_sweep(*, tmp_path: Path) -> None:
    rows = [
        _row(arm=arm, seed=s, reward=1.0 + 0.1 * s, cell=f"c{c}", x=float(c))
        for arm in ("ees", "dp", "pomdp")
        for s in range(3)
        for c in (1, 10)
    ]
    paths = TwoRoutesReport.figures(rows=rows, output_dir=tmp_path, slug="unit")
    assert [p.name for p in paths] == ["unit-cost.png"]
    assert paths[0].stat().st_size > 10_000
