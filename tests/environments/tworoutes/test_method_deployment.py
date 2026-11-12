"""Each method deploys the policy it valued: EES for success alone, DP and the
belief-space method for reward."""

import json
from pathlib import Path

import pytest

from tests.environments.tworoutes.run_fixtures import TwoRoutesRun

pytestmark = pytest.mark.xdist_group("tworoutes_method_deployment")

DP = (
    "--goal-pursuit-horizon", "0", "--dp-budget", "30", "--dp-solver", "enumeration",
    "--dp-initial-competence", "RouteA=0.2", "--dp-initial-competence", "RouteB=0.1",
    "--evaluation-policy", "method",
)  # fmt: skip
OURS = (
    "--pomdp-solver", "determinized_astar", "--pomdp-determinized-variant", "curtis",
    "--pomdp-observation-probability-weight", "0.01", "--pomdp-inference-engine", "grid",
    "--evaluation-policy", "method",
)  # fmt: skip


def policy_kinds(*, output: Path) -> list[str]:
    evaluations = json.loads((output / "step_evaluations.json").read_text())
    assert all(e["complete"] for e in evaluations)
    return [e["deployment_policy"]["kind"] for e in evaluations]


@pytest.fixture(scope="module")
def dp_run(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return TwoRoutesRun.run(directory=tmp_path_factory.mktemp("dp"), method="dp", extra=DP)


def test_dp_plans_and_deploys_for_reward(*, dp_run: Path) -> None:
    assert policy_kinds(output=dp_run) == ["dp_mdp", "dp_mdp"]
    solves = [
        json.loads(line)
        for line in (dp_run / "dp_allocations.jsonl").read_text().splitlines()
        if "allocation" in json.loads(line) and "objective" in json.loads(line)
    ]
    # Untrained, the calibrated priors make Route B worth 3 * 0.1 and Route A 0.2, so a
    # first allocation that can only reach an objective above 1 is counting Route B's
    # reward: a success probability cannot exceed 1.
    assert solves[0]["objective"] > 1.0
    before = TwoRoutesRun.evaluation_actions(output=dp_run, index=0)
    assert {a["label"] for a in before} == {"RouteB(robot)"}
    assert len(before) == 8


def test_the_belief_space_method_runs_and_deploys_reward_greedy(
    *, tmp_path_factory: pytest.TempPathFactory
) -> None:
    output = TwoRoutesRun.run(
        directory=tmp_path_factory.mktemp("ours"), method="pomdp", budget=20, extra=OURS
    )
    assert policy_kinds(output=output) == ["reward_greedy", "reward_greedy"]
    for index in (0, 1):
        actions = TwoRoutesRun.evaluation_actions(output=output, index=index)
        assert sorted(a["task"] for a in actions) == list(range(8))
    # Untrained, both routes sit at the Beta(1, 3) prior, so the better-paid one wins.
    assert {a["label"] for a in TwoRoutesRun.evaluation_actions(output=output, index=0)} == {
        "RouteB(robot)"
    }
    assert json.loads((output / "status.json").read_text())["practice_steps"] == 20


def test_ees_still_deploys_for_success_alone(*, tmp_path_factory: pytest.TempPathFactory) -> None:
    output = TwoRoutesRun.run(
        directory=tmp_path_factory.mktemp("ees"),
        method="ees",
        budget=20,
        extra=("--evaluation-policy", "method"),
    )
    assert policy_kinds(output=output) == ["shared_ees", "shared_ees"]
