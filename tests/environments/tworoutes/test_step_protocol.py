import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.tworoutes.environment import TwoRoutesEnvironment
from hitl_pmp.environments.tworoutes.step_protocol import TwoRoutesStepProtocol
from tests.environments.tworoutes.run_fixtures import TwoRoutesRun

pytestmark = pytest.mark.xdist_group("tworoutes_step_protocol")


@pytest.fixture(scope="module")
def ees_run(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return TwoRoutesRun.run(
        directory=tmp_path_factory.mktemp("ees"), method="ees", budget=40, human_cost=7.0
    )


def test_ees_practises_to_the_budget_and_is_evaluated_twice(*, ees_run: Path) -> None:
    status = json.loads((ees_run / "status.json").read_text())
    assert status["phase"] == "complete"
    assert status["practice_steps"] == 40
    evaluations = json.loads((ees_run / "step_evaluations.json").read_text())
    assert [e["practice_steps"] for e in evaluations] == [0, 40]
    assert all(e["complete"] and e["num_total"] == 8 for e in evaluations)


def test_every_evaluation_task_is_one_route_attempt(*, ees_run: Path) -> None:
    for index in (0, 1):
        actions = TwoRoutesRun.evaluation_actions(output=ees_run, index=index)
        assert sorted(a["task"] for a in actions) == list(range(8))
        assert all(a["label"].startswith(("RouteA", "RouteB")) for a in actions)


def test_evaluation_runs_at_the_competence_practice_had_reached(*, ees_run: Path) -> None:
    before = TwoRoutesRun.evaluation_actions(output=ees_run, index=0)
    assert all(a["skill_trace"]["attempts"] == {"RouteA": 0, "RouteB": 0} for a in before)
    practised = json.loads((ees_run / "tworoutes_attempts.jsonl").read_text().splitlines()[-1])
    after = TwoRoutesRun.evaluation_actions(output=ees_run, index=1)
    assert sum(practised["attempts"].values()) > 0
    assert all(a["skill_trace"]["attempts"] == practised["attempts"] for a in after)


def test_every_human_reset_is_charged_the_cost_files_price(*, ees_run: Path) -> None:
    humans = [e for e in TwoRoutesRun.events(output=ees_run) if e["event"] == "human"]
    robots = [e for e in TwoRoutesRun.events(output=ees_run) if e["event"] == "robot"]
    assert all(e["skill"] == "human_reset_b" and e["cost"] == 7.0 for e in humans)
    assert all(e["end_step"] - e["start_step"] == 1 and e["cost"] == 1.0 for e in robots)
    route_b = [e for e in robots if e["label"].startswith("RouteB")]
    # Route B is only ever attempted from ready, and only a human makes it ready again.
    assert len(route_b) - 1 <= len(humans) <= len(route_b)


def test_a_route_attempt_is_one_shot_and_a_reset_is_not() -> None:
    env = TwoRoutesEnvironment()
    reason = TwoRoutesStepProtocol.one_shot_failure_reason
    assert reason(action=env.action_for(skill="RouteA")) == "route_failed"
    assert reason(action=env.action_for(skill="RouteB")) == "route_failed"
    assert reason(action=env.action_for(skill="ResetA")) is None
    assert reason(action=env.noop_action()) is None
    assert reason(action=np.zeros(3)) is None


def test_a_missing_measurement_is_an_error(*, tmp_path: Path) -> None:
    (tmp_path / "step_events.jsonl").write_text(
        '{"event": "measurement", "index": 0, "robot_invocations": 4}\n'
    )
    assert TwoRoutesStepProtocol.robot_invocations_at(output=tmp_path, checkpoint=0) == 4
    with pytest.raises(ValueError, match="No measurement 1"):
        TwoRoutesStepProtocol.robot_invocations_at(output=tmp_path, checkpoint=1)
