"""`--env tossing3d_real`: the real loop's wiring, end to end, on the fake plant."""

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real.cli import (
    END_ONLY_MEASUREMENT_INTERVAL,
    REAL_EVENTS_FILENAME,
    RealWorldRegistry,
    Tossing3DRealCli,
)
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.human import RealHumanOracle
from hitl_pmp.environments.tossing3d.real.step_protocol import InProcessEvaluations
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS

pytestmark = pytest.mark.xdist_group("tossing3d_real_cli")


@pytest.fixture(autouse=True)
def _fresh_registry():
    RealWorldRegistry.clear()
    yield
    RealWorldRegistry.clear()


def _args(*, output: Path, extra: tuple[str, ...] = ()) -> argparse.Namespace:
    return Cli.parse_args(
        argv=[
            "--env",
            "tossing3d_real",
            "--method",
            "ees",
            "--robot-plant",
            "fake",
            "--output-dir",
            str(output),
            "--defer-rendering",
            *extra,
        ]
    )


def test_the_real_cli_defaults_to_evaluating_at_the_end_only(*, tmp_path: Path) -> None:
    args = _args(output=tmp_path)
    assert args.measurement_interval_steps == END_ONLY_MEASUREMENT_INTERVAL
    assert args.initial_measurement is False
    assert args.checkpoint_sessions is True
    assert args.robot_plant == "fake"


def test_the_sim_cli_keeps_its_own_defaults() -> None:
    args = Cli.parse_args(argv=["--env", "tossing3d", "--method", "ees"])
    assert args.measurement_interval_steps == 1700
    assert args.initial_measurement is True


def test_practice_and_evaluation_share_one_world_and_the_real_set(*, tmp_path: Path) -> None:
    args = _args(output=tmp_path)
    practice = Tossing3DRealCli.build_practice_problem(args=args)
    evaluation = Tossing3DRealCli.build_evaluation_problem(args=args)
    assert isinstance(practice.env, RealTossing3DEnvironment)
    assert isinstance(evaluation.env, RealTossing3DEnvironment)
    assert practice.env.world is evaluation.env.world
    assert (practice.env.owner, evaluation.env.owner) == ("practice", "evaluation")
    assert practice.env.toss_parameters == REAL_TOSS_PARAMETERS
    assert practice.human is RealHumanOracle


def test_drawing_test_tasks_asks_nobody_for_anything(*, tmp_path: Path) -> None:
    args = _args(output=tmp_path)
    evaluation = Tossing3DRealCli.build_evaluation_problem(args=args)
    evaluation.hard_reset()
    tasks = [evaluation.sample_test_task() for _ in range(3)]
    placer = evaluation.env.world.placer
    assert placer.requests() == []
    evaluation.reset_to_task(task=tasks[1])
    assert [r.kind for r in placer.requests()] == ["scene"]
    assert placer.requests()[0].seed == int(
        tasks[1].initial_state.get(obj=evaluation.env.scene, feature_name="seed")
    )


def test_the_robot_client_is_imported_by_path_and_given_the_host(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict] = []
    module = types.ModuleType("fake_skill_client")
    module.Client = lambda **kwargs: made.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "fake_skill_client", module)
    RealWorldRegistry.client(spec="fake_skill_client:Client", host="nuc", port=None)
    RealWorldRegistry.client(spec="fake_skill_client:Client", host="nuc", port=7)
    assert made == [{"host": "nuc"}, {"host": "nuc", "port": 7}]


def test_in_process_evaluations_complete_on_submit() -> None:
    pool = InProcessEvaluations()
    assert pool.submit(fn=lambda *, x: x + 1, x=1).result() == 2

    def fails(**kwargs) -> None:
        raise RuntimeError("evaluation failed")

    assert isinstance(pool.submit(fn=fails).exception(), RuntimeError)


@pytest.mark.parametrize("method", ["ees", "pomdp"])
def test_the_full_step_protocol_runs_through_the_real_backend(
    *, method: str, tmp_path: Path
) -> None:
    Cli.main(
        argv=[
            "--env",
            "tossing3d_real",
            "--method",
            method,
            "--robot-plant",
            "fake",
            "--practice-step-budget",
            "800",
            "--practice-reset-policy",
            "never",
            "--num-test-tasks",
            "2",
            "--evaluation-control-steps",
            "500",
            "--output-dir",
            str(tmp_path),
            "--defer-rendering",
        ]
    )
    evaluations = json.loads((tmp_path / "step_evaluations.json").read_text())
    # End only: one measurement, at the end of practice.
    assert len(evaluations) == 1
    assert evaluations[0]["complete"] is True
    assert evaluations[0]["num_total"] == 2
    events = [
        json.loads(line) for line in (tmp_path / "step_events.jsonl").read_text().splitlines()
    ]
    # Whether practice acts is the planner's choice (Model B may STOP at these
    # priors); every session either acted or stopped, and nothing failed.
    assert {e["event"] for e in events} >= {"session_start", "practice_end"}
    assert not (tmp_path / "failure.txt").exists()
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "complete"
    # The deployed policy did drive the robot through the real backend.
    evaluated = [
        json.loads(line)
        for line in (tmp_path / "evaluations" / "0000" / "actions.jsonl").read_text().splitlines()
    ]
    assert any(a["label"].startswith("PickCube") for a in evaluated)
    assert any(a["label"].startswith("MoveToTossLocationAndToss") for a in evaluated)
    real_events = [
        json.loads(line) for line in (tmp_path / REAL_EVENTS_FILENAME).read_text().splitlines()
    ]
    kinds = [e.get("kind") for e in real_events if e["event"] == "placement"]
    # The practice scene, then one per evaluation task; a restore only if practice
    # needed the room again after evaluation, which an end-only run never does.
    assert kinds[0] == "scene"
    assert kinds.count("scene") == 1 + 2
    assert (tmp_path / "stats.json").exists()
