"""Completion requires independent physical logs to reconcile with harness metrics."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def completed_arm(*, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "sweep_completion",
        Path(__file__).resolve().parents[2] / "scripts/run_sweep_verified_arm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "pomdp/0"
    (root / "practice").mkdir(parents=True)
    (root / "evaluation").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                deployment_horizon=10,
                arms=[
                    dict(
                        name="pomdp-low",
                        method="pomdp",
                        human_reset=True,
                        human_reset_practice_cost=1.0,
                    )
                ],
            )
        )
    )
    stats = dict(
        evaluations=[[i, 0, 10] for i in range(51)],
        practice_session_ends=[
            dict(cycle_index=i, actions_executed=2, action_limit=20) for i in range(50)
        ],
        breakdowns=[
            dict(outcomes=[dict(task_index=i, solved=False) for i in range(10)]) for _ in range(51)
        ],
        num_practice_resets=0,
        num_human_interventions_recorded=50,
        summed_human_cost_recorded=50.0,
    )
    (root / "stats.json").write_text(json.dumps(stats))
    (root / "timing.json").write_text(json.dumps(dict(succeeded=True, returncode=0)))
    args = dict(
        num_cycles="50",
        max_steps_per_interaction="20",
        num_test_tasks="10",
        practice_reset_policy="never",
        goal_pursuit_horizon="0",
        method="pomdp",
        canonical_seed="0",
        seed="0",
        no_human_reset="False",
        sweep_manifest=str(manifest),
        human_reset_practice_cost="1.0",
    )
    (root / "config_snapshot.json").write_text(
        json.dumps(
            dict(
                args=args,
                git_commit="a" * 40,
                git_dirty=False,
                kindergarden_dirty=False,
                kinder_models_dirty=False,
            )
        )
    )
    events = [dict(kind="episode_start", seed=0, evaluation=False, validation=dict(valid=True))]
    for i in range(1, 51):
        events.extend([
            dict(kind="action", index=i),
            dict(kind="human_reset", index=i, success=True, validation=dict(valid=True)),
        ])
    (root / "practice/sweep_events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    (root / "evaluation/sweep_events.jsonl").write_text(
        (
            json.dumps(dict(kind="episode_start", evaluation=True, validation=dict(valid=True)))
            + "\n"
        )
        * 510
    )
    (root / "episode_traces.jsonl").write_text(
        json.dumps(dict(checkpoint=0, task_index=0, step_index=0)) + "\n"
    )
    return (
        module.SweepCompletion,
        root,
        dict(
            output=tmp_path,
            manifest_path=manifest,
            arm_name="pomdp-low",
            seed=0,
            revision="a" * 40,
            job_id="pomdp-low-s0",
        ),
    )


def test_reconciled_completion_includes_hashed_evidence(*, completed_arm):
    validator, _, arguments = completed_arm
    report = validator.validate(**arguments)
    assert report["status"] == "PASS"
    assert report["human_resets"] == 50
    assert report["robot_actions"] == 50
    assert len(report["artifacts"]) == 7
    assert all(len(a["sha256"]) == 64 for a in report["artifacts"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_practice_resets", 1),
        ("num_human_interventions_recorded", 49),
        ("summed_human_cost_recorded", 49.0),
    ],
)
def test_mismatched_accounting_fails(*, completed_arm, field, value):
    validator, root, arguments = completed_arm
    path = root / "stats.json"
    stats = json.loads(path.read_text())
    stats[field] = value
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError):
        validator.validate(**arguments)


def test_unreported_physical_reset_fails(*, completed_arm):
    validator, root, arguments = completed_arm
    with (root / "practice/sweep_events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(kind="episode_start", seed=0, evaluation=False)) + "\n")
    with pytest.raises(ValueError, match="restarted"):
        validator.validate(**arguments)


def test_truncated_run_is_not_completion(*, completed_arm):
    validator, root, arguments = completed_arm
    path = root / "stats.json"
    stats = json.loads(path.read_text())
    stats["evaluations"].pop()
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="51 complete"):
        validator.validate(**arguments)
