import hashlib
import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.tossing3d_dp_launch import CALIBRATION_SEEDS, Tossing3DDpLaunch
from scripts.tossing3d_dp_priors import Tossing3DDpPriors


@pytest.fixture
def priors(*, tmp_path):
    path = tmp_path / "independent-priors.json"
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "environment": "tossing3d",
            "layout": "barrier",
            "variant": "o1",
            "skills": {
                "PickCube": {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0},
                "MoveToTossLocationAndToss": {
                    "alpha": 0.1,
                    "beta": 1.9,
                    "initial_learning_rate": 0.005,
                },
                "OpenGripper": {"alpha": 1.9, "beta": 0.1, "initial_learning_rate": 0},
            },
            "fit_seeds": [2000, 2001, 2002],
            "validation_seeds": [2010, 2011, 2012],
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "validation_passed": True,
            "uses_experiment_evaluations": False,
        })
    )
    return path


def _report(*, priors):
    return {
        "target_cost": 2000,
        "source_sha256": Tossing3DDpLaunch.source_hashes(),
        "prior_calibration_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
        "arms": {"plain": {"episode_budget": 250}, "always_ask": {"episode_budget": 100}},
    }


def test_full_matrix_uses_gurobi_and_explicit_units_without_submitting(*, tmp_path, priors):
    metadata = Tossing3DDpLaunch.prepare(
        directory=tmp_path,
        phase="full",
        tag="test-dp",
        code_key="test-snapshot",
        report=_report(priors=priors),
        prior_calibration=priors,
    )
    assert len(metadata["records"]) == 24
    assert metadata["submitted"] is False
    for record in metadata["records"]:
        args = Cli.parse_args(argv=record["argv"])
        assert args.dp_solver == "gurobi" and args.ees_score_human_skills
        assert args.practice_step_budget == 160000
        assert args.seed in range(6)
        if args.dp_budget_unit == "episodes":
            assert args.dp_allocation_schedule == "once"
            assert args.dp_budget == (100 if args.always_ask_reset else 250)
        else:
            assert args.dp_budget == 2000 and args.dp_allocation_schedule == "period"
        assert "--sampler-max-train-iters" in record["argv"]
        assert args.dp_prior_calibration.name == "prior-calibration.json"
    assert (tmp_path / "prior-calibration.json").read_bytes() == priors.read_bytes()
    sbatch = (tmp_path / "array-dp.sbatch").read_text()
    assert "module load gurobi/13.0.0" in sbatch
    assert "DP_GUROBI_PYTHONPATH:?" in sbatch
    assert "#SBATCH --array=1-24" in sbatch


@pytest.mark.parametrize("phase", ["full", "smoke"])
def test_missing_or_stale_calibration_is_refused(*, phase, priors):
    with pytest.raises(ValueError, match="calibration"):
        Tossing3DDpLaunch.records(
            phase=phase, tag="test", code_key="test", report=None, prior_calibration=priors
        )
    report = _report(priors=priors)
    report["source_sha256"] = {}
    with pytest.raises(ValueError, match="stale"):
        Tossing3DDpLaunch.records(
            phase=phase, tag="test", code_key="test", report=report, prior_calibration=priors
        )


def test_calibration_uses_reserved_seeds_and_only_cost_arm(*, priors):
    records = Tossing3DDpLaunch.records(
        phase="calibration",
        tag="test",
        code_key="test",
        report=None,
        prior_calibration=priors,
    )
    assert len(records) == 6
    assert {record["seed"] for record in records} == set(CALIBRATION_SEEDS)
    for record in records:
        args = Cli.parse_args(argv=record["argv"])
        assert args.dp_budget_unit == "cost" and args.dp_allocation_schedule == "period"
        assert args.dp_budget == 2000 and args.practice_step_budget == 3200


def test_validation_requests_production_duration_not_the_two_job_test_queue(*, tmp_path, priors):
    Tossing3DDpLaunch.prepare(
        directory=tmp_path,
        phase="calibration",
        tag="test",
        code_key="test",
        report=None,
        prior_calibration=priors,
    )
    sbatch = (tmp_path / "array-dp.sbatch").read_text()
    assert "#SBATCH --time=01:02:00" in sbatch
    assert "#SBATCH --qos=" not in sbatch


def _pilot_metadata(*, tmp_path, priors):
    metadata = {
        "phase": "calibration",
        "source_sha256": Tossing3DDpLaunch.source_hashes(),
        "records": [],
        "prior_calibration_sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
    }
    for always_ask in [False, True]:
        for seed in CALIBRATION_SEEDS:
            arm = "dp-cost-period" + ("-always-ask" if always_ask else "")
            output = tmp_path / arm / str(seed)
            output.mkdir(parents=True)
            cost = 100 if always_ask else 50
            (output / "status.json").write_text(
                json.dumps({"phase": "complete", "practice_steps": 3200, "physical_cost": cost})
            )
            (output / "dp_allocations.jsonl").write_text(
                json.dumps({"event": "allocation", "solver": "gurobi", "status": "OPTIMAL"})
                + "\n"
                + json.dumps({"event": "period_end", "episodes": 10, "physical_cost": cost})
                + "\n"
            )
            metadata["records"].append({"output": str(output), "arm": arm, "seed": seed})
    return metadata


def test_calibration_matches_cost_not_evaluation_outcomes(*, tmp_path, priors):
    report = Tossing3DDpLaunch.calibrate(metadata=_pilot_metadata(tmp_path=tmp_path, priors=priors))
    assert report["arms"]["plain"]["episode_budget"] == 400
    assert report["arms"]["always_ask"]["episode_budget"] == 200
    assert report["uses_evaluation_scores"] is False


def test_incomplete_or_nonoptimal_pilot_is_not_usable(*, tmp_path, priors):
    metadata = _pilot_metadata(tmp_path=tmp_path, priors=priors)
    output = Path(metadata["records"][0]["output"])
    (output / "dp_allocations.jsonl").write_text(
        json.dumps({"event": "allocation", "solver": "gurobi", "status": "TIME_LIMIT"}) + "\n"
    )
    with pytest.raises(ValueError, match="certified"):
        Tossing3DDpLaunch.calibrate(metadata=metadata)


def test_cost_calibration_from_old_priors_is_refused(*, priors):
    report = _report(priors=priors)
    report["prior_calibration_sha256"] = "different-priors"
    with pytest.raises(ValueError, match="skill priors differ"):
        Tossing3DDpLaunch.records(
            phase="full", tag="test", code_key="test", report=report, prior_calibration=priors
        )


@pytest.mark.parametrize("phase", ["calibration", "smoke", "full"])
@pytest.mark.parametrize("missing", ["PickCube", "MoveToTossLocationAndToss", "OpenGripper"])
def test_launcher_rejects_incomplete_shared_skill_coverage(*, phase, missing, priors):
    report = json.loads(priors.read_text())
    del report["skills"][missing]
    priors.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=f"robot skills lack calibrated DP priors.*{missing}"):
        Tossing3DDpLaunch.records(
            phase=phase,
            tag="test",
            code_key="test",
            report=None if phase == "calibration" else _report(priors=priors),
            prior_calibration=priors,
        )
