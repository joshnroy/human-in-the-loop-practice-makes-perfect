import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration
from scripts import tossing3d_shared_learning_validation as validation
from scripts.tossing3d_dp_priors import SIMULATOR_PINS
from scripts.tossing3d_launch import DELLA, PILOT, Tossing3DLaunch
from scripts.tossing3d_shared_learning_validation import (
    ALGORITHMS,
    GROUP,
    ROBOT_SKILLS,
    VARIANTS,
    Tossing3DSharedLearningValidation,
    main,
)

SHA = "a" * 40


@pytest.fixture
def evidence(*, tmp_path):
    source = validation.REPO / "scripts/tossing3d_launch.py"
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps({
            "schema_version": 1,
            "environment": "tossing3d",
            "layout": "barrier",
            "variant": "o1",
            "skills": {
                name: {"alpha": 1.0, "beta": 1.0, "initial_learning_rate": 0.0}
                for name in (
                    "PickCube",
                    "MoveToTossLocationAndToss",
                    "OpenGripper",
                )
            },
            "validation_passed": True,
            "uses_experiment_evaluations": False,
            "fit_seeds": [2000, 2001, 2002],
            "validation_seeds": [2010, 2011, 2012],
            "source_sha256": {
                "scripts/tossing3d_launch.py": hashlib.sha256(source.read_bytes()).hexdigest()
            },
            "simulator_pins": SIMULATOR_PINS,
        })
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({
            "target_cost": 2000.0,
            "source_sha256": {"historical-algorithm.py": "b" * 64},
            "prior_calibration_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
            "uses_evaluation_scores": False,
            "arms": {"plain": {"episode_budget": 350}},
        })
    )
    return {"calibration_report": report, "prior_calibration": prior}


@pytest.fixture
def refreshed_evidence(*, evidence):
    budget_prior = evidence["prior_calibration"]
    current = json.loads(budget_prior.read_text())
    historical = json.loads(budget_prior.read_text())
    historical["source_sha256"]["scripts/tossing3d_launch.py"] = "0" * 64
    budget_prior.write_text(json.dumps(historical))
    report = json.loads(evidence["calibration_report"].read_text())
    report["prior_calibration_sha256"] = hashlib.sha256(budget_prior.read_bytes()).hexdigest()
    evidence["calibration_report"].write_text(json.dumps(report))
    current["skills"]["MoveToTossLocationAndToss"]["alpha"] = 1.5
    current_prior = budget_prior.with_name("refreshed-prior.json")
    current_prior.write_text(json.dumps(current))
    return {
        **evidence,
        "prior_calibration": current_prior,
        "budget_prior_calibration": budget_prior,
    }


def test_complete_matrix_parses_with_the_real_cli_and_preserves_the_ees_template():
    records = Tossing3DSharedLearningValidation.records(tag="validation", sha=SHA)
    assert len(records) == 30
    assert {(record["variant"], record["algorithm"], record["seed"]) for record in records} == {
        (variant, algorithm, seed)
        for variant in VARIANTS
        for algorithm in ALGORITHMS
        for seed in (0, 1)
    }
    for record in records:
        args = Cli.parse_args(argv=record["argv"])
        assert args.practice_step_budget == 6400
        assert args.measurement_interval_steps == 3200
        assert args.num_test_tasks == 10 and args.evaluation_control_steps == 500
        assert args.stop_after_perfect_evaluations == 0
        assert args.ees_score_human_skills and args.record_skill_competence
        assert args.record_learning_forecasts
        assert (args.competence_sharing, args.practice_groundings) == VARIANTS[record["variant"]]
        assert not args.always_ask_reset and args.practice_reset_policy == "never"
        assert args.layout == args.evaluation_layout == "barrier"
        assert args.bin_mass_kg == 1.0 and args.cube_mass_kg == 0.1
        assert args.variant == "o1" and not args.enable_bin_pouring
        assert "--no-bin-pouring" in record["argv"]
        assert args.sampler_max_train_iters == 10000 and args.exploration_epsilon == 0.5
        assert args.record_sampler_draws
        template = Tossing3DLaunch.cli_argv(
            arm=Tossing3DLaunch.arms()["ees"],
            size=PILOT,
            seed=record["seed"],
            output_dir=Path(record["output"]),
            repo=Path(DELLA) / "code" / SHA,
        )
        template[template.index("--method") + 1] = args.method
        assert record["argv"][: len(template)] == template
        if record["algorithm"] == "ees":
            assert args.method == "ees" and "--dp-budget" not in record["argv"]
        else:
            assert args.method == "dp" and args.dp_solver == "gurobi"
            assert args.dp_prior_calibration == (
                Path(DELLA) / "launch/validation/prior-calibration.json"
            )
            if record["algorithm"] == "dp-episodes-once":
                assert (args.dp_budget_unit, args.dp_allocation_schedule, args.dp_budget) == (
                    "episodes",
                    "once",
                    350,
                )
            else:
                assert (args.dp_budget_unit, args.dp_allocation_schedule, args.dp_budget) == (
                    "cost",
                    "period",
                    2000,
                )


def test_subset_is_paired_across_all_three_algorithms():
    records = Tossing3DSharedLearningValidation.records(
        tag="subset", sha=SHA, variants=("flexible_only", "shared_rate"), seeds=(1,)
    )
    assert len(records) == 6
    assert {record["seed"] for record in records} == {1}
    assert {record["algorithm"] for record in records} == set(ALGORITHMS)


@pytest.mark.parametrize(
    "variant,sharing", [("shared_flexible", "parent"), ("shared_rate", "rate")]
)
def test_parent_or_rate_with_all_requests_flexible_context_choice(*, variant, sharing):
    records = Tossing3DSharedLearningValidation.records(
        tag="flexible", sha=SHA, variants=(variant,), seeds=(0,)
    )
    for record in records:
        args = Cli.parse_args(argv=record["argv"])
        assert args.competence_sharing == sharing and args.practice_groundings == "all"
        assert args.record_learning_forecasts


@pytest.mark.parametrize(
    "overrides",
    [
        {"phase": "full"},
        {"phase": "calibration"},
        {"sha": "HEAD"},
        {"sha": "a" * 7},
        {"tag": "../escape"},
        {"tag": "bad\trow"},
        {"variants": ()},
        {"variants": ("baseline", "baseline")},
        {"variants": ("unknown",)},
        {"seeds": ()},
        {"seeds": (0, 0)},
        {"seeds": (2,)},
        {"seeds": (-1,)},
    ],
)
def test_invalid_or_expanded_protocol_is_rejected(*, overrides):
    with pytest.raises(ValueError):
        Tossing3DSharedLearningValidation.records(**{"tag": "test", "sha": SHA, **overrides})


def test_prepare_writes_only_launch_artifacts_and_evidence_without_execution(
    *, tmp_path, evidence, monkeypatch
):
    def forbidden(*arguments, **keywords):
        raise AssertionError("generation must never execute, submit, or instantiate a simulator")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(validation, "Tossing3DEnvironment", forbidden)
    monkeypatch.setattr(DpPriorCalibration, "load", forbidden)
    output = tmp_path / "planned files"
    metadata = Tossing3DSharedLearningValidation.prepare(
        directory=output, tag="planned", sha=SHA, **evidence
    )
    assert metadata["submitted"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        f"array-{GROUP}.sbatch",
        "budget-prior-calibration.json",
        "calibration-report.json",
        f"manifest-{GROUP}.tsv",
        "prior-calibration.json",
    ]
    assert (output / "prior-calibration.json").read_bytes() == (
        evidence["prior_calibration"].read_bytes()
    )
    assert (output / "budget-prior-calibration.json").read_bytes() == (
        evidence["prior_calibration"].read_bytes()
    )
    assert (output / "calibration-report.json").read_bytes() == (
        evidence["calibration_report"].read_bytes()
    )
    assert metadata["current_prior_calibration_sha256"] == metadata["budget_reference_prior_digest"]
    lines = (output / f"manifest-{GROUP}.tsv").read_text().splitlines()
    assert len(lines) == 30
    for line, record in zip(lines, metadata["records"], strict=True):
        name, sha, arm, seed, directory, command = line.split("\t")
        assert (name, sha, arm, int(seed), directory) == (
            record["name"],
            SHA,
            record["arm"],
            record["seed"],
            record["output"],
        )
        assert shlex.split(command) == record["argv"]
    sbatch = (output / f"array-{GROUP}.sbatch").read_text()
    assert "#SBATCH --array=1-30" in sbatch and "#SBATCH --time=02:00:00" in sbatch
    assert "module load gurobi/13.0.0" in sbatch and "DP_GUROBI_PYTHONPATH:?" in sbatch
    assert "import zstandard" in sbatch and "ShelfLaunch.require_isolated()" in sbatch
    assert "LD_PRELOAD" in sbatch and "OMP_NUM_THREADS=1" in sbatch
    assert "scripts.tossing3d_launch worker" in sbatch
    assert f"--sha {SHA}" in sbatch and 'eval "ARGS=' not in sbatch
    assert sbatch.index("unexpected manifest SHA") < sbatch.index('CODE="')
    assert sbatch.index(" verify ") < sbatch.index("scripts.tossing3d_launch worker")
    assert metadata["calibration_report_sha256"] in sbatch
    assert metadata["prior_calibration_sha256"] in sbatch
    assert "/launch/planned/prior-calibration.json" in sbatch
    assert "/launch/planned/calibration-report.json" in sbatch
    assert "--budget-prior-calibration" in sbatch
    assert "/launch/planned/budget-prior-calibration.json" in sbatch


def test_main_reports_subset_and_parse_alone_has_no_writes(*, tmp_path, evidence, capsys):
    output = tmp_path / "generated"
    argv = [
        "prepare",
        "--sha",
        SHA,
        "--tag",
        "test",
        "--out",
        str(output),
        "--calibration-report",
        str(evidence["calibration_report"]),
        "--prior-calibration",
        str(evidence["prior_calibration"]),
        "--variants",
        "baseline",
        "shared_rate",
        "--seeds",
        "0",
    ]
    args = Tossing3DSharedLearningValidation.parse(argv=argv)
    assert args.phase == "validation" and args.variants == ["baseline", "shared_rate"]
    assert args.seeds == [0] and not output.exists()
    assert args.budget_prior_calibration is None
    assert main(argv=argv) == 0
    assert capsys.readouterr().out == "Prepared 6 validation tasks; submitted nothing\n"


@pytest.mark.parametrize(
    "argument",
    [
        ["--phase", "full"],
        ["--seeds", "2"],
        ["--variants", "unknown"],
        ["--submit"],
    ],
)
def test_cli_refuses_full_and_submission_options(*, argument):
    with pytest.raises(SystemExit):
        Tossing3DSharedLearningValidation.parse(
            argv=[
                "prepare",
                "--sha",
                SHA,
                "--tag",
                "test",
                "--out",
                "/unused",
                "--calibration-report",
                "/unused/report",
                "--prior-calibration",
                "/unused/prior",
                *argument,
            ]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_cost", 3000),
        ("arms", {"plain": {"episode_budget": 351}}),
        ("uses_evaluation_scores", True),
        ("source_sha256", {}),
        ("prior_calibration_sha256", "wrong"),
    ],
)
def test_bad_calibration_does_not_create_output(*, tmp_path, evidence, field, value):
    report = json.loads(evidence["calibration_report"].read_text())
    report[field] = value
    evidence["calibration_report"].write_text(json.dumps(report))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        Tossing3DSharedLearningValidation.prepare(directory=output, tag="test", sha=SHA, **evidence)
    assert not output.exists()


def _verify_arguments(*, evidence):
    manifest = evidence["prior_calibration"].with_name("runtime-manifest.tsv")
    manifest.write_text(
        "\t".join([
            "test-baseline-ees-s0",
            SHA,
            "baseline-ees",
            "0",
            "/unused",
            "--env tossing3d --method ees --output-dir /unused --no-bin-pouring",
        ])
        + "\n"
    )
    return {
        "sha": SHA,
        **evidence,
        "report_sha256": hashlib.sha256(evidence["calibration_report"].read_bytes()).hexdigest(),
        "prior_sha256": hashlib.sha256(evidence["prior_calibration"].read_bytes()).hexdigest(),
        "manifest": manifest,
        "task": 1,
    }


def _git(*, monkeypatch, dirty="", sha=SHA, pin=None):
    def output(*arguments, **keywords):
        command = arguments[0]
        if "rev-parse" in command:
            return sha + "\n"
        if "status" in command:
            assert "--untracked-files=all" in command and "--ignore-submodules=none" in command
            return dirty
        assert "ls-tree" in command
        package = "kinder" if command[-1] == "reference/kindergarden" else "kinder_models"
        return f"160000 commit {pin or SIMULATOR_PINS[package]}\t{command[-1]}\n"

    monkeypatch.setattr(subprocess, "check_output", output)


@pytest.mark.parametrize(
    "state",
    [
        {"dirty": " M src/changed.py\n"},
        {"dirty": "?? scripts/untracked.py\n"},
        {"sha": "b" * 40},
        {"pin": "c" * 40},
    ],
)
def test_runtime_refuses_dirty_wrong_commit_and_changed_pins(*, evidence, monkeypatch, state):
    _git(monkeypatch=monkeypatch, **state)
    with pytest.raises(ValueError, match="clean checkout|simulator pin"):
        Tossing3DSharedLearningValidation.verify(**_verify_arguments(evidence=evidence))


@pytest.mark.parametrize("key", ["calibration_report", "prior_calibration"])
def test_runtime_refuses_changed_deployed_evidence(*, evidence, monkeypatch, key):
    _git(monkeypatch=monkeypatch)
    arguments = _verify_arguments(evidence=evidence)
    evidence[key].write_text(evidence[key].read_text() + "\n")
    with pytest.raises(ValueError, match="digest changed"):
        Tossing3DSharedLearningValidation.verify(**arguments)


def test_runtime_checks_source_bound_priors_and_skill_coverage(*, evidence, monkeypatch):
    _git(monkeypatch=monkeypatch)
    checked = []
    monkeypatch.setattr(
        DpPriorCalibration, "verify_simulators", lambda self: checked.append("pins")
    )

    def environment(*, enable_bin_pouring):
        assert enable_bin_pouring is False
        return "environment"

    monkeypatch.setattr(validation, "Tossing3DEnvironment", environment)

    class Provider:
        def __init__(self, *, env):
            assert env == "environment"

        def skills(self):
            return tuple(SimpleNamespace(name=name) for name in sorted(ROBOT_SKILLS))

    def verify_skills(*arguments, skills):
        checked.append({skill.name for skill in skills})

    monkeypatch.setattr(validation, "Tossing3DSkillProvider", Provider)
    monkeypatch.setattr(DpPriorCalibration, "verify_robot_skills", verify_skills)
    Tossing3DSharedLearningValidation.verify(**_verify_arguments(evidence=evidence))
    assert checked == ["pins", ROBOT_SKILLS]


def test_disabled_pouring_preserves_exactly_the_original_robot_skill_names():
    provider = validation.Tossing3DSkillProvider(
        env=validation.Tossing3DEnvironment(enable_bin_pouring=False)
    )
    assert {skill.name for skill in provider.skills()} == {
        "PickCube",
        "MoveToTossLocationAndToss",
        "OpenGripper",
    }


def test_runtime_refuses_default_enabled_pouring(*, evidence, monkeypatch):
    _git(monkeypatch=monkeypatch)
    arguments = _verify_arguments(evidence=evidence)
    manifest = arguments["manifest"]
    manifest.write_text(manifest.read_text().replace(" --no-bin-pouring", ""))
    with pytest.raises(ValueError, match="--no-bin-pouring"):
        Tossing3DSharedLearningValidation.verify(**arguments)


def test_runtime_does_not_relax_stale_prior_source_binding(*, evidence, monkeypatch):
    _git(monkeypatch=monkeypatch)
    prior = json.loads(evidence["prior_calibration"].read_text())
    prior["source_sha256"]["scripts/tossing3d_launch.py"] = "0" * 64
    evidence["prior_calibration"].write_text(json.dumps(prior))
    report = json.loads(evidence["calibration_report"].read_text())
    report["prior_calibration_sha256"] = hashlib.sha256(
        evidence["prior_calibration"].read_bytes()
    ).hexdigest()
    evidence["calibration_report"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale prior calibration source"):
        Tossing3DSharedLearningValidation.verify(**_verify_arguments(evidence=evidence))


def test_refreshed_prior_and_historical_budget_prior_have_separate_digests(
    *, tmp_path, refreshed_evidence
):
    output = tmp_path / "refreshed-launch"
    metadata = Tossing3DSharedLearningValidation.prepare(
        directory=output, tag="refreshed", sha=SHA, **refreshed_evidence
    )
    current_digest = hashlib.sha256(
        refreshed_evidence["prior_calibration"].read_bytes()
    ).hexdigest()
    budget_digest = hashlib.sha256(
        refreshed_evidence["budget_prior_calibration"].read_bytes()
    ).hexdigest()
    assert current_digest != budget_digest
    assert metadata["prior_calibration_sha256"] == current_digest
    assert metadata["current_prior_calibration_sha256"] == current_digest
    assert metadata["budget_reference_prior_digest"] == budget_digest
    assert metadata["episode_budget"] == 350
    assert "not recalibrated" in metadata["budget_interpretation"]
    assert (output / "prior-calibration.json").read_bytes() == (
        refreshed_evidence["prior_calibration"].read_bytes()
    )
    assert (output / "budget-prior-calibration.json").read_bytes() == (
        refreshed_evidence["budget_prior_calibration"].read_bytes()
    )
    assert (
        json.loads((output / "calibration-report.json").read_text())["prior_calibration_sha256"]
        == budget_digest
    )
    sbatch = (output / f"array-{GROUP}.sbatch").read_text()
    assert f"--prior-sha256 {current_digest}" in sbatch
    for record in metadata["records"]:
        if record["algorithm"] != "ees":
            args = Cli.parse_args(argv=record["argv"])
            assert args.dp_prior_calibration == (
                Path(DELLA) / "launch/refreshed/prior-calibration.json"
            )


@pytest.mark.parametrize("runtime", [False, True])
def test_tampered_historical_budget_prior_is_refused(
    *, tmp_path, refreshed_evidence, monkeypatch, runtime
):
    _git(monkeypatch=monkeypatch)
    arguments = _verify_arguments(evidence=refreshed_evidence)
    budget = refreshed_evidence["budget_prior_calibration"]
    budget.write_text(budget.read_text() + "\n")
    with pytest.raises(ValueError, match="digest differs from the existing report"):
        if runtime:
            Tossing3DSharedLearningValidation.verify(**arguments)
        else:
            Tossing3DSharedLearningValidation.prepare(
                directory=tmp_path / "not-created", tag="test", sha=SHA, **refreshed_evidence
            )
    assert not (tmp_path / "not-created").exists()


def test_runtime_loads_only_refreshed_prior_and_keeps_historical_source_hashes(
    *, refreshed_evidence, monkeypatch
):
    _git(monkeypatch=monkeypatch)
    loaded = []
    original_load = DpPriorCalibration.load

    def load(*, path):
        loaded.append(path)
        return original_load(path=path)

    monkeypatch.setattr(DpPriorCalibration, "load", staticmethod(load))
    monkeypatch.setattr(DpPriorCalibration, "verify_simulators", lambda self: None)
    Tossing3DSharedLearningValidation.verify(**_verify_arguments(evidence=refreshed_evidence))
    assert loaded == [refreshed_evidence["prior_calibration"]]
    assert (
        json.loads(refreshed_evidence["budget_prior_calibration"].read_text())["source_sha256"][
            "scripts/tossing3d_launch.py"
        ]
        == "0" * 64
    )


def test_stale_refreshed_prior_still_fails_with_valid_historical_budget(
    *, refreshed_evidence, monkeypatch
):
    _git(monkeypatch=monkeypatch)
    current = refreshed_evidence["prior_calibration"]
    prior = json.loads(current.read_text())
    prior["source_sha256"]["scripts/tossing3d_launch.py"] = "1" * 64
    current.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="stale prior calibration source"):
        Tossing3DSharedLearningValidation.verify(**_verify_arguments(evidence=refreshed_evidence))


@pytest.mark.parametrize("invalid", ["failed-validation", "missing-skill"])
def test_current_prior_is_validated_independently_before_writing(
    *, tmp_path, refreshed_evidence, invalid
):
    current = refreshed_evidence["prior_calibration"]
    prior = json.loads(current.read_text())
    if invalid == "failed-validation":
        prior["validation_passed"] = False
    else:
        del prior["skills"]["OpenGripper"]
    current.write_text(json.dumps(prior))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        Tossing3DSharedLearningValidation.prepare(
            directory=output, tag="test", sha=SHA, **refreshed_evidence
        )
    assert not output.exists()


def test_budget_prior_option_is_available_for_prepare_and_verify(*, evidence, tmp_path):
    shared = [
        "--sha",
        SHA,
        "--calibration-report",
        str(evidence["calibration_report"]),
        "--prior-calibration",
        str(evidence["prior_calibration"]),
        "--budget-prior-calibration",
        str(tmp_path / "historical.json"),
    ]
    prepare = Tossing3DSharedLearningValidation.parse(
        argv=["prepare", "--tag", "test", "--out", str(tmp_path / "out"), *shared]
    )
    verify = Tossing3DSharedLearningValidation.parse(
        argv=[
            "verify",
            "--manifest",
            "/unused",
            "--task",
            "1",
            "--report-sha256",
            "r",
            "--prior-sha256",
            "p",
            *shared,
        ]
    )
    assert prepare.budget_prior_calibration == verify.budget_prior_calibration
    assert prepare.budget_prior_calibration == tmp_path / "historical.json"


def test_runtime_requires_dp_to_use_the_pinned_current_prior(*, evidence, monkeypatch):
    _git(monkeypatch=monkeypatch)
    arguments = _verify_arguments(evidence=evidence)
    manifest = arguments["manifest"]
    manifest.write_text(
        manifest.read_text().replace(
            "--method ees",
            "--method dp --dp-budget 350 --dp-prior-calibration /unverified/prior.json",
        )
    )
    with pytest.raises(ValueError, match="digest-pinned current prior"):
        Tossing3DSharedLearningValidation.verify(**arguments)
