import json
import sys

import pytest

from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration
from scripts import tossing3d_dp_priors
from scripts.tossing3d_dp_priors import (
    CHECKPOINTS,
    FIT_SEEDS,
    OPEN,
    PICK,
    PROBES,
    SIMULATOR_PINS,
    TOSS,
    VALIDATION_SEEDS,
    Tossing3DDpPriors,
)


@pytest.fixture
def curves(*, tmp_path):
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        output = tmp_path / str(seed)
        output.mkdir()
        metadata = {
            "complete": True,
            "seed": seed,
            "source_sha256": Tossing3DDpPriors.source_hashes(),
            "protocol": Tossing3DDpPriors.protocol(enable_bin_pouring=False),
            "config_snapshot": {
                "kindergarden_commit": SIMULATOR_PINS["kinder"],
                "kinder_models_commit": SIMULATOR_PINS["kinder_models"],
                "kindergarden_dirty": False,
                "kinder_models_dirty": False,
                "python_version": "test",
                "torch_version": "test",
                "numpy_version": "test",
            },
        }
        (output / "metadata.json").write_text(json.dumps(metadata))
        rows = []
        for checkpoint, successes in zip(CHECKPOINTS, [2, 7, 12], strict=True):
            for index in range(PROBES):
                rows.extend([
                    {
                        "phase": "probe",
                        "checkpoint": checkpoint,
                        "skill": PICK,
                        "success": True,
                    },
                    {
                        "phase": "probe",
                        "checkpoint": checkpoint,
                        "skill": TOSS,
                        "success": index < successes,
                    },
                ])
        (output / "rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (output / "step_evaluations.json").write_text("must not read experiment evaluations")
    return tmp_path


def test_fit_uses_initial_rate_and_per_rollout_improvement_without_old_beta_prior(
    *, curves, open_probes
):
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)
    toss = report["skills"][TOSS]
    probability = (6 + 0.5) / (48 + 1)
    assert toss["alpha"] / (toss["alpha"] + toss["beta"]) == pytest.approx(probability)
    assert 0 < toss["initial_learning_rate"] < 0.02
    assert report["skills"][PICK]["initial_learning_rate"] == 0
    assert report["diagnostics"][TOSS]["predicted_saturation_rollouts"] > 2
    assert report["validation_passed"] is True
    assert report["uses_experiment_evaluations"] is False


def test_validation_outcomes_do_not_change_the_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)[
        "skills"
    ]
    for seed in VALIDATION_SEEDS:
        path = curves / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)[
            "skills"
        ]
        == before
    )


def test_no_observed_improvement_does_not_invent_a_positive_rate(*, curves, open_probes):
    for seed in FIT_SEEDS:
        path = curves / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["skill"] == TOSS and row["checkpoint"]:
                row["success"] = False
        path.write_text("\n".join(json.dumps(row) for row in rows))
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)
    assert report["skills"][TOSS]["initial_learning_rate"] == 0
    assert report["diagnostics"][TOSS]["predicted_saturation_rollouts"] is None


@pytest.mark.parametrize("problem", ["incomplete", "stale", "missing_probes"])
def test_bad_collection_is_rejected(*, curves, open_probes, problem):
    directory = curves / str(FIT_SEEDS[0])
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    if problem == "incomplete":
        metadata["complete"] = False
    elif problem == "stale":
        metadata["source_sha256"] = {}
    else:
        path = directory / "rollouts.jsonl"
        path.write_text("")
    if problem != "missing_probes":
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)


@pytest.fixture
def open_probes(*, curves):
    root = curves / "open-gripper"
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        directory = root / str(seed)
        directory.mkdir(parents=True)
        metadata = json.loads((curves / str(seed) / "metadata.json").read_text())
        metadata["protocol"] = Tossing3DDpPriors.open_protocol(enable_bin_pouring=False)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        rows = [
            {
                "phase": "probe",
                "skill": "OpenGripper",
                "success": index != 0,
                "preconditions_satisfied": True,
                "closed_empty": True,
                "consultation": "no_sampler",
                "scene_seed": seed * 100 + index,
                "skill_error": None,
            }
            for index in range(PROBES)
        ]
        (directory / "rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return root


def test_fit_requires_independent_open_gripper_probes(*, curves):
    with pytest.raises(ValueError, match="OpenGripper.*probes"):
        Tossing3DDpPriors.fit(root=curves)


def test_open_gripper_prior_uses_actual_probes_and_zero_learning_rate(*, curves, open_probes):
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)
    prior = report["skills"]["OpenGripper"]
    assert prior["alpha"] / (prior["alpha"] + prior["beta"]) == pytest.approx(45.5 / 49)
    assert prior["initial_learning_rate"] == 0
    assert report["diagnostics"]["OpenGripper"]["initial_probe_trials"] == 48
    assert report["diagnostics"]["OpenGripper"]["validation_probe_trials"] == 48


def test_open_validation_outcomes_do_not_change_any_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)[
        "skills"
    ]
    for seed in VALIDATION_SEEDS:
        path = open_probes / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)[
            "skills"
        ]
        == before
    )


def test_refit_preserves_old_collection_driver_provenance(*, curves, open_probes):
    driver = "scripts/tossing3d_dp_priors.py"
    old_hash = "a" * 64
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        path = curves / str(seed) / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["source_sha256"][driver] = old_hash
        path.write_text(json.dumps(metadata))
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)
    assert report["collection_driver_sha256"] == [old_hash]
    assert report["source_sha256"][driver] == Tossing3DDpPriors.source_hashes()[driver]


@pytest.mark.parametrize("seed", [0, 5, 42])
def test_open_collection_refuses_experiment_and_unreserved_seeds(*, tmp_path, seed):
    output = tmp_path / "probe"
    with pytest.raises(ValueError, match="reserved"):
        Tossing3DDpPriors.collect_open(seed=seed, output=output)
    assert not output.exists()


def test_open_collection_checks_simulator_pins_before_creating_probe_artifacts(
    *, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ConfigSnapshot, "_repo_state", staticmethod(lambda **kwargs: ("newer", False))
    )
    output = tmp_path / "probe"
    with pytest.raises(ValueError, match="OpenGripper.*simulator mismatch"):
        Tossing3DDpPriors.collect_open(seed=FIT_SEEDS[0], output=output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["precondition", "empty", "stale", "incomplete", "sampler"])
def test_invalid_open_gripper_probes_are_rejected(*, curves, open_probes, problem):
    directory = open_probes / str(FIT_SEEDS[0])
    if problem in {"stale", "incomplete"}:
        path = directory / "metadata.json"
        metadata = json.loads(path.read_text())
        if problem == "stale":
            metadata["source_sha256"] = {}
        else:
            metadata["complete"] = False
        path.write_text(json.dumps(metadata))
    else:
        path = directory / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if problem == "precondition":
            rows[0]["closed_empty"] = False
        elif problem == "empty":
            rows = []
        else:
            rows[0]["consultation"] = "learned"
        path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)


def test_no_pouring_fit_is_source_bound_and_covers_exactly_the_configured_robot_skills(
    *, curves, open_probes
):
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)
    assert report["enable_bin_pouring"] is False
    assert report["protocol"]["enable_bin_pouring"] is False
    assert report["open_gripper_protocol"]["enable_bin_pouring"] is False
    assert set(report["skills"]) == {PICK, TOSS, OPEN}
    path = curves / "fresh-priors.json"
    path.write_text(json.dumps(report))
    calibration = DpPriorCalibration.load(path=path)
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(enable_bin_pouring=False))
    calibration.verify_robot_skills(skills=provider.skills())
    with pytest.raises(ValueError, match="PourCubeOutOfBin"):
        calibration.verify_robot_skills(
            skills=Tossing3DSkillProvider(env=Tossing3DEnvironment()).skills()
        )


@pytest.mark.parametrize("collection", ["curves", "open"])
@pytest.mark.parametrize("recorded_flag", [True, None])
def test_no_pouring_fit_rejects_mixed_or_unspecified_protocol_flags(
    *, curves, open_probes, collection, recorded_flag
):
    directory = curves if collection == "curves" else open_probes
    path = directory / str(FIT_SEEDS[0]) / "metadata.json"
    metadata = json.loads(path.read_text())
    if recorded_flag is None:
        metadata["protocol"].pop("enable_bin_pouring")
    else:
        metadata["protocol"]["enable_bin_pouring"] = recorded_flag
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="protocol"):
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, enable_bin_pouring=False)


def test_default_fit_still_requires_a_pouring_prior(*, curves, open_probes):
    for root in (curves, open_probes):
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
            path = root / str(seed) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["protocol"]["enable_bin_pouring"] = True
            path.write_text(json.dumps(metadata))
    with pytest.raises(
        ValueError, match="robot skills lack calibrated DP priors.*PourCubeOutOfBin"
    ):
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes)


@pytest.mark.parametrize("collector", ["collect", "collect_open"])
@pytest.mark.parametrize("enable_bin_pouring", [True, False])
def test_collectors_preserve_the_default_and_accept_explicit_no_pouring(
    *, monkeypatch, tmp_path, collector, enable_bin_pouring
):
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("stop before simulator probe")

    monkeypatch.setattr(tossing3d_dp_priors, "EesMethod", build)
    monkeypatch.setattr(
        ConfigSnapshot,
        "_repo_state",
        staticmethod(lambda **kwargs: (SIMULATOR_PINS[kwargs["package"]], False)),
    )
    arguments = {"seed": FIT_SEEDS[0], "output": tmp_path / collector}
    if not enable_bin_pouring:
        arguments["enable_bin_pouring"] = False
    with pytest.raises(RuntimeError, match="stop before simulator probe"):
        getattr(Tossing3DDpPriors, collector)(**arguments)
    assert captured[0]["env"].enable_bin_pouring is enable_bin_pouring
    names = {skill.name for skill in captured[0]["skill_provider"].skills()}
    assert names == (
        {PICK, TOSS, OPEN, "PourCubeOutOfBin"} if enable_bin_pouring else {PICK, TOSS, OPEN}
    )


@pytest.mark.parametrize("command", ["collect", "collect-open", "fit"])
@pytest.mark.parametrize("enable_bin_pouring", [True, False])
def test_cli_forwards_the_pouring_flag_for_each_command(
    *, monkeypatch, tmp_path, capsys, command, enable_bin_pouring
):
    captured = []

    def run(**kwargs):
        captured.append(kwargs)
        return {"validation_passed": True}

    monkeypatch.setattr(Tossing3DDpPriors, command.replace("-", "_"), staticmethod(run))
    arguments = ["tossing3d_dp_priors", command, "--output", str(tmp_path / "output.json")]
    if command == "fit":
        arguments.extend(["--root", str(tmp_path), "--open-root", str(tmp_path / "open")])
    else:
        arguments.extend(["--seed", str(FIT_SEEDS[0])])
    if not enable_bin_pouring:
        arguments.append("--no-bin-pouring")
    monkeypatch.setattr(sys, "argv", arguments)
    tossing3d_dp_priors.main()
    assert captured[0]["enable_bin_pouring"] is enable_bin_pouring
    capsys.readouterr()


def test_protocol_toggle_keeps_the_reserved_seed_and_probe_schedule():
    assert FIT_SEEDS == (2000, 2001, 2002)
    assert VALIDATION_SEEDS == (2010, 2011, 2012)
    for protocol in (Tossing3DDpPriors.protocol, Tossing3DDpPriors.open_protocol):
        default = protocol()
        disabled = protocol(enable_bin_pouring=False)
        assert default["enable_bin_pouring"] is True
        assert disabled == {**default, "enable_bin_pouring": False}
    assert Tossing3DDpPriors.protocol()["checkpoints"] == [0, 20, 60]
    assert Tossing3DDpPriors.protocol()["probes_per_checkpoint"] == 16
    assert Tossing3DDpPriors.open_protocol()["probes_per_seed"] == 16
