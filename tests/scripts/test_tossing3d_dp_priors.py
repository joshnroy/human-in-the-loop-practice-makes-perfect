import json

import pytest

from scripts.tossing3d_dp_priors import (
    CHECKPOINTS,
    FIT_SEEDS,
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
            "protocol": Tossing3DDpPriors.protocol(),
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
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)
    toss = report["skills"][TOSS]
    probability = (6 + 0.5) / (48 + 1)
    assert toss["alpha"] / (toss["alpha"] + toss["beta"]) == pytest.approx(probability)
    assert 0 < toss["initial_learning_rate"] < 0.02
    assert report["skills"][PICK]["initial_learning_rate"] == 0
    assert report["diagnostics"][TOSS]["predicted_saturation_rollouts"] > 2
    assert report["validation_passed"] is True
    assert report["uses_experiment_evaluations"] is False


def test_validation_outcomes_do_not_change_the_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)["skills"]
    for seed in VALIDATION_SEEDS:
        path = curves / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert Tossing3DDpPriors.fit(root=curves, open_root=open_probes)["skills"] == before


def test_no_observed_improvement_does_not_invent_a_positive_rate(*, curves, open_probes):
    for seed in FIT_SEEDS:
        path = curves / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["skill"] == TOSS and row["checkpoint"]:
                row["success"] = False
        path.write_text("\n".join(json.dumps(row) for row in rows))
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)
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
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes)


@pytest.fixture
def open_probes(*, curves):
    root = curves / "open-gripper"
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        directory = root / str(seed)
        directory.mkdir(parents=True)
        metadata = json.loads((curves / str(seed) / "metadata.json").read_text())
        metadata["protocol"] = Tossing3DDpPriors.open_protocol()
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
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)
    prior = report["skills"]["OpenGripper"]
    assert prior["alpha"] / (prior["alpha"] + prior["beta"]) == pytest.approx(45.5 / 49)
    assert prior["initial_learning_rate"] == 0
    assert report["diagnostics"]["OpenGripper"]["initial_probe_trials"] == 48
    assert report["diagnostics"]["OpenGripper"]["validation_probe_trials"] == 48


def test_open_validation_outcomes_do_not_change_any_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)["skills"]
    for seed in VALIDATION_SEEDS:
        path = open_probes / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert Tossing3DDpPriors.fit(root=curves, open_root=open_probes)["skills"] == before


def test_refit_preserves_old_collection_driver_provenance(*, curves, open_probes):
    driver = "scripts/tossing3d_dp_priors.py"
    old_hash = "a" * 64
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        path = curves / str(seed) / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["source_sha256"][driver] = old_hash
        path.write_text(json.dumps(metadata))
    report = Tossing3DDpPriors.fit(root=curves, open_root=open_probes)
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
    from hitl_pmp.config_snapshot import ConfigSnapshot

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
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes)
