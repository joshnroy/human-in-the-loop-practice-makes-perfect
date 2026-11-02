import json
import sys
from types import SimpleNamespace

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
from tests.environments.tossing3d.observations import HOLDING_ATOMS, INITIAL_ATOMS, state


@pytest.fixture
def curves(*, tmp_path):
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        output = tmp_path / str(seed)
        output.mkdir()
        metadata = {
            "complete": True,
            "seed": seed,
            "split": "fit" if seed in FIT_SEEDS else "validation",
            "compiled_masses_kg": {"bin_0": 1.0, "cube_0": 0.1},
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
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )
    toss = report["skills"][TOSS]
    probability = (6 + 0.5) / (48 + 1)
    assert toss["alpha"] / (toss["alpha"] + toss["beta"]) == pytest.approx(probability)
    assert 0 < toss["initial_learning_rate"] < 0.02
    assert report["skills"][PICK]["initial_learning_rate"] == 0
    assert report["diagnostics"][TOSS]["predicted_saturation_rollouts"] > 2
    assert report["validation_passed"] is True
    assert report["uses_experiment_evaluations"] is False


def test_validation_outcomes_do_not_change_the_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )["skills"]
    for seed in VALIDATION_SEEDS:
        path = curves / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
            if "add_effects_satisfied" in row:
                row["add_effects_satisfied"] = row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            open_aggregation="equal-context",
            enable_bin_pouring=False,
        )["skills"]
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
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )
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
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            open_aggregation="equal-context",
            enable_bin_pouring=False,
        )


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
                "closed_empty": context == "closed_empty",
                "holding": context == "holding",
                "context": context,
                "add_effects_satisfied": index != 0,
                "delete_effects_satisfied": True,
                "consultation": "no_sampler",
                "scene_seed": seed * 100 + index,
                "skill_error": None,
            }
            for context in ("closed_empty", "holding")
            for index in range(PROBES)
        ]
        (directory / "rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return root


def test_fit_requires_independent_open_gripper_probes(*, curves):
    with pytest.raises(ValueError, match="OpenGripper.*probes"):
        Tossing3DDpPriors.fit(root=curves)


def test_open_gripper_prior_uses_actual_probes_and_zero_learning_rate(*, curves, open_probes):
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )
    prior = report["skills"]["OpenGripper"]
    assert prior["alpha"] / (prior["alpha"] + prior["beta"]) == pytest.approx(45.5 / 49)
    assert prior["initial_learning_rate"] == 0
    assert report["diagnostics"]["OpenGripper"]["initial_probe_trials"] == 96
    assert report["diagnostics"]["OpenGripper"]["validation_probe_trials"] == 96


def test_open_validation_outcomes_do_not_change_any_fitted_prior(*, curves, open_probes):
    before = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )["skills"]
    for seed in VALIDATION_SEEDS:
        path = open_probes / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = not row["success"]
            if "add_effects_satisfied" in row:
                row["add_effects_satisfied"] = row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            open_aggregation="equal-context",
            enable_bin_pouring=False,
        )["skills"]
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
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )
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
    with pytest.raises(ValueError, match="prior calibration simulator mismatch"):
        Tossing3DDpPriors.collect_open(seed=FIT_SEEDS[0], output=output)
    assert not output.exists()


@pytest.mark.parametrize("collector", ["collect", "collect_open"])
@pytest.mark.parametrize("dirty", [False, True])
def test_both_collectors_reject_wrong_or_dirty_simulators(
    *, tmp_path, monkeypatch, collector, dirty
):
    monkeypatch.setattr(
        ConfigSnapshot,
        "_repo_state",
        staticmethod(
            lambda **kwargs: (SIMULATOR_PINS[kwargs["package"]] if dirty else "wrong", dirty)
        ),
    )
    output = tmp_path / "probes"
    with pytest.raises(ValueError, match="simulator mismatch"):
        getattr(Tossing3DDpPriors, collector)(seed=FIT_SEEDS[0], output=output)
    assert not output.exists()


@pytest.mark.parametrize("aggregation", [None, "pooled", "deployment"])
def test_fit_requires_explicit_supported_aggregation(*, curves, open_probes, aggregation):
    with pytest.raises(ValueError, match="explicit OpenGripper aggregation"):
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            enable_bin_pouring=False,
            open_aggregation=aggregation,
        )


def test_open_context_estimates_are_separate_and_scalar_cannot_mask_validation(
    *, curves, open_probes
):
    for seed in (*FIT_SEEDS, *VALIDATION_SEEDS):
        path = open_probes / str(seed) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["success"] = row["context"] == "closed_empty"
            row["add_effects_satisfied"] = row["success"]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        enable_bin_pouring=False,
        open_aggregation="equal-context",
    )
    diagnostic = report["diagnostics"][OPEN]
    assert diagnostic["contexts"]["closed_empty"]["initial_competence"] == pytest.approx(48.5 / 49)
    assert diagnostic["contexts"]["holding"]["initial_competence"] == pytest.approx(0.5 / 49)
    assert diagnostic["initial_competence"] == pytest.approx(0.5)
    assert report["skills"][OPEN] == {"alpha": 1.0, "beta": 1.0, "initial_learning_rate": 0.0}
    assert diagnostic["aggregation"]["weights"] == {"closed_empty": 0.5, "holding": 0.5}
    assert diagnostic["validation_brier"] < diagnostic["old_defaults_validation_brier"]
    assert diagnostic["contexts"]["closed_empty"]["validation_passed"] is False
    assert report["validation_passed"] is False


@pytest.mark.parametrize("context", ["closed_empty", "holding"])
@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "wrong_context", "holding", "effects", "error"]
)
def test_each_open_context_requires_valid_complete_unique_probes(
    *, curves, open_probes, context, problem
):
    path = open_probes / str(FIT_SEEDS[0]) / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [row for row in rows if row["context"] == context]
    if problem == "missing":
        rows.remove(selected[0])
    elif problem == "duplicate":
        selected[0]["scene_seed"] = selected[1]["scene_seed"]
    elif problem == "wrong_context":
        selected[0]["context"] = "unknown"
    elif problem == "holding":
        selected[0]["holding"] = not selected[0]["holding"]
    elif problem == "effects":
        selected[1]["delete_effects_satisfied"] = False
    else:
        selected[1]["skill_error"] = "open_gripper_settle_timeout"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            enable_bin_pouring=False,
            open_aggregation="equal-context",
        )


@pytest.mark.parametrize("collection", ["curves", "open"])
@pytest.mark.parametrize("problem", ["mass", "missing_mass", "split", "protocol_mass", "stack"])
def test_calibration_rejects_environment_and_split_drift(
    *, curves, open_probes, collection, problem
):
    root = curves if collection == "curves" else open_probes
    path = root / str(FIT_SEEDS[0]) / "metadata.json"
    metadata = json.loads(path.read_text())
    if problem == "mass":
        metadata["compiled_masses_kg"]["cube_0"] = 0.2
    elif problem == "missing_mass":
        metadata.pop("compiled_masses_kg")
    elif problem == "split":
        metadata["split"] = "validation"
    elif problem == "protocol_mass":
        metadata["protocol"]["bin_mass_kg"] = 2.0
    else:
        metadata["config_snapshot"]["numpy_version"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            enable_bin_pouring=False,
            open_aggregation="equal-context",
        )


@pytest.mark.parametrize("cube_mass", [0.1, 0.2])
def test_compiled_mass_check_reads_actual_model(*, cube_mass):
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id={"bin_0": 0, "cube_0": 1}[name]),
        body_subtreemass=[1.0, cube_mass],
    )
    scene = SimpleNamespace(
        _robot_env=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(mj_model=model)))
    )
    env = SimpleNamespace(kinder_backend=lambda: SimpleNamespace(_object_centric=lambda: scene))
    if cube_mass == 0.1:
        assert Tossing3DDpPriors.compiled_masses(env=env) == {"bin_0": 1.0, "cube_0": 0.1}
    else:
        with pytest.raises(ValueError, match="compiled mass mismatch"):
            Tossing3DDpPriors.compiled_masses(env=env)


@pytest.mark.parametrize("open_error", [None, "open_gripper_settle_timeout"])
def test_open_collector_retains_held_cube_and_resets_only_closed_empty(
    *, monkeypatch, tmp_path, open_error
):
    observations = {}
    resets = []
    actions = []
    closed = []

    class SimulatedActions:
        def reset(self, *, seed):
            observations["state"] = state(env=self, abstract_atoms=INITIAL_ATOMS)
            return observations["state"]

        def reset_movables(self):
            resets.append(True)
            observations["state"] = state(
                env=self, abstract_atoms=INITIAL_ATOMS - {("HandEmpty", ("robot",))}
            )
            return True

        def take_action(self, *, action):
            is_pick = action[0] == self.pick_cube_id
            actions.append("pick" if is_pick else "open")
            observations["state"] = state(
                env=self, abstract_atoms=HOLDING_ATOMS if is_pick else INITIAL_ATOMS
            )
            return observations["state"]

    monkeypatch.setattr(Tossing3DEnvironment, "reset_to_seed", SimulatedActions.reset)
    monkeypatch.setattr(Tossing3DEnvironment, "reset_movables", SimulatedActions.reset_movables)
    monkeypatch.setattr(Tossing3DEnvironment, "take_action", SimulatedActions.take_action)
    monkeypatch.setattr(
        Tossing3DEnvironment, "get_current_state", lambda self: observations["state"]
    )
    monkeypatch.setattr(
        Tossing3DEnvironment,
        "last_skill_error",
        lambda self: open_error if actions[-1] == "open" else None,
    )
    monkeypatch.setattr(Tossing3DEnvironment, "close", lambda self: closed.append(True))
    monkeypatch.setattr(
        Tossing3DDpPriors,
        "compiled_masses",
        staticmethod(lambda **kwargs: {"bin_0": 1.0, "cube_0": 0.1}),
    )
    monkeypatch.setattr(
        ConfigSnapshot,
        "_repo_state",
        staticmethod(lambda **kwargs: (SIMULATOR_PINS[kwargs["package"]], False)),
    )
    metadata = Tossing3DDpPriors.collect_open(
        seed=FIT_SEEDS[0], output=tmp_path / "open", enable_bin_pouring=False
    )
    assert metadata["complete"] is True
    assert metadata["elapsed_seconds"] > 0
    assert set(metadata["context_elapsed_seconds"]) == {"closed_empty", "holding"}
    assert len(resets) == PROBES
    assert actions == ["pick", "open"] * (2 * PROBES)
    assert closed == [True]
    rows = [
        json.loads(line) for line in (tmp_path / "open" / "rollouts.jsonl").read_text().splitlines()
    ]
    probes = [row for row in rows if row["phase"] == "probe"]
    assert len(probes) == 2 * PROBES
    for context in ("closed_empty", "holding"):
        selected = [row for row in probes if row["context"] == context]
        assert len(selected) == PROBES
        assert all(row["success"] is (open_error is None) for row in selected)
        assert all(row["closed_empty"] is (context == "closed_empty") for row in selected)
        assert all(row["holding"] is (context == "holding") for row in selected)
        assert all(row["consultation"] == "no_sampler" for row in selected)


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
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            open_aggregation="equal-context",
            enable_bin_pouring=False,
        )


def test_no_pouring_fit_is_source_bound_and_covers_exactly_the_configured_robot_skills(
    *, curves, open_probes
):
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        open_aggregation="equal-context",
        enable_bin_pouring=False,
    )
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
        Tossing3DDpPriors.fit(
            root=curves,
            open_root=open_probes,
            open_aggregation="equal-context",
            enable_bin_pouring=False,
        )


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
        Tossing3DDpPriors.fit(root=curves, open_root=open_probes, open_aggregation="equal-context")


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
    assert captured[0]["env"].layout.value == "barrier"
    assert captured[0]["env"].variant == "o1"
    assert captured[0]["env"].bin_mass_kg == 1.0
    assert captured[0]["env"].cube_mass_kg == 0.1
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
        arguments.extend([
            "--root",
            str(tmp_path),
            "--open-root",
            str(tmp_path / "open"),
            "--open-aggregation",
            "equal-context",
        ])
    else:
        arguments.extend(["--seed", str(FIT_SEEDS[0])])
    if not enable_bin_pouring:
        arguments.append("--no-bin-pouring")
    monkeypatch.setattr(sys, "argv", arguments)
    tossing3d_dp_priors.main()
    assert captured[0]["enable_bin_pouring"] is enable_bin_pouring
    if command == "fit":
        assert captured[0]["open_aggregation"] == "equal-context"
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
    assert Tossing3DDpPriors.open_protocol()["probes_per_seed_per_context"] == 16


def test_fit_cli_refuses_implicit_aggregation(*, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tossing3d_dp_priors",
            "fit",
            "--root",
            str(tmp_path),
            "--open-root",
            str(tmp_path / "open"),
            "--output",
            str(tmp_path / "prior.json"),
            "--no-bin-pouring",
        ],
    )
    with pytest.raises(SystemExit) as error:
        tossing3d_dp_priors.main()
    assert error.value.code == 2
    assert not (tmp_path / "prior.json").exists()
