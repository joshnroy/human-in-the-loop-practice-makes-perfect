"""Bounded coverage contracts, using fake trials rather than native physics."""

import json
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.validate_rack_coverage import Coverage, NativeTrial, build_manifest, main


@pytest.mark.parametrize("shared_module", ["kinder", "kinder_models"])
def test_native_factory_rejects_shared_checkout_dependencies(
    *, tmp_path, monkeypatch, shared_module
):
    from scripts import validate_rack_coverage as coverage_module

    local = tmp_path / "worktree"
    shared = tmp_path / "shared"
    monkeypatch.setattr(coverage_module, "__file__", str(local / "scripts/coverage.py"))
    monkeypatch.setattr(
        coverage_module.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout=str(shared / ".git"))),
    )
    relative = {
        "kinder": "reference/kindergarden/src/kinder/__init__.py",
        "kinder_models": "reference/kinder-baselines/kinder-models/src/kinder_models/__init__.py",
    }
    monkeypatch.setattr(
        coverage_module.importlib.util,
        "find_spec",
        lambda module: SimpleNamespace(
            origin=str((shared if module == shared_module else local) / relative[module])
        ),
    )
    constructor = Mock(side_effect=AssertionError("must reject before native construction"))
    monkeypatch.setattr(
        "hitl_pmp.environments.racktoserving3d.environment.RackServingEnvironment", constructor
    )
    manifest = build_manifest()
    with pytest.raises(RuntimeError, match="must resolve inside this worktree"):
        NativeTrial.make_env(request=manifest)
    constructor.assert_not_called()


def test_manifest_distinguishes_fixed_repeats_from_seeded_human_layouts():
    manifest = build_manifest(seeds=[2, 7])
    fixed, human = manifest["states"][:2]
    assert len(manifest["states"]) == 4
    assert fixed["seed_changes_layout"] is False
    assert human["seed_changes_layout"] is True
    assert human["setup_interventions"] == ["human_batch_return"]
    assert human["expected_human_setup_cost"] == 5
    assert manifest["config"]["mode"] == "training"
    assert manifest["execution_implemented"] is True
    assert not any(manifest["claim"].values())
    assert all(state["measured_locations"] is None for state in manifest["states"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": []},
        {"seeds": [-1]},
        {"layouts": ["unknown"]},
        {"param_sets": [(0.07, 0.0, 0.13)]},
        {"param_sets": [(0, 0.26, 0)]},
        {"param_sets": [(0.07, 0.0)]},
        {"param_sets": [(0.07, 0.046, 0.1)]},
        {"param_sets": [(float("nan"), 0.0, 0.1)]},
        {"max_trials": 0},
        {"max_seconds": float("inf")},
        {"max_seconds": 43201},
        {"trial_seconds": 901},
        {"search_mode": "unknown"},
        {"constant_params": (0.07, 0.0, 0.03)},
        {"scraper_tilt_degrees": float("nan")},
        {"states": ["fixed:-1"]},
        {"step_limit": 0},
    ],
)
def test_rejects_invalid_or_unbounded_manifest(*, kwargs):
    with pytest.raises(ValueError):
        build_manifest(**kwargs)


def test_full_robot_trial_can_have_a_bounded_ten_minute_wall_budget():
    manifest = build_manifest(trial_seconds=600, max_seconds=1800)
    assert manifest["limits"]["wall_seconds_per_trial"] == 600


def test_dry_run_writes_only_an_unexecuted_manifest(*, tmp_path):
    output = tmp_path / "coverage"
    assert (
        main(
            argv=[
                "--dry-run",
                "--output",
                str(output),
                "--seeds",
                "3",
                "--params",
                "0.05",
                "0.01",
                "0.1",
            ]
        )
        == 0
    )
    assert [path.name for path in output.iterdir()] == ["manifest.json"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["param_sets"] == [{"s_m": 0.05, "delta_m": 0.01, "d_m": 0.1}]
    assert manifest["native_workers_started"] == 0
    assert all(state["status"] == "untested" for state in manifest["states"])


def test_manifest_refuses_existing_output(*, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("preserve me")
    with pytest.raises(SystemExit):
        main(argv=["--output", str(output)])
    assert sentinel.read_text() == "preserve me"


def test_native_execution_requires_explicit_witness_confirmation(*, tmp_path):
    with pytest.raises(SystemExit):
        main(argv=["--run", "--output", str(tmp_path / "native")])
    assert not (tmp_path / "native").exists()


def test_configurable_tilt_and_explicit_three_state_pilot(*, tmp_path):
    output = tmp_path / "pilot"
    assert (
        main(
            argv=[
                "--dry-run",
                "--output",
                str(output),
                "--scraper-tilt-degrees",
                "0",
                "--state",
                "fixed:0",
                "--state",
                "human_batch_return:0",
                "--state",
                "human_batch_return:1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["scraper_tilt_degrees"] == 0
    assert [state["id"] for state in manifest["states"]] == [
        "fixed:seed=0",
        "human_batch_return:seed=0",
        "human_batch_return:seed=1",
    ]


class TrialStub:
    def __init__(self, *, mismatch=False, timeout=False, invalid=False):
        self.calls = []
        self.mismatch = mismatch
        self.timeout = timeout
        self.invalid = invalid

    def run(self, *, request, output):
        self.calls.append(request)
        state, params = request["state"], request["params"]
        offset = len(self.calls) if self.mismatch else state["seed"]
        snapshot = {
            "qpos": [float(offset)],
            "qvel": [0.0],
            "poses": {"cube_0": [float(offset), 0, 1, 1, 0, 0, 0]},
            "locations": {"cube_0": "OnRack"},
            "supports": {"cube_0": ["rack_0"]},
        }
        success = (state["seed"] == 0 and params["d_m"] == 0.12) or (
            state["seed"] == 1 and params["d_m"] == 0.10
        )
        result = {
            "status": "timeout" if self.timeout else "complete",
            "initial": snapshot,
            "final": snapshot,
            "success": success,
            "task_succeeded": success,
            "scraper_home": success,
            "settled": True,
            "return_completed": True,
            "errors": [],
            "invalidating_issue": "bad setup" if self.invalid else None,
        }
        Coverage.write(path=output / "trial.json", payload=result)
        return result


def test_existential_coverage_retains_failures_and_stops_each_state(*, tmp_path):
    manifest = build_manifest(
        seeds=[0, 1],
        layouts=["human_batch_return"],
        param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)],
    )
    stub = TrialStub()
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=stub.run)
    assert len(stub.calls) == 3
    assert summary["claim"]["all_sampled_states_covered"]
    assert not summary["claim"]["common_parameter_required"]
    assert not summary["claim"]["full_space_proof"]
    assert [state["witness"]["d_m"] for state in summary["states"]] == [0.12, 0.10]
    assert [trial["success"] for trial in summary["attempts"]] == [False, True, True]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert len(list((tmp_path / "trials").glob("*/trial.json"))) == 3


@pytest.mark.parametrize("kind", ["mismatch", "timeout", "invalid"])
def test_partial_evidence_and_invalidating_issues_stop_run(*, tmp_path, kind):
    manifest = build_manifest(seeds=[0, 1], param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)])
    stub = TrialStub(**{kind: True})
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=stub.run)
    assert len(stub.calls) == (2 if kind == "mismatch" else 1)
    assert not summary["claim"]["all_sampled_states_covered"]
    assert summary["stop_reason"] == ("timeout" if kind == "timeout" else "invalidating_issue")
    assert (tmp_path / "summary.json").exists()


def test_trial_budget_and_duplicate_layout_reporting(*, tmp_path):
    manifest = build_manifest(
        seeds=[0, 1], layouts=["fixed"], param_sets=[(0.07, 0.0, 0.12)], max_trials=1
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=TrialStub().run)
    assert len(summary["attempts"]) == 1
    assert summary["stop_reason"] == "max_trials"
    assert summary["states"][1]["status"] == "untested"


@pytest.mark.parametrize("field", ["task_succeeded", "scraper_home", "settled", "return_completed"])
def test_aggregation_does_not_trust_success_flag_alone(*, tmp_path, field):
    stub = TrialStub()

    def run(*, request, output):
        result = stub.run(request=request, output=output)
        result[field] = False
        return result

    manifest = build_manifest(seeds=[1], layouts=["fixed"], param_sets=[(0.07, 0.0, 0.10)])
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=run)
    assert not summary["claim"]["all_sampled_states_covered"]


def test_grid_reports_constant_baseline_and_per_state_oracle(*, tmp_path):
    manifest = build_manifest(
        seeds=[0, 1],
        layouts=["human_batch_return"],
        param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)],
        search_mode="grid",
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=TrialStub().run)
    assert len(summary["attempts"]) == 4
    assert summary["comparison"]["per_state_oracle"]["covered"] == 2
    assert summary["comparison"]["constant_policy"]["covered"] == 1
    assert summary["comparison"]["constant_policy"]["tested"] == 2
    assert summary["comparison"]["common_params_status"] == "none_in_tested_grid"
    assert not summary["claim"]["parameter_learning_required"]


def test_early_success_does_not_disprove_untested_common_params(*, tmp_path):
    manifest = build_manifest(
        seeds=[0, 1],
        layouts=["human_batch_return"],
        param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)],
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=TrialStub().run)
    assert summary["comparison"]["common_params_status"] == "inconclusive"
    assert summary["comparison"]["per_state_oracle"]["covered"] == 2


def test_common_setting_and_duplicate_layouts_are_reported(*, tmp_path):
    stub = TrialStub()

    def run(*, request, output):
        request["state"]["seed"] = 0
        return stub.run(request=request, output=output)

    manifest = build_manifest(seeds=[0, 1], layouts=["fixed"], param_sets=[(0.07, 0.0, 0.12)])
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=run)
    assert summary["comparison"]["common_params_status"] == "witness_found"
    assert summary["comparison"]["measured_distinct_layouts"] == 1
    assert summary["states"][1]["duplicate_of"] == ["fixed:seed=0"]


def test_explicit_baseline_is_first_and_budgeted():
    manifest = build_manifest(param_sets=[(0.05, 0.01, 0.12)], constant_params=(0.07, 0.015, 0.1))
    assert manifest["param_sets"][0] == {"s_m": 0.07, "delta_m": 0.015, "d_m": 0.1}
    assert len(manifest["param_sets"]) == 2


@pytest.mark.parametrize("exit_kind", ["timeout", "crash"])
def test_supervisor_keeps_partial_evidence_and_native_log(*, tmp_path, exit_kind):
    payload = {"status": "running", "initial": {"measured": True}, "success": False}
    Coverage.write(path=tmp_path / "trial.json", payload=payload)
    code = (
        "import time; print('worker output', flush=True); time.sleep(10)"
        if exit_kind == "timeout"
        else "raise SystemExit(9)"
    )
    result = Coverage.supervise(
        command=[sys.executable, "-c", code], deadline=time.monotonic() + 0.5, output=tmp_path
    )
    assert result["status"] == ("timeout" if exit_kind == "timeout" else "worker_crashed")
    assert result["initial"] == {"measured": True}
    assert not result["success"]
    assert json.loads((tmp_path / "trial.json").read_text()) == result
    assert (tmp_path / "native.log").exists()


class FakeEnvironment:
    def __init__(self, *, request, verdict=True, step_burst=1, wrong_cost=False):
        self.request = request
        self.verdict = verdict
        self.step_burst = step_burst
        self.wrong_cost = wrong_cost
        self.events = []
        self.steps = 0
        self.cost = 0
        self.interventions = 0
        self.closed = False
        self.shelf = self.tool = self

    def backend(self):
        return self

    def set_control_step_observers(self, *, before=None, after=None):
        self.before, self.after = before, after

    def reset_to_seed(self, *, seed):
        self.events.append(("reset", seed))

    def reset_movables(self, *, destination):
        self.events.append(destination)
        self.interventions += 1
        self.cost += 0 if self.wrong_cost else 5

    def control_steps(self):
        return self.steps

    def take_action(self, *, action):
        self.events.append(int(action[0]))
        for _ in range(self.step_burst):
            self.hold(count=1, label="action")

    def hold(self, *, count, label):
        self.before()
        self.steps += count
        self.events.append(label)
        self.after()

    def last_skill_error(self):
        return None

    def rack_controllers(self):
        return self

    def feedback(self):
        return nullcontext()

    def task_succeeded(self):
        self.events.append("verdict")
        assert self.events[-2] == "coverage_settling"
        assert 2 in self.events
        return self.verdict

    def at_home(self):
        return True

    def close(self):
        self.closed = True

    @staticmethod
    def snapshot(*, env):
        offset = env.request["state"]["seed"] if env.interventions else 0
        return {
            "qpos": [offset],
            "qvel": [0],
            "poses": {"cube_0": [offset]},
            "locations": {"cube_0": "OnRack"},
            "supports": {"cube_0": ["rack_0"]},
            "ledger": {"human_cost": env.cost, "human_interventions": env.interventions},
        }


@pytest.mark.parametrize("verdict", [True, False])
def test_native_adapter_uses_costed_layout_and_post_return_settled_verdict(
    *, tmp_path, monkeypatch, verdict
):
    manifest = build_manifest(states=["human_batch_return:7"], step_limit=30)
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + 30,
    }
    env = FakeEnvironment(request=request, verdict=verdict)
    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert result["success"] == verdict
    assert result["actual_setup_cost"] == 5
    assert result["initial"]["qpos"] == [7]
    assert env.events[:2] == [("reset", 7), "human_batch_return"]
    assert [event for event in env.events if isinstance(event, int)] == [6, 0, 1, 2]
    assert env.events.count("coverage_settling") == 20
    assert env.closed


@pytest.mark.parametrize("failure", ["steps", "timeout", "cost", "mismatch"])
def test_native_adapter_limits_and_setup_failures_preserve_partial(
    *, tmp_path, monkeypatch, failure
):
    manifest = build_manifest(states=["human_batch_return:7"], step_limit=2, settle_steps=1)
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + (-1 if failure == "timeout" else 30),
    }
    env = FakeEnvironment(request=request, step_burst=3, wrong_cost=failure == "cost")
    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    if failure == "mismatch":
        request["reference_initial"] = FakeEnvironment.snapshot(env=env)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert not result["success"]
    assert result["traceback"]
    assert env.closed
    assert env.steps == (2 if failure == "steps" else 0)
    assert result["status"] == {"steps": "control_step_limit", "timeout": "timeout"}.get(
        failure, "error"
    )
    assert json.loads((tmp_path / "trial.json").read_text())["final"] is not None


def test_source_drift_invalidates_previous_witnesses(*, tmp_path, monkeypatch):
    digest = ["before"]
    monkeypatch.setattr(Coverage, "source_digest", lambda: digest[0])
    stub = TrialStub()

    def run(*, request, output):
        result = stub.run(request=request, output=output)
        digest[0] = "after"
        return result

    manifest = build_manifest(seeds=[1], layouts=["fixed"], param_sets=[(0.07, 0.0, 0.10)])
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=run)
    assert summary["stop_reason"] == "invalidating_issue"
    assert summary["comparison"]["common_params_status"] == "invalidated"
    assert not summary["claim"]["all_sampled_states_covered"]


def test_unstarted_states_and_control_cutoffs_remain_unresolved(*, tmp_path):
    stub = TrialStub()

    def run(*, request, output):
        result = stub.run(request=request, output=output)
        result.update(status="control_step_limit", success=False)
        return result

    manifest = build_manifest(
        seeds=[0, 1], layouts=["human_batch_return"], param_sets=[(0.07, 0.0, 0.10)], max_trials=1
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=run)
    assert summary["comparison"]["constant_policy"]["tested"] == 0
    assert summary["comparison"]["common_params_status"] == "inconclusive"
    assert len(summary["comparison"]["constant_policy"]["unresolved_state_ids"]) == 2


def test_exhausted_wall_budget_starts_no_trial(*, tmp_path, monkeypatch):
    moments = iter([0, 200])
    monkeypatch.setattr(time, "monotonic", lambda: next(moments))
    stub = TrialStub()
    summary = Coverage.run(manifest=build_manifest(), output=tmp_path, trial_runner=stub.run)
    assert not stub.calls
    assert summary["stop_reason"] == "timeout"


def test_native_close_still_runs_when_final_measurement_fails(*, tmp_path, monkeypatch):
    manifest = build_manifest(seeds=[0], layouts=["fixed"])
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + 30,
    }
    env = FakeEnvironment(request=request)

    def snapshot(*, env):
        if "verdict" in env.events:
            raise RuntimeError("measurement failed")
        return FakeEnvironment.snapshot(env=env)

    monkeypatch.setattr(NativeTrial, "snapshot", snapshot)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert env.closed
    assert not result["success"]
    assert result["cleanup_traceback"]


def test_worker_entrypoint_cannot_bypass_witness_gate(*, tmp_path):
    with pytest.raises(SystemExit):
        main(argv=["--worker", str(tmp_path / "request.json")])


def test_native_adapters_recreate_each_params_and_seeded_layout(*, tmp_path, monkeypatch):
    environments = []

    def factory(*, request):
        env = FakeEnvironment(request=request, verdict=request["params"]["d_m"] == 0.12)
        environments.append(env)
        return env

    def runner(*, request, output):
        return NativeTrial.execute(request=request, output=output, env_factory=factory)

    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    manifest = build_manifest(
        seeds=[3, 7],
        layouts=["human_batch_return"],
        param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)],
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=runner)
    assert len(environments) == 4
    assert all(env.closed for env in environments)
    assert all(env.events.count("human_batch_return") == 1 for env in environments)
    assert [row["initial"]["qpos"] for row in summary["attempts"]] == [[3], [3], [7], [7]]
    assert summary["comparison"]["measured_distinct_layouts"] == 2
    assert summary["comparison"]["common_params_status"] == "witness_found"


def test_manifest_records_current_contact_config():
    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig

    config = RackServingConfig().model_dump(mode="json")
    manifest = build_manifest(scraper_tilt_degrees=0)
    for key in (
        "support_contact_dimension",
        "dumpling_contact_dimension",
        "dumpling_friction",
        "rack_friction",
        "tray_friction",
    ):
        assert manifest["config"][key] == config[key]


def test_native_controller_caught_step_cutoff_is_not_a_physical_failure(*, tmp_path, monkeypatch):
    manifest = build_manifest(seeds=[0], layouts=["fixed"], step_limit=2, settle_steps=1)
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + 30,
    }
    env = FakeEnvironment(request=request, step_burst=2)
    monkeypatch.setattr(env, "last_skill_error", lambda: "2 control steps reached")
    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert result["status"] == "control_step_limit"
    assert not result["success"]
    assert env.closed


def test_physical_delivery_without_tool_home_is_not_coverage(*, tmp_path, monkeypatch):
    manifest = build_manifest(seeds=[0], layouts=["fixed"])
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + 30,
    }
    env = FakeEnvironment(request=request)
    monkeypatch.setattr(env, "at_home", lambda: False)
    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert result["task_succeeded"]
    assert not result["success"]


def test_same_initial_detects_velocity_and_orientation_changes():
    initial = {
        "qpos": [1],
        "qvel": [0],
        "poses": {"cube_0": [0, 0, 0, 1, 0, 0, 0]},
        "locations": {"cube_0": "OnRack"},
        "supports": {"cube_0": ["rack_0"]},
    }
    assert Coverage.same_initial(left=initial, right=initial)
    changed = json.loads(json.dumps(initial))
    changed["qvel"] = [0.01]
    assert not Coverage.same_initial(left=initial, right=changed)
    changed = json.loads(json.dumps(initial))
    changed["poses"]["cube_0"][3:] = [0, 1, 0, 0]
    assert not Coverage.same_initial(left=initial, right=changed)


def test_default_param_sets_are_squeeze_gather_and_sweep_depth():
    manifest = build_manifest()
    assert {"s_m": 0.07, "delta_m": 0.015, "d_m": 0.1} in manifest["param_sets"]
    assert all(set(params) == {"s_m", "delta_m", "d_m"} for params in manifest["param_sets"])
    assert manifest["constant_params"] == manifest["param_sets"][0]
    with pytest.raises(ValueError):
        build_manifest(param_sets=[])


def test_native_adapter_sends_resolved_squeeze_gather_and_depth(*, tmp_path, monkeypatch):
    manifest = build_manifest(seeds=[0], layouts=["fixed"], param_sets=[(0.05, 0.01, 0.1)])
    request = {
        "state": manifest["states"][0],
        "params": manifest["param_sets"][0],
        "config": manifest["config"],
        "limits": manifest["limits"],
        "deadline": time.monotonic() + 30,
    }
    env = FakeEnvironment(request=request)
    actions = []
    original = env.take_action

    def record(*, action):
        actions.append(list(action))
        original(action=action)

    env.take_action = record
    monkeypatch.setattr(NativeTrial, "snapshot", FakeEnvironment.snapshot)
    result = NativeTrial.execute(request=request, output=tmp_path, env_factory=lambda **_: env)
    assert [action[2:] for action in actions if action[0] == 1] == [[0.05, 0.01, 0.1]]
    assert result["resolved_params"] == {"s_m": 0.05, "delta_m": 0.01, "d_m": 0.1}


@pytest.mark.parametrize(
    ("location", "supports", "settled", "expected"),
    [
        ("OnTray", {"box_0"}, True, "on_plate"),
        ("OnTray", {"box_0"}, False, "on_plate_unsettled"),
        (None, {"box_0", "rack_0"}, False, "bridging"),
        (None, {"box_0"}, False, "overhang"),
        ("SpilledCounter", {"box_0", "cupboard_1"}, False, "overhang"),
        ("OnRack", {"rack_0"}, False, "on_rack"),
        (None, {"rack_0"}, False, "on_rack"),
        ("SpilledCounter", {"cupboard_1"}, False, "on_counter"),
        ("SpilledFloor", {"floor"}, False, "floor"),
        ("HoldingFood", set(), False, "held"),
        (None, set(), False, "unsupported"),
    ],
)
def test_physical_delivery_category_is_separate_from_controller_errors(
    *, location, supports, settled, expected
):
    category = NativeTrial.physical_category(
        location=location, supports=frozenset(supports), settled=settled
    )
    assert category == expected


def test_grid_mode_keeps_running_after_an_unresolved_trial(*, tmp_path):
    stub = TrialStub()

    def run(*, request, output):
        result = stub.run(request=request, output=output)
        if len(stub.calls) == 1:
            result.update(status="timeout", success=False)
        return result

    manifest = build_manifest(
        seeds=[0, 1],
        layouts=["human_batch_return"],
        param_sets=[(0.07, 0.0, 0.10), (0.07, 0.0, 0.12)],
        search_mode="grid",
    )
    summary = Coverage.run(manifest=manifest, output=tmp_path, trial_runner=run)
    assert len(stub.calls) == 4
    assert summary["stop_reason"] == "search_complete"
    assert summary["attempts"][0]["status"] == "timeout"
    assert not summary["attempts"][0]["success"]
    constant = summary["comparison"]["constant_policy"]
    assert "human_batch_return:seed=0" in constant["unresolved_state_ids"]
