"""Preflight contracts without constructing KINDER or running physics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.sweeptable3d_preflight import (
    BoundedStop,
    MotionAudit,
    Preflight,
    PreflightCase,
    PreflightOptions,
)


class FakeEnvironment:
    def __init__(self, *, partial: bool = False, teleport: bool = False) -> None:
        self.cube_names = tuple(f"cube_{index}" for index in range(5))
        self.box_name = "box_0"
        self.wiper_name = "wiper_0"
        self.positions = {name: "OnTable" for name in self.cube_names}
        self.failed = False
        self.held = False
        self.partial = partial
        self.teleport = teleport
        self.resets = 0
        self.steps = 0
        self.error = None
        self.closed = False
        self.state = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3), time=0.0)
        self.simulator = SimpleNamespace(step=self.physics_step)
        self.gym = SimpleNamespace(step=self.control_step)

    def backend(self):
        return self

    def env(self):
        return self.gym

    def sim(self):
        return self.simulator

    def data(self):
        return self.state

    def core(self):
        return self

    def model(self):
        return SimpleNamespace(
            body_jntadr=np.zeros(5, dtype=int),
            jnt_dofadr=[0],
            body=lambda name: SimpleNamespace(id=self.cube_names.index(name)),
        )

    def get_object(self, *args):
        return SimpleNamespace(get_pose=lambda: (self.state.qpos, [1, 0, 0, 0]))

    def cube_in_box(self, *args):
        return self.positions[args[0]] == "InBox"

    def controllers(self):
        return SimpleNamespace(contact_metrics={"controller_revision": 3})

    def physics_step(self, *args, **kwargs):
        self.state.qpos += 1
        self.state.time += 0.01

    def control_step(self, *args, **kwargs):
        self.simulator.step()

    def reset_to_seed(self, *, seed):
        self.resets += 1

    def task_failed(self):
        return self.failed

    def ground_cubes(self):
        return tuple(name for name, place in self.positions.items() if place == "OnFloor")

    def task_succeeded(self):
        return not self.failed and all(place == "InBox" for place in self.positions.values())

    def abstract_atoms(self):
        atoms = {
            (place, (name, self.box_name) if place == "InBox" else (name,))
            for name, place in self.positions.items()
        }
        atoms.add(
            ("HoldingWiper", ("robot", self.wiper_name))
            if self.held
            else ("WiperHome", (self.wiper_name,))
        )
        if not self.held:
            atoms.add(("HandEmpty", ("robot",)))
        if not self.failed:
            atoms.add(("TaskActive", ()))
        return frozenset(atoms)

    def take_action(self, *, action):
        skill, index, angle, _ = action
        self.error = None
        self.steps = 0
        if self.failed and int(skill) == 1:
            self.error = "task failed: reset required"
            return
        for _ in range(2):
            self.gym.step(action)
            self.steps += 1
        if self.teleport:
            self.state.qpos += 100
        if int(skill) == 0:
            self.held = True
        elif int(skill) == 2:
            self.held = False
        elif int(skill) == 1:
            self.failed = angle == 45
            for name in self.cube_names:
                self.positions[name] = "OnFloor" if self.failed else "InBox"
            if self.partial:
                self.positions[self.cube_names[-1]] = "OnTable"
        elif int(skill) in {6, 7}:
            self.positions[self.cube_names[int(index)]] = "OnTable"
            if all(place == "OnTable" for place in self.positions.values()):
                self.failed = False

    def reset_movables(self, *, destination):
        assert destination == "reset_cubes"
        self.positions = dict.fromkeys(self.cube_names, "OnTable")
        self.failed = False
        self.state.qpos += 100

    def last_skill_error(self):
        return self.error

    def last_controller_steps(self):
        return self.steps

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def mock_live_geometry(*, monkeypatch):
    monkeypatch.setattr(Preflight, "verify_live", lambda **kwargs: {"unit_test_fake": True})


def test_default_plan_and_reference_cases() -> None:
    options = PreflightOptions()
    assert options.scene_bg is True
    cases = Preflight.cases(options=options)
    assert options.fixed_bin_offset == -0.18
    assert options.mode == "plan"
    assert next(case for case in cases if case.name == "centered").bin_offset == 0
    left = next(case for case in cases if case.name == "left")
    assert (left.angle_deg, left.speed_m_s, left.bin_offset) == (-27, 0.15, -0.18)
    centered = next(case for case in cases if case.name == "centered")
    assert (centered.angle_deg, centered.speed_m_s) == (0, 0.05)


@pytest.mark.parametrize(
    "field",
    [
        "max_actions",
        "max_control_steps",
        "skill_step_limit",
        "max_cases",
        "case_seconds",
        "record_stride",
    ],
)
def test_caps_must_be_positive(*, field) -> None:
    with pytest.raises(ValueError):
        PreflightOptions(**{field: 0})


def test_subset_and_seed_expansion_is_bounded() -> None:
    options = PreflightOptions(subset=["full5"], seeds=[3, 8])
    assert [(case.name, case.seed) for case in Preflight.cases(options=options)] == [
        ("full5", 3),
        ("full5", 8),
    ]
    with pytest.raises(ValueError, match="max_cases"):
        Preflight.cases(options=options.model_copy(update={"max_cases": 1}))


def test_full5_is_physical_and_has_a_next_sweep(*, tmp_path: Path) -> None:
    env = FakeEnvironment()
    result = Preflight.run_case(
        env=env, case=PreflightCase(name="full5"), options=PreflightOptions(), output=tmp_path
    )
    assert result["passed"], result
    assert env.resets == 1
    actions = result["actions"]
    assert [row["skill"] for row in actions].count("ReturnCubeToTableFromBox") == 5
    assert [row["skill"] for row in actions].count("SweepTable") == 2
    assert all(row["actor"] == "robot" for row in actions)
    assert all(row["cost_control_steps"] == row["actual_control_steps"] == 2 for row in actions)
    assert result["audit"]["outside_step_qpos_changes"] == 0


def test_partial_delivery_is_binary_failure_not_full5_setup(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(partial=True),
        case=PreflightCase(name="full5"),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert not result["passed"]
    assert not any(row["skill"].startswith("ReturnCube") for row in result["actions"])
    assert result["actions"][1]["binary_delivery_success"] is False
    assert "all five" in result["finding"]


def test_failure_latch_and_held_tool_return(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(),
        case=PreflightCase(name="failure", angle_deg=45),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert result["passed"], result
    blocked = result["actions"][2]
    assert blocked["skill"] == "SweepTable"
    assert blocked["actual_control_steps"] == 0
    assert result["checks"]["held_tool_return_after_failure"]
    assert result["checks"]["failure_latch_cleared_after_full_reset"]


def test_human_reset_is_labeled_and_never_a_robot_reset(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(),
        case=PreflightCase(name="human", angle_deg=45),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert result["passed"], result
    human = next(row for row in result["actions"] if row["actor"] == "human")
    assert human["skill"] == "reset_cubes"
    assert human["cost_control_steps"] is None
    assert human["privileged_pose_reset"] is True
    assert result["audit"]["outside_step_qpos_changes"] == 0


def test_qpos_change_after_last_step_is_a_failure(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(teleport=True),
        case=PreflightCase(name="centered"),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert not result["passed"]
    assert result["audit"]["outside_step_qpos_changes"] == 1
    assert "qpos" in result["finding"]


def test_actual_budget_stops_inside_an_action(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(),
        case=PreflightCase(name="full5"),
        options=PreflightOptions(max_control_steps=3),
        output=tmp_path,
    )
    assert not result["passed"]
    assert sum(row["actual_control_steps"] for row in result["actions"]) == 3
    assert result["actions"][-1]["interrupted"]


def test_audit_restores_wrapped_methods_on_error() -> None:
    env = FakeEnvironment()
    original = env.gym.step
    with (
        pytest.raises(BoundedStop),
        MotionAudit(env=env, options=PreflightOptions(max_control_steps=1)),
    ):
        env.gym.step()
        env.gym.step()
    assert env.gym.step == original


def test_plan_does_not_construct_environment(*, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Preflight, "build_environment", lambda **kwargs: pytest.fail("simulation"))
    assert Preflight.main(argv=["--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "plan.json").read_text())
    assert report["physics_validated"] is False
    assert report["cases"]
    assert report["source"]["sha256"]


def test_learner_plan_uses_production_cli_not_investigation() -> None:
    commands = Preflight.learner_commands(options=PreflightOptions(), output=Path("/tmp/run"))
    assert len(commands) == 2
    for command in commands:
        assert "hitl_pmp.cli" in command
        assert "--fixed-bin-offset" in command and "-0.18" in command
        assert "--evaluation-control-steps" in command
        assert "--practice-step-budget" in command
        assert "run-heavy" in command and "MemoryMax=8G" in command
        assert not any("investigation" in arg or "ball_review" in arg for arg in command)


def test_learner_plan_has_valid_bounded_search_and_cost_evidence() -> None:
    from hitl_pmp.cli import Cli

    for command in Preflight.learner_commands(options=PreflightOptions(), output=Path("/tmp/run")):
        args = Cli.parse_args(argv=command[command.index("hitl_pmp.cli") + 1 :])
        assert args.practice_cost_config == Path("configs/sweepoffshelf3d_practice_costs.json")
        assert args.record_sampler_draws
        assert args.sampler_max_train_iters == 5
        assert args.measurement_interval_steps == args.practice_step_budget
        assert not args.initial_measurement
        if args.method == "pomdp":
            assert args.pomdp_observation_probability_weight == 0
            assert args.pomdp_max_search_iterations == 20


def test_failed_run_returns_nonzero_and_preserves_report(*, tmp_path: Path, monkeypatch) -> None:
    env = FakeEnvironment(partial=True)
    monkeypatch.setattr(Preflight, "build_environment", lambda **kwargs: env)
    monkeypatch.setattr(Preflight, "verify_contract", lambda **kwargs: {})
    assert (
        Preflight.main(
            argv=[
                "--mode",
                "physics",
                "--subset",
                "full5",
                "--seeds",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert env.closed
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] is False
    assert report["launch_ready"] is False


def test_contract_refuses_old_offset_yaw_schema() -> None:
    with pytest.raises(ValueError, match="angle.*speed"):
        Preflight.verify_schema(bounds=((-0.4, 0.4), (-75, 75)))


def test_source_digest_changes_with_uncommitted_content(*, tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("old")
    before = Preflight.hash_files(paths=[source])
    source.write_text("new")
    assert before["sha256"] != Preflight.hash_files(paths=[source])["sha256"]


def test_exact_frozen_bounds_and_invalid_cases() -> None:
    Preflight.verify_schema(bounds=((-45, 45), (0.05, 0.30)))
    for values in (
        {"angle_deg": 46},
        {"angle_deg": -46},
        {"speed_m_s": 0.04},
        {"speed_m_s": 0.31},
        {"speed_m_s": float("nan")},
    ):
        with pytest.raises(ValueError):
            PreflightCase(name="full5", **values)


def test_one_successful_return_is_not_full5(*, tmp_path: Path) -> None:
    env = FakeEnvironment()
    original = env.take_action

    def refuse_second_return(*, action):
        if action[0] == 6 and action[1] == 1:
            env.error = "inaccessible grasp"
            env.steps = 0
        else:
            original(action=action)

    env.take_action = refuse_second_return
    result = Preflight.run_case(
        env=env, case=PreflightCase(name="full5"), options=PreflightOptions(), output=tmp_path
    )
    assert not result["passed"]
    assert result["checks"]["returned_cube_0"]
    assert "full5_table_restored" not in result["checks"]
    assert sum(row["skill"] == "SweepTable" for row in result["actions"]) == 1


def test_per_case_config_and_state_hashes(*, tmp_path: Path) -> None:
    result = Preflight.run_case(
        env=FakeEnvironment(),
        case=PreflightCase(name="full5"),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert result["config_sha256"]
    assert result["initial_qpos_sha256"]
    assert result["final_qpos_sha256"]
    assert (tmp_path / "initial_state.npz").is_file()
    assert (tmp_path / "final_state.npz").is_file()


def test_video_replays_saved_recording_in_separate_process(*, tmp_path: Path, monkeypatch) -> None:
    import scripts.sweeptable3d_preflight as module

    seen = []

    def replay(*args, **kwargs):
        seen.append((args, kwargs))
        (tmp_path / "episode.mp4").write_bytes(b"fake video")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", replay)
    assert Preflight.render_video(output=tmp_path, options=PreflightOptions())["saved"]
    command = seen[0][0][0]
    assert "analysis.render_recording" in command
    assert seen[0][1]["timeout"] == 600


def test_passed_subset_never_means_launch_ready(*, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Preflight, "build_environment", lambda **kwargs: FakeEnvironment())
    monkeypatch.setattr(Preflight, "verify_contract", lambda **kwargs: {"scene_sha256": "fake"})
    monkeypatch.setattr(
        Preflight,
        "source_manifest",
        lambda: {"sha256": "unchanged", "head": "fake", "submodules": "fake"},
    )
    assert (
        Preflight.main(
            argv=[
                "--mode",
                "physics",
                "--subset",
                "full5",
                "--seeds",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] and not report["launch_ready"]
    assert not report["coverage_complete"]
    assert report["learner_gate"] == "not_run"
    case = report["results"][0]
    assert case["source_sha256"] == "unchanged"
    assert case["scene_sha256"] == "fake"


def test_aborted_attempt_is_not_ground_contact_evidence(*, tmp_path: Path) -> None:
    env = FakeEnvironment()
    env.ground_cubes = lambda: ()
    result = Preflight.run_case(
        env=env,
        case=PreflightCase(name="failure", angle_deg=45),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert not result["passed"]
    assert result["checks"]["failed_attempt_observed"]
    assert not result["checks"]["ground_failure_observed"]


def test_human_reset_can_cover_aborted_attempt_without_claiming_ground(*, tmp_path: Path) -> None:
    env = FakeEnvironment()
    env.ground_cubes = lambda: ()
    result = Preflight.run_case(
        env=env,
        case=PreflightCase(name="human", angle_deg=45),
        options=PreflightOptions(),
        output=tmp_path,
    )
    assert result["passed"], result
    assert result["ground_contact_observed"] is False
    assert "ground_failure_observed" not in result["checks"]


def test_diagnostics_separate_capture_ground_and_production_verdict(*, tmp_path: Path) -> None:
    env = FakeEnvironment(partial=True)
    result = Preflight.run_case(
        env=env, case=PreflightCase(name="full5"), options=PreflightOptions(), output=tmp_path
    )
    observation = result["final_observation"]
    assert observation["ground_cubes"] == []
    assert observation["task_succeeded"] is False
    assert len(observation["balls"]) == 5
    assert sum(ball["in_tray"] for ball in observation["balls"].values()) == 4
    assert "xyz" in observation["balls"]["cube_0"]
    assert observation["contact_metrics"]["controller_revision"] == 3
    assert observation["qvel"] == [0, 0, 0]


@pytest.mark.parametrize("scene_bg", [True, False])
def test_background_reaches_environment_recording_and_learner_plan(
    *, scene_bg, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "hitl_pmp.environments.sweeptable3d.environment",
        SimpleNamespace(SweepTable3DEnvironment=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    options = PreflightOptions(scene_bg=scene_bg)
    case = PreflightCase(name="full5")
    env = Preflight.build_environment(case=case, options=options)
    assert env.scene_bg is scene_bg
    backend = SimpleNamespace(
        CONTROL_FREQUENCY=100, ROBOT_JOINTS=(), motion_state=lambda: {"robot": [0.0]}
    )
    env.backend = lambda: backend
    env.get_current_state = lambda: SimpleNamespace(data={})
    env.attach_motion_sink = lambda **kwargs: None
    writer = Preflight.recording(env=env, case=case, options=options, output=tmp_path)
    writer.close()
    header = json.loads((tmp_path / "recording.jsonl").read_text().splitlines()[0])
    assert header["scene"]["scene_bg"] is scene_bg
    for command in Preflight.learner_commands(options=options, output=tmp_path):
        assert ("--no-scene-bg" in command) is (not scene_bg)


@pytest.mark.parametrize(
    "flags, expected", [([], True), (["--scene-bg"], True), (["--no-scene-bg"], False)]
)
def test_background_cli_defaults_and_overrides(
    *, flags, expected, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Preflight, "source_manifest", lambda: {"sha256": "fake"})
    monkeypatch.setattr(Preflight, "build_environment", lambda **kwargs: pytest.fail("simulation"))
    assert Preflight.main(argv=["--output-dir", str(tmp_path), *flags]) == 0
    report = json.loads((tmp_path / "plan.json").read_text())
    assert report["options"]["scene_bg"] is expected
