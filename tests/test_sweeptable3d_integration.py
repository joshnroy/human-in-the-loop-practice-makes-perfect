"""First-class SweepTable CLI, deployment and replay wiring without simulation."""

from pathlib import Path

import numpy as np
import pytest

from analysis.render_recording import RecordingVideo
from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweeptable3d.cli import SweepTable3DCli
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.recording import SweepTable3DReplayRenderer
from hitl_pmp.environments.sweeptable3d.skill_provider import SweepTable3DSkillProvider
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S
from hitl_pmp.environments.sweeptable3d.step_protocol import SweepTable3DStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from tests.environments.sweeptable3d.test_skill_provider import SymbolicEnvironment


def test_table_is_registered_for_both_protocols():
    assert ENVIRONMENTS["sweeptable3d"] is SweepTable3DCli
    assert STEP_PROTOCOL_ENVIRONMENTS["sweeptable3d"] is SweepTable3DStepProtocol


def test_table_cli_defaults_to_the_practiced_offcenter_tray_distribution():
    args = Cli.parse_args(argv=["--env", "sweeptable3d", "--method", "ees"])
    assert args.fixed_bin_offset == -0.18


def test_sampler_contract_is_angle_then_speed_for_one_five_ball_sweep():
    assert len(O.cubes) == 5
    assert S.SWEEP_TABLE.param_dim == 2
    assert S.SWEEP_BOUNDS == ((-45.0, 45.0), (0.05, 0.30))
    provider = SweepTable3DSkillProvider(env=SymbolicEnvironment())
    sweep = GroundSkill(skill=S.SWEEP_TABLE, objects=(O.robot, O.wiper, O.box, *O.cubes))
    rng = np.random.default_rng(7)
    expected_rng = np.random.default_rng(7)
    params = provider.sample_params(ground_skill=sweep, rng=rng)
    np.testing.assert_array_equal(
        params, [expected_rng.uniform(-45.0, 45.0), expected_rng.uniform(0.05, 0.30)]
    )


def test_evaluation_provider_cannot_solve_with_individual_delivery_or_human_resets():
    args = Cli.parse_args(argv=["--env", "sweeptable3d", "--method", "ees"])
    problem = SweepTable3DStepProtocol.build_evaluation_problem(args=args)
    provider = SweepTable3DStepProtocol.evaluation_skill_provider(problem=problem)
    goal_predicates = {atom.predicate.name for atom in problem.tasks.goal().atoms}
    assert "SweepSucceeded" in goal_predicates
    producers = {
        skill.name
        for skill in provider.skills()
        if any(effect.predicate.name == "SweepSucceeded" for effect in skill.add_effects)
    }
    assert producers == {"SweepTable"}
    assert provider.human_cube_bin_reset_skills() == ()


@pytest.mark.parametrize("method", ["skill-oracle", "random-skills", "ees", "pomdp"])
@pytest.mark.parametrize("human_reset", [True, False])
def test_global_parser_accepts_table_method_and_reset_flags(*, method, human_reset):
    args = Cli.parse_args(
        argv=[
            "--env",
            "sweeptable3d",
            "--method",
            method,
            "--human-reset" if human_reset else "--no-human-reset",
            "--fixed-bin-offset",
            "-0.18",
            "--no-scene-bg",
        ]
    )
    assert args.env == "sweeptable3d"
    assert args.method == method
    assert args.human_reset is human_reset
    assert args.fixed_bin_offset == -0.18
    assert not args.scene_bg


@pytest.mark.parametrize("protocol_flag", ["--evaluate-only", "--practice-step-budget"])
def test_global_main_dispatches_table_step_protocol_without_simulation(
    *, monkeypatch, protocol_flag
):
    from hitl_pmp.launch_gate import LaunchGate
    from hitl_pmp.step_protocol import StepProtocolEntry

    calls = []
    monkeypatch.setattr(LaunchGate, "enforce", lambda **kwargs: None)
    monkeypatch.setattr(StepProtocolEntry, "run", lambda **kwargs: calls.append(kwargs))
    flags = [protocol_flag] if protocol_flag == "--evaluate-only" else [protocol_flag, "10"]
    Cli.main(argv=["--env", "sweeptable3d", "--method", "ees", *flags])
    assert len(calls) == 1
    assert calls[0]["env_cli"] is SweepTable3DCli
    assert calls[0]["environment"] is SweepTable3DStepProtocol


@pytest.mark.parametrize("realistic", [True, False])
def test_recording_dispatches_table_renderer_lazily(*, realistic):
    scene = {"scene_bg": False, "fixed_bin_offset": -0.18}
    renderer = RecordingVideo.default_renderer(
        header=RecordingHeader(environment="sweeptable3d", scene=scene),
        realistic_background=realistic,
    )
    assert isinstance(renderer, SweepTable3DReplayRenderer)
    assert renderer.scene == scene
    assert renderer.scene_bg is (True if realistic else None)
    assert renderer.storyboard


def test_table_scene_is_included_in_package_data():
    project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    package_data = project.split("[tool.setuptools.package-data]", 1)[1]
    assert '"environments/sweeptable3d/*.json"' in package_data
