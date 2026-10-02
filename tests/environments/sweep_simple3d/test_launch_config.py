"""Launch guards and method parity without constructing physical simulators."""

import argparse
import json

import numpy as np
import pytest

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.sweep_simple3d.cli import SweepSimpleCli
from hitl_pmp.methods.belief_space.sweep_cli import SweepPomdpCli


@pytest.mark.parametrize("status", [None, "DRAFT", "READY", "IMPLEMENTING_NOT_READY", "unknown"])
def test_simple_manifest_requires_explicit_frozen_status(*, status):
    with pytest.raises(ValueError, match="FROZEN"):
        SweepSimpleCli.validate_manifest(manifest={"status": status})


@pytest.fixture
def approved_manifest():
    return dict(
        status="FROZEN",
        environment="sweep_simple3d",
        num_cycles=50,
        max_steps_per_interaction=20,
        num_test_tasks=10,
        deployment_horizon=10,
        practice_reset_policy="never",
        valid_practice_seeds=[0, 1, 2],
        valid_evaluation_seeds=list(range(10000, 10010)),
        pomdp=dict(
            goal_pursuit_horizon=0,
            pomdp_solver="determinized_astar",
            pomdp_max_search_iterations=1000,
            pomdp_search_depth=20,
            pomdp_inference_engine="grid",
            pomdp_grid_competence_bins="25",
            pomdp_grid_learning_rate_bins="16",
            pomdp_num_particles="1024",
            pomdp_linear_cost_lambda="3e-6",
        ),
    )


def test_simple_manifest_accepts_frozen_protocol(*, approved_manifest):
    SweepSimpleCli.validate_manifest(manifest=approved_manifest)


@pytest.mark.parametrize(
    "key,value",
    [
        ("goal_pursuit_horizon", None),
        ("goal_pursuit_horizon", 1),
        ("pomdp_solver", "expectimax"),
        ("pomdp_max_search_iterations", 999),
        ("pomdp_num_particles", 512),
        ("pomdp_grid_competence_bins", 24),
    ],
)
def test_simple_rejects_conflicting_model_settings(*, approved_manifest, key, value):
    approved_manifest["pomdp"][key] = value
    with pytest.raises(ValueError, match=key):
        SweepSimpleCli.validate_manifest(manifest=approved_manifest)


@pytest.mark.parametrize("method", ["ees", "pomdp"])
def test_resolved_goal_pursuit_rejected_before_simulation(
    *, approved_manifest, tmp_path, monkeypatch, method
):
    from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(approved_manifest))
    monkeypatch.setattr(
        SweepSimpleEnvironment,
        "__init__",
        lambda *args, **kwargs: pytest.fail("Simulator constructed before preflight"),
    )
    args = argparse.Namespace(sweep_manifest=path, method=method, goal_pursuit_horizon=None)
    with pytest.raises(ValueError, match="goal-pursuit-horizon"):
        SweepSimpleCli.run_method(
            args=args,
            method_factory=lambda context: None,
            num_cycles=50,
            max_steps_per_interaction=20,
        )


@pytest.mark.parametrize(
    ("name", "bounds"),
    [
        ("PickFloorWiper", ((0.54, 0.85), (-np.pi / 12, np.pi / 12))),
        ("SweepCubeToGoal", ((0.40, 0.71), (-np.pi / 12, np.pi / 12))),
        ("SweepCubeToGoal", ((0.40, 0.70), (-0.27, np.pi / 12))),
    ],
)
def test_simple_support_cannot_expand_approved_ranges(*, name, bounds):
    pytest.importorskip("shapely")
    from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
    from hitl_pmp.environments.sweep_simple3d.skill_provider import SweepSimpleSkillProvider

    provider = SweepSimpleSkillProvider(env=SweepSimpleEnvironment())
    provider.stock_parameter_bounds[name] = bounds
    with pytest.raises(ValueError, match="approved"):
        provider.validate_trainable_support()
    assert provider.env._session is None


@pytest.mark.parametrize(
    "calibration",
    [None, {"PickWiper": 0.25, "Sweep": 0.25}, {"PickFloorWiper": 0.25, "SweepCubeToGoal": 0.5}],
)
def test_simple_factory_uses_domain_trainable_names_and_requires_calibration(
    *, tmp_path, calibration
):
    pytest.importorskip("shapely")
    from hitl_pmp.environments.sweep_simple3d.environment import SweepSimpleEnvironment
    from hitl_pmp.environments.sweep_simple3d.skill_provider import (
        SweepSimpleOracle,
        SweepSimpleSkillProvider,
    )
    from hitl_pmp.environments.sweep_simple3d.symbolic import SimpleSymbols

    manifest = {
        "pomdp": {
            "pomdp_inference_engine": "grid",
            "pomdp_num_particles": 8,
            "pomdp_search_depth": 20,
            "pomdp_solver": "determinized_astar",
        },
        # Synthetic test inputs, not scientific calibration.
        "random_competences": calibration,
        "deployment_horizon": 10,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = argparse.Namespace(
        env="sweep_simple3d",
        sweep_manifest=path,
        output_dir=tmp_path / "out",
        seed=0,
        ees_reset_gate=False,
        record_sampler_draws=False,
        num_cycles=50,
        max_steps_per_interaction=20,
    )
    env = SweepSimpleEnvironment()
    context = DomainContext(
        env=env, skill_provider=SweepSimpleSkillProvider(env=env), oracle=SweepSimpleOracle(env=env)
    )
    captured = {}

    class CaptureCli:
        @staticmethod
        def run_method(**kwargs):
            captured["method"] = kwargs["method_factory"](context)

    if calibration is None or "PickWiper" in calibration:
        with pytest.raises(ValueError, match="random_competences"):
            SweepPomdpCli.run(args=args, env_cli=CaptureCli)
    else:
        SweepPomdpCli.run(args=args, env_cli=CaptureCli)
        method = captured["method"]
        assert method.trainable_skill_names == SimpleSymbols.TRAINABLE
        assert method.random_competences == calibration
        assert method.deployment_horizon == 10
        saved = json.loads((args.output_dir / "sweep_resolved_method.json").read_text())
        assert saved["trainable_skill_names"] == list(SimpleSymbols.TRAINABLE)
    assert env._session is None
