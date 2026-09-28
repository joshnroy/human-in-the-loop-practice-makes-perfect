"""Factory metadata matches resolved settings without constructing a simulator."""

import argparse
import json

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.sweep_drawer3d.environment import SweepDrawerEnvironment
from hitl_pmp.environments.sweep_drawer3d.skill_provider import (
    SweepDrawerOracle,
    SweepDrawerSkillProvider,
)
from hitl_pmp.methods.belief_space.sweep_cli import SweepPomdpCli


def test_manifest_engine_gate_recorder_and_snapshot_match(*, tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {
        "pomdp": {
            "pomdp_inference_engine": "grid",
            "pomdp_num_particles": 1024,
            "pomdp_search_depth": 20,
            "pomdp_observation_probability_weight": 0.0,
            "pomdp_linear_cost_lambda": 3e-6,
        },
        "random_competences": {name: 0.25 for name in ("OpenDrawer", "PickWiper", "Sweep")},
        "deployment_horizon": 5,
    }
    path.write_text(json.dumps(manifest))
    args = argparse.Namespace(
        sweep_manifest=path,
        output_dir=tmp_path / "out",
        seed=0,
        ees_reset_gate=False,
        record_sampler_draws=True,
        num_cycles=50,
        max_steps_per_interaction=20,
        pomdp_inference_engine="particle",
    )
    env = SweepDrawerEnvironment()
    context = DomainContext(
        env=env,
        skill_provider=SweepDrawerSkillProvider(env=env, human_reset_enabled=False),
        oracle=SweepDrawerOracle(env=env),
    )
    captured = {}

    class CaptureCli:
        @staticmethod
        def run_method(**kwargs):
            captured["method"] = kwargs["method_factory"](context)

    SweepPomdpCli.run(args=args, env_cli=CaptureCli)
    method = captured["method"]
    assert method.pomdp_inference_engine == args.pomdp_inference_engine == "grid"
    assert not method.reset_cost_gate
    assert method.draw_recorder is not None
    assert method.pomdp_search_depth == args.pomdp_search_depth == 20
    assert method.pomdp_observation_probability_weight == 0.0
    saved = json.loads((args.output_dir / "sweep_resolved_method.json").read_text())
    assert saved["pomdp_inference_engine"] == "grid"
    assert saved["pomdp_num_particles"] == 1024
    assert saved["reset_cost_gate"] is False
    assert env._session is None
