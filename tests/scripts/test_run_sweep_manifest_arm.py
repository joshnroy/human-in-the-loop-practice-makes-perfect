"""Manifest launch routing and GPoff flags are checked without spawning runs."""

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("environment", [None, "sweep_drawer3d", "sweep_simple3d", "typo"])
@pytest.mark.parametrize("method", ["ees", "pomdp"])
def test_manifest_routes_environment_and_disables_goal_pursuit(
    *, monkeypatch, tmp_path, environment, method
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    launcher = importlib.import_module("run_sweep_manifest_arm")
    memory = importlib.import_module("check_slurm_memory")
    monkeypatch.setattr(memory, "main", lambda: None)
    manifest = {
        "arms": [
            {
                "name": "test",
                "method": method,
                "human_reset": True,
                "human_reset_practice_cost": 1.0,
            }
        ],
        "num_cycles": 50,
        "max_steps_per_interaction": 20,
        "num_test_tasks": 10,
    }
    if environment is not None:
        manifest["environment"] = environment
    if environment == "sweep_simple3d":
        manifest.update(
            status="FROZEN", deployment_horizon=10, practice_reset_policy="never",
            valid_practice_seeds=[0], valid_evaluation_seeds=list(range(10000, 10010)),
            pomdp=dict(
                goal_pursuit_horizon=0, pomdp_solver="determinized_astar",
                pomdp_max_search_iterations=1000, pomdp_search_depth=20,
                pomdp_inference_engine="grid", pomdp_grid_competence_bins=25,
                pomdp_grid_learning_rate_bins=16, pomdp_num_particles=1024,
                pomdp_linear_cost_lambda=3e-6,
            ),
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "--manifest", str(path), "--arm", "test", "--output", str(tmp_path / "out")],
    )
    captured = []

    def execute(**kwargs):
        captured.extend(kwargs["runs"])
        return []

    monkeypatch.setattr(launcher.SweepRunner, "execute", execute)
    if environment == "typo":
        with pytest.raises(ValueError, match="environment"):
            launcher.main()
        assert not captured
        return
    assert launcher.main() == 0
    assert len(captured) == 1
    run = captured[0]
    assert run.command[run.command.index("--env") + 1] == (environment or "sweep_drawer3d")
    assert run.command[run.command.index("--goal-pursuit-horizon") + 1] == "0"
