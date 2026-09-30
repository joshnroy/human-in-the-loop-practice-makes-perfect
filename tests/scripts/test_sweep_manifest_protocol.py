"""The launcher must explicitly disable goal pursuit for both methods."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("method", ["ees", "pomdp"])
@pytest.mark.parametrize("smoke", [False, True])
def test_launcher_freezes_goal_pursuit(*, tmp_path, monkeypatch, method, smoke):
    captured = {}

    def plan(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setitem(
        sys.modules,
        "run_sweep",
        SimpleNamespace(
            SpawnRetryPolicy=lambda **kwargs: kwargs,
            SweepRunner=SimpleNamespace(
                plan=plan, new_sweep_id=lambda: "test", execute=lambda **kwargs: []
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "check_slurm_memory", SimpleNamespace(main=lambda: None))
    spec = importlib.util.spec_from_file_location(
        "manifest_launcher",
        Path(__file__).resolve().parents[2] / "scripts/run_sweep_manifest_arm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                num_cycles=50,
                max_steps_per_interaction=20,
                num_test_tasks=10,
                arms=[
                    dict(name="test", method=method, human_reset=True, human_reset_practice_cost=1)
                ],
            )
        )
    )
    argv = ["launcher", "--manifest", str(manifest), "--arm", "test", "--output", str(tmp_path)]
    if smoke:
        argv.append("--smoke")
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    flags = captured["shared_args"]
    assert "--goal-pursuit-horizon" in flags
    assert flags[flags.index("--goal-pursuit-horizon") + 1] == "0"
