"""Failed physical skill calls must consume the approved readiness action budget."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("finish_selected_first", [False, True])
def test_failed_forward_calls_are_counted_without_reset(
    *, monkeypatch, finish_selected_first
) -> None:
    from hitl_pmp.environments.sweep_simple3d import environment, regions

    path = Path(__file__).resolve().parents[2] / "scripts/probe_sweepsimple_pick.py"
    spec = importlib.util.spec_from_file_location("simple_probe_budget_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = module.PickupProbe
    core = SimpleNamespace(_check_goals=lambda: False)
    session = SimpleNamespace(
        seed=0, ticks=0, env=SimpleNamespace(unwrapped=SimpleNamespace(_object_centric_env=core))
    )

    class FakeEnvironment:
        def __init__(self, **kwargs):
            del kwargs

        def observe(self):
            return None

    monkeypatch.setattr(environment, "SweepSimpleEnvironment", FakeEnvironment)
    monkeypatch.setattr(regions.SimpleRegions, "contains", lambda **kwargs: False)
    monkeypatch.setattr(probe, "native_counts", lambda **kwargs: {"sweep_region": 0})
    calls = []

    def fail_call(*, env, name, cube, report, params):
        del env, params
        calls.append(cube)
        report["stages"].append({"name": name, "success": False, "note": "physical failure"})
        raise RuntimeError("physical failure")

    monkeypatch.setattr(probe, "cycle_action", fail_call)
    report = {"stages": [
        {"name": "PickFloorWiper", "success": True},
        {"name": "SweepCubeToGoal", "success": False},
    ]}
    args = SimpleNamespace(
        forward_budget=10, cube_order=[0, 1, 2, 3, 4], sweep_distance=0.4, sweep_angle=0.0,
        finish_selected_first=finish_selected_first,
    )
    with pytest.raises(RuntimeError, match="All-cube native goal check failed"):
        probe.full_cycle(
            session=session, primitive=None, initial_state=None, args=args, report=report
        )
    assert calls == ([0] * 8 if finish_selected_first else [1, 2, 3, 4, 0, 1, 2, 3])
    assert report["forward_robot_actions_executed"] == 10
    assert sum(bool(s.get("counted_failure")) for s in report["stages"]) == 8
    assert report["native_goal_success"] is False
