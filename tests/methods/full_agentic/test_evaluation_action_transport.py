"""Frozen controllers may return numeric arrays as well as Python lists."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def run_worker(*, monkeypatch, action):
    """Exercise the worker's real action path and strict JSON encoding offline."""
    calls = []

    class Relay:
        @staticmethod
        def request(*, payload):
            calls.append(json.loads(json.dumps(payload, allow_nan=False)))
            return {"action_spec": {}}

    class Approach:
        def __init__(self, *args):
            pass

        def reset(self, *args):
            pass

        def get_action(self, observation):  # noqa: PLR0917 -- deployment interface
            return action

    monkeypatch.setitem(sys.modules, "policy_worker", SimpleNamespace(ContainerPolicyWorker=Relay))
    monkeypatch.setitem(sys.modules, "approach", SimpleNamespace(GeneratedApproach=Approach))
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):  # noqa: PLR0917 -- Path method replacement
        if str(path) == "/sandbox/evaluation_config.json":
            return '{"max_steps": 1}'
        if str(path) == "/sandbox/submission/robot_spec.json":
            return '{"action_spec": {}}'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(sys, "path", list(sys.path))
    worker_path = Path(__file__).parents[3] / "src/hitl_pmp/full_agentic/evaluation_worker.py"
    spec = importlib.util.spec_from_file_location("evaluation_transport_test", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.main()
    return calls


@pytest.mark.parametrize(
    "action",
    [
        np.linspace(-0.1, 0.1, 18),
        np.full((100, 18), 0.01, dtype=np.float32),
        [np.float32(0.1)] * 18,
        [0.0] * 18,
    ],
)
def test_numeric_actions_reach_relay_unchanged(*, monkeypatch, action):
    calls = run_worker(monkeypatch=monkeypatch, action=action)
    steps = [call for call in calls if call["operation"] == "step"]
    assert len(steps) == 1
    np.testing.assert_array_equal(steps[0]["action"], action)
    assert calls[-1] == {"operation": "finish", "done": True}


def test_none_finishes_without_dispatching_action(*, monkeypatch):
    calls = run_worker(monkeypatch=monkeypatch, action=None)
    assert not any(call["operation"] == "step" for call in calls)
    assert calls[-1] == {"operation": "finish", "done": True}


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_actions_are_still_rejected(*, monkeypatch, invalid):
    calls = run_worker(monkeypatch=monkeypatch, action=np.full(18, invalid))
    assert not any(call["operation"] == "step" for call in calls)
    assert calls[-1]["done"] is False
    assert "Out of range float" in calls[-1]["error"]
