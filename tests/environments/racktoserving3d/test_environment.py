import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.semantics import RackServingLedger, SupportRegion


@pytest.mark.parametrize("destination", ["human_batch_return", "human_tool_recovery"])
def test_deployment_rejects_human_before_any_physical_write(*, monkeypatch, destination):
    backend = RackServingBackend(
        config=RackServingConfig(mode="deployment"),
        ledger=RackServingLedger(mode="deployment", elapsed_seconds=123),
    )
    env = RackServingEnvironment(config=backend.config)
    env._backend = backend
    monkeypatch.setattr(RackServingEnvironment, "_bookkeeping", lambda self: (0, 0))
    monkeypatch.setattr(RackServingBackend, "sync_clock", lambda self: None)
    writes = []
    monkeypatch.setattr(RackServingBackend, "_commit", lambda self, **kwargs: writes.append(kwargs))
    before = backend.ledger
    with pytest.raises(PermissionError, match="training-only"):
        env.reset_movables(destination=destination)
    assert not writes
    assert backend.ledger == before


def test_batch_return_is_charged_and_does_not_restart_the_clock(*, monkeypatch):
    backend = RackServingBackend(ledger=RackServingLedger(elapsed_seconds=400))
    monkeypatch.setattr(RackServingBackend, "sync_clock", lambda self: None)
    monkeypatch.setattr(RackServingBackend, "food_placements", lambda self, **kwargs: {})
    monkeypatch.setattr(RackServingBackend, "_commit", lambda self, **kwargs: None)
    backend.batch_return()
    assert backend.ledger.elapsed_seconds == 400
    assert backend.ledger.human_cost == 5
    assert backend.ledger.human_interventions == 1


def test_failed_human_attempt_is_still_an_interruption(*, monkeypatch):
    backend = RackServingBackend()
    monkeypatch.setattr(RackServingBackend, "sync_clock", lambda self: None)

    def unavailable(*args, **kwargs):
        raise RuntimeError("no clear return placement")

    monkeypatch.setattr(RackServingBackend, "food_placements", unavailable)
    with pytest.raises(RuntimeError):
        backend.batch_return()
    assert backend.ledger.human_cost == 5


def test_tool_recovery_does_not_drop_held_food(*, monkeypatch):
    backend = RackServingBackend()
    env = RackServingEnvironment(config=backend.config)
    env._backend = backend
    monkeypatch.setattr(RackServingEnvironment, "_bookkeeping", lambda self: (0, 0))
    monkeypatch.setattr(RackServingEnvironment, "_observed_state", lambda self, **kwargs: None)
    monkeypatch.setattr(RackServingBackend, "sync_clock", lambda self: None)
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: True)
    with pytest.raises(RuntimeError, match="held food"):
        env.reset_movables(destination="human_tool_recovery")
    assert backend.ledger.human_cost == 5


def test_near_tray_without_physical_support_is_not_delivered(*, monkeypatch):
    backend = RackServingBackend()
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: False)
    monkeypatch.setattr(
        RackServingBackend,
        "corners",
        lambda self, **kwargs: np.array([[1.18, -0.01, 0.912], [1.22, 0.01, 0.932]]),
    )
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, **kwargs: SupportRegion(
            x_min=1.0,
            x_max=1.35,
            y_min=-0.2,
            y_max=0.2,
            top_z=0.912,
        ),
    )
    monkeypatch.setattr(RackServingBackend, "supporting_bodies", lambda self, **kwargs: set())
    assert backend.location(name="cube_0") is None
    monkeypatch.setattr(
        RackServingBackend,
        "supporting_bodies",
        lambda self, **kwargs: {"box_0"},
    )
    assert backend.location(name="cube_0") == "OnTray"


def _dispatching_env(*, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    env = RackServingEnvironment(config=RackServingConfig())
    backend = RackServingBackend(config=env.config)
    env._backend = backend
    controllers = SimpleNamespace(
        shelf=SimpleNamespace(ticks=0), gather_and_sweep=Mock(), wait_until_ready=Mock()
    )
    env._rack_controllers = controllers
    monkeypatch.setattr(RackServingEnvironment, "_bookkeeping", lambda self: (0, 0))
    monkeypatch.setattr(RackServingEnvironment, "_observed_state", lambda self, **kwargs: None)
    monkeypatch.setattr(RackServingBackend, "control_steps", lambda self: 0)
    monkeypatch.setattr(RackServingEnvironment, "get_current_state", lambda self: None)
    return env, controllers


def test_gather_and_sweep_action_carries_squeeze_gather_and_depth(*, monkeypatch):
    env, controllers = _dispatching_env(monkeypatch=monkeypatch)
    env.take_action(action=np.array([1, 0, 0.07, 0.015, 0.10]))
    controllers.gather_and_sweep.assert_called_once_with(s=0.07, delta=0.015, d=0.10)
    assert env._last_skill_error is None


def test_actions_are_five_wide_and_the_no_op_matches(*, monkeypatch):
    env, controllers = _dispatching_env(monkeypatch=monkeypatch)
    env.take_action(action=np.array([1, 0, 0.0, 0.0]))
    assert "five finite entries" in env._last_skill_error
    controllers.gather_and_sweep.assert_not_called()
    assert env.noop_action().shape == (5,)
    env.take_action(action=env.noop_action())
    assert env._last_skill_error is None
