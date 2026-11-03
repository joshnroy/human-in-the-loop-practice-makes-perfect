"""Real rack facade/backend hooks driven by fake physics, never a native scene."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached, ControlStepObserverFailed
from hitl_pmp.environments.racktoserving3d import backend as backend_module
from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.semantics import RackServingLedger
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.step_protocol import PracticeClock


@pytest.fixture
def scene(*, monkeypatch):
    events = []
    data = SimpleNamespace(time=0.0)

    def physics(*args, **kwargs):
        events.append("physics")
        data.time += 0.1
        return "step-result"

    raw_step = Mock(side_effect=physics)
    native = SimpleNamespace(step=raw_step, close=Mock())
    kinder = SimpleNamespace(register_all_environments=Mock(), make=Mock(return_value=native))
    monkeypatch.setitem(sys.modules, "kinder", kinder)
    monkeypatch.setitem(
        sys.modules, "kinder.envs.dynamic3d.envs", SimpleNamespace(TidyBot3DConfig=Mock())
    )
    monkeypatch.setattr(backend_module, "install_geometry", Mock())
    monkeypatch.setattr(backend_module, "build_scene", Mock(return_value={}))
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    backend = RackServingBackend()
    try:
        yield SimpleNamespace(
            backend=backend,
            events=events,
            data=data,
            native=native,
            raw_step=raw_step,
            kinder=kinder,
        )
    finally:
        backend.close()


@pytest.fixture
def rack(*, scene, monkeypatch):
    backend = scene.backend
    backend.env()
    atoms = set()
    held = set()

    def observed_atoms():
        backend.sync_clock()
        return frozenset(atoms)

    monkeypatch.setattr(RackServingBackend, "abstract_atoms", staticmethod(observed_atoms))
    monkeypatch.setattr(
        RackServingBackend,
        "features",
        lambda self: {obj.name: np.zeros(obj.type.dim) for obj in Objects.all_objects()},
    )
    monkeypatch.setattr(RackServingBackend, "held", lambda self, *, name: name in held)
    env = RackServingEnvironment()
    env._backend = backend
    shelf = SimpleNamespace(
        ticks=0,
        grip=0.7,
        held_cube="cube_4",
        in_hand=np.eye(4),
        site_pose=Mock(return_value=np.eye(4)),
    )
    tool = SimpleNamespace(wiper_pose=Mock(return_value=np.eye(4)), initialize_rest=Mock())

    def run_skill():
        for _ in range(10):
            backend.env().step(np.zeros(11))
            shelf.ticks += 1

    controllers = SimpleNamespace(
        shelf=shelf,
        tool=tool,
        held_food="cube_4",
        pick_scraper=run_skill,
        steps=999,
    )
    env._rack_controllers = controllers
    env.current_state = env._observed_state(seed=23, steps_taken=4)
    monkeypatch.setattr(
        RackServingEnvironment,
        "controllers",
        Mock(side_effect=AssertionError("use rack_controllers")),
    )
    return SimpleNamespace(
        env=env,
        backend=backend,
        controllers=controllers,
        atoms=atoms,
        held=held,
        scene=scene,
    )


def test_backend_observers_survive_creation_replace_detach_and_do_not_stack(*, scene):
    backend = scene.backend
    backend.set_step_observers(
        before=lambda: scene.events.append("before"),
        after=lambda: scene.events.append("after"),
    )
    backend.set_tick_observer(observer=lambda: scene.events.append("tick"))
    assert backend._env is None
    assert backend.env().step(np.zeros(11)) == "step-result"
    assert scene.events == ["before", "physics", "after", "tick"]
    scene.events.clear()
    backend.set_step_observers(before=None, after=lambda: scene.events.append("replacement"))
    backend.env().step(np.zeros(11))
    assert scene.events == ["physics", "replacement", "tick"]
    scene.events.clear()
    backend.set_step_observers(before=None, after=None)
    backend.env().step(np.zeros(11))
    assert scene.events == ["physics", "tick"]
    assert backend.control_steps() == 3
    scene.kinder.make.assert_called_once()


@pytest.mark.parametrize("phase", ["before", "after", "tick"])
def test_observer_exceptions_escape_controller_failure_handling(*, scene, phase):
    observer = Mock(side_effect=ValueError("observer failed"))
    scene.backend.set_step_observers(
        before=observer if phase == "before" else None,
        after=observer if phase == "after" else None,
    )
    scene.backend.set_tick_observer(observer=observer if phase == "tick" else None)
    with pytest.raises(ControlStepObserverFailed) as caught:
        scene.backend.env().step(np.zeros(11))
    assert isinstance(caught.value.__cause__, ValueError)
    assert scene.backend.control_steps() == (0 if phase == "before" else 1)


def test_failed_physics_does_not_count_or_call_after_observers(*, scene):
    after = Mock()
    tick = Mock()
    scene.raw_step.side_effect = RuntimeError("physics failed")
    scene.backend.set_step_observers(before=None, after=after)
    scene.backend.set_tick_observer(observer=tick)
    with pytest.raises(RuntimeError, match="physics failed"):
        scene.backend.env().step(np.zeros(11))
    after.assert_not_called()
    tick.assert_not_called()
    assert scene.backend.control_steps() == 0


@pytest.mark.parametrize("stop_after", [False, True])
@pytest.mark.parametrize("held", [None, "cube_1", "wiper_0"])
def test_cutoff_adopts_exact_completed_steps_and_live_state_without_physics(
    *, rack, stop_after, held
):
    env, backend = rack.env, rack.backend
    backend.env().step(np.zeros(11))
    clock = PracticeClock(budget=3, interval=100, robot_active=True, on_measure=lambda: None)

    def after():
        clock.after_robot_step()
        if stop_after and clock.steps == 3:
            raise ControlStepLimitReached()

    env.set_control_step_observers(before=clock.before_robot_step, after=after)
    action = np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0], dtype=float)
    with pytest.raises(ControlStepLimitReached):
        env.take_action(action=action)
    assert clock.steps == env.last_controller_steps() == rack.controllers.shelf.ticks == 3
    assert rack.scene.raw_step.call_count == 4
    env.set_control_step_observers(before=None, after=None)
    rack.atoms.add(("Ready", ()))
    if held is not None:
        rack.held.add(held)
    state = env.adopt_interrupted_skill(action=action, reason="practice_step_budget")
    assert state is env.current_state
    assert state.get(obj=Objects.scene, feature_name="seed") == 23
    assert state.get(obj=Objects.scene, feature_name="steps_taken") == 5
    assert state.get(obj=Objects.scene, feature_name="elapsed_seconds") == pytest.approx(0.4)
    assert state.abstract_atoms == frozenset(rack.atoms)
    assert env.last_skill_error() == "interrupted: practice_step_budget"
    assert rack.scene.raw_step.call_count == 4
    assert backend.grip == rack.controllers.shelf.grip
    expected_food = held if held == "cube_1" else None
    assert rack.controllers.held_food == rack.controllers.shelf.held_cube == expected_food
    assert (rack.controllers.shelf.in_hand is not None) == (held == "wiper_0")
    attempt = env.last_skill_attempt()
    assert attempt[1] == frozenset()
    assert attempt[2] == frozenset(rack.atoms)
    action[:] = -99
    assert attempt[0][0] == SKILL_IDS["PickScraper"]


def test_normal_skill_and_noop_report_actual_steps_not_controller_estimates(*, rack):
    rack.env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0]))
    assert rack.env.last_controller_steps() == rack.controllers.shelf.ticks == 10
    rack.env.take_action(action=np.array([-1, 0, 0, 0, 0]))
    assert rack.env.last_controller_steps() == rack.controllers.shelf.ticks == 0


def test_observer_failure_is_not_reported_as_a_robot_skill_failure(*, rack):
    rack.env.set_control_step_observers(
        before=None, after=Mock(side_effect=ValueError("logging failed"))
    )
    with pytest.raises(ControlStepObserverFailed):
        rack.env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0]))
    assert rack.env.last_controller_steps() == rack.controllers.shelf.ticks == 1
    assert rack.env.last_skill_error() is None


def test_cutoff_before_first_step_does_not_adopt_previous_skill_count(*, rack):
    rack.env.take_action(action=np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0]))
    clock = PracticeClock(
        budget=1, interval=100, robot_active=True, robot_steps=1, on_measure=lambda: None
    )
    rack.env.set_control_step_observers(
        before=clock.before_robot_step, after=clock.after_robot_step
    )
    action = np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0])
    with pytest.raises(ControlStepLimitReached):
        rack.env.take_action(action=action)
    rack.env.adopt_interrupted_skill(action=action, reason="practice_step_budget")
    assert rack.env.last_controller_steps() == rack.controllers.shelf.ticks == 0
    assert rack.scene.raw_step.call_count == 10


@pytest.mark.parametrize("destination", ["human_batch_return", "human_tool_recovery"])
def test_failed_human_attempt_keeps_charge_and_updates_observed_clock(
    *, rack, monkeypatch, destination
):
    backend = rack.backend
    backend.ledger = RackServingLedger(elapsed_seconds=400)
    rack.scene.data.time = 2.0

    def fail(*args, **kwargs):
        rack.scene.data.time += 3.0
        raise RuntimeError("human attempt failed")

    if destination == "human_batch_return":
        monkeypatch.setattr(RackServingBackend, "food_placements", fail)
    else:
        rack.controllers.tool.initialize_rest.side_effect = fail
    with pytest.raises(RuntimeError, match="human attempt failed"):
        rack.env.reset_movables(destination=destination)
    backend.sync_clock()
    assert backend.ledger.elapsed_seconds == 405
    assert backend.ledger.human_cost == 5
    assert backend.ledger.human_interventions == 1
    state = rack.env.get_current_state()
    assert state.get(obj=Objects.scene, feature_name="human_cost") == 5
    assert state.get(obj=Objects.scene, feature_name="elapsed_seconds") == 405
    assert state.get(obj=Objects.scene, feature_name="steps_taken") == 5
    assert rack.env._human_events[-1].skill == destination
    assert rack.env._human_events[-1].moved == ()


@pytest.mark.parametrize("fails", [False, True])
def test_human_physics_does_not_spend_robot_budget_and_charge_survives_failure(*, rack, fails):
    clock = PracticeClock(budget=5, interval=100, on_measure=lambda: None)
    rack.env.set_control_step_observers(
        before=clock.before_robot_step, after=clock.after_robot_step
    )

    def recover():
        for _ in range(2):
            rack.backend.env().step(np.zeros(11))
        if fails:
            raise RuntimeError("human failed after physics")

    rack.controllers.tool.initialize_rest.side_effect = recover
    charge = clock.before_human(skill="human_tool_recovery")
    try:
        if fails:
            with pytest.raises(RuntimeError, match="human failed"):
                rack.env.reset_movables(destination="human_tool_recovery")
        else:
            assert rack.env.reset_movables(destination="human_tool_recovery")
    finally:
        clock.human_invoked(charge=charge)
    rack.backend.sync_clock()
    rack.backend.sync_clock()
    assert rack.backend.ledger.elapsed_seconds == pytest.approx(0.2)
    assert rack.backend.ledger.human_cost == 5
    assert rack.backend.ledger.human_interventions == 1
    assert clock.steps == clock.human_invocations == 1
    assert clock.robot_steps == 0
    clock.robot_active = True
    action = np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0])
    with pytest.raises(ControlStepLimitReached):
        rack.env.take_action(action=action)
    rack.env.adopt_interrupted_skill(action=action, reason="practice_step_budget")
    assert clock.steps == 5
    assert clock.robot_steps == rack.env.last_controller_steps() == 4
    assert rack.backend.control_steps() == 6


def test_invalid_human_destination_is_not_charged_or_observed(*, rack):
    state = rack.env.current_state
    with pytest.raises(ValueError, match="unknown human"):
        rack.env.reset_movables(destination="reset_cubes")
    assert rack.env.current_state is state
    assert rack.backend.ledger.human_interventions == 0
    assert rack.env._human_events == []
