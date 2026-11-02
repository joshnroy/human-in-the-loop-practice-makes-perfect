"""Episode scoring uses live delivery/readiness/velocity, not symbolic effects alone."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.problem import RackServingProblem
from hitl_pmp.environments.racktoserving3d.semantics import RackServingLedger
from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks


class _ObservedState(State):
    data: dict[Object, np.ndarray]
    abstract_atoms: frozenset[tuple[str, tuple[str, ...]]]


def _setup(*, monkeypatch, speed=0.0, elapsed=300.0):
    state = _ObservedState(
        data={},
        abstract_atoms=frozenset({
            ("Ready", ()),
            *(("OnTray", (food.name, Objects.tray.name)) for food in Objects.food),
        }),
    )
    env = RackServingEnvironment(current_state=state)
    backend = env.backend()
    backend.ledger = RackServingLedger(elapsed_seconds=elapsed)
    names = [food.name for food in Objects.food]
    locations = dict.fromkeys(names, "OnTray")
    velocities = np.zeros(6 * len(names))
    velocities[0] = speed
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=names.index(name)),
        body_jntadr=np.arange(len(names)),
        jnt_dofadr=np.arange(0, len(velocities), 6),
    )
    monkeypatch.setattr(RackServingBackend, "sync_clock", Mock())
    monkeypatch.setattr(RackServingBackend, "model", lambda backend: model)
    monkeypatch.setattr(
        RackServingBackend, "data", lambda backend: SimpleNamespace(qvel=velocities)
    )
    monkeypatch.setattr(RackServingBackend, "location", lambda backend, *, name: locations[name])
    monkeypatch.setattr(
        RackServingEnvironment,
        "set_state",
        lambda env, *, state: setattr(env, "current_state", state),
    )
    monkeypatch.setattr(RackServingEnvironment, "take_action", lambda env, *, action: state)
    tasks = RackServingTasks(env=env)
    problem = RackServingProblem(env=env, tasks=tasks)
    task = Task(initial_state=state, goal=tasks.goal())
    assert task.goal.is_satisfied(state=state)
    return problem, task, velocities, locations


def test_symbolic_delivery_does_not_stop_or_score_moving_food(*, monkeypatch):
    problem, task, _, _ = _setup(monkeypatch=monkeypatch, speed=0.03)
    policy = Mock(return_value=LabeledAction(action=np.array([-1, 0, 0, 0]), label="no-op"))
    success, frames, trace = problem.run_task_episode(task=task, policy=policy)
    assert not success
    assert not frames
    assert policy.call_count == problem.max_episode_steps()
    assert len(trace.actions) == problem.max_episode_steps()
    assert len(trace.states) == len(trace.actions) + 1


def test_settled_delivery_stops_without_requiring_a_sweep(*, monkeypatch):
    problem, task, _, _ = _setup(monkeypatch=monkeypatch)
    policy = Mock(side_effect=AssertionError("already delivered"))
    success, _, trace = problem.run_task_episode(task=task, policy=policy)
    assert success
    assert not trace.actions
    policy.assert_not_called()


@pytest.mark.parametrize("failure", ["not_ready", "off_tray"])
def test_live_measurements_override_stale_symbolic_goal(*, monkeypatch, failure):
    problem, task, _, locations = _setup(
        monkeypatch=monkeypatch, elapsed=299.0 if failure == "not_ready" else 300.0
    )
    if failure == "off_tray":
        locations[Objects.food[-1].name] = "SpilledCounter"
    policy = Mock(return_value=LabeledAction(action=np.array([-1, 0, 0, 0]), label="no-op"))
    success, _, _ = problem.run_task_episode(task=task, policy=policy)
    assert not success


def test_last_action_settling_is_scored_from_live_backend(*, monkeypatch):
    problem, task, velocities, _ = _setup(monkeypatch=monkeypatch, speed=0.03)
    monkeypatch.setattr(RackServingProblem, "max_episode_steps", lambda problem: 1)

    def settle(*, action):
        velocities[:] = 0
        return task.initial_state

    monkeypatch.setattr(RackServingEnvironment, "take_action", staticmethod(settle))
    policy = Mock(return_value=LabeledAction(action=np.array([-1, 0, 0, 0]), label="no-op"))
    success, _, trace = problem.run_task_episode(task=task, policy=policy)
    assert success
    assert len(trace.actions) == 1


def test_autonomous_pick_place_delivery_scores_success_without_sweep(*, monkeypatch):
    problem, task, _, locations = _setup(monkeypatch=monkeypatch)
    locations.update(dict.fromkeys(locations, "OnRack"))
    actions = [
        LabeledAction(action=np.array([skill_id, index, 0, 0]), label=label)
        for index in range(len(Objects.food))
        for skill_id, label in ((3, "PickFood"), (4, "PlaceFoodOnTray"))
    ]

    def deliver(*, action):
        name = Objects.food[int(action[1])].name
        locations[name] = "HoldingFood" if action[0] == 3 else "OnTray"
        return task.initial_state

    monkeypatch.setattr(RackServingEnvironment, "take_action", staticmethod(deliver))
    policy = Mock(side_effect=actions)
    success, _, trace = problem.run_task_episode(task=task, policy=policy)
    assert success
    assert len(trace.actions) == 2 * len(Objects.food)
    assert all(action.label in {"PickFood", "PlaceFoodOnTray"} for action in trace.actions)
