"""Replay actual premature-reset states through the charged toss execution loop."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.practice_costs import PracticeAccounting, PracticeCosts
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend, TossSettlingController
from hitl_pmp.environments.tossing3d.predicates import HOLDING, ON_GROUND, PICK_PLANNABLE
from hitl_pmp.step_protocol import PracticeClock

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER unavailable"),
    pytest.mark.xdist_group(name="toss_settling"),
]


class _FinishedToss:
    """The fixture starts at the instant upstream's toss controller terminated."""

    def reset(self, state, params):  # noqa: PLR0917 (KINDER signature)
        del state, params

    def terminated(self):
        return True


@pytest.fixture
def settling_scene(*, monkeypatch, tmp_path):
    original = KinderBackend._ground_controller

    def ground(self, *, key, **kwargs):  # noqa: PLR0917 (bound method)
        if key == "move_to_toss_location_and_toss":
            return TossSettlingController(backend=self, controller=_FinishedToss())
        return original(self, key=key, **kwargs)

    monkeypatch.setattr(KinderBackend, "_ground_controller", ground)
    # The historical snapshots used a movable 1 kg bin and a 100 g cube.
    # Keep that scene explicit without changing the production scene in this PR.
    api = KinderBackend().api()
    path = Path(api.kinder.__file__).parent / "envs/dynamic3d/tasks/Tossing3D/Tossing3D-o1.json"
    config = json.loads(path.read_text())
    config["objects"]["bin"]["bin_0"].update(fixed=False, mass=1.0)
    config["objects"]["cube"]["cube_0"]["mass"] = 0.1
    task_path = tmp_path / "settling-scene.json"
    task_path.write_text(json.dumps(config))
    env = Tossing3DEnvironment()
    env._backend = KinderBackend(task_config_path=task_path)
    env.hard_reset()
    try:
        yield env
    finally:
        env.close()


def _clock(*, env, budget):
    costs = PracticeCosts.load(path="src/hitl_pmp/presets/tossing3d_tick_priced_costs.json")
    accounting = PracticeAccounting(costs=costs)
    clock = PracticeClock(
        budget=budget,
        interval=1000,
        human_steps=1,
        on_measure=lambda: None,
        on_progress=lambda: None,
        accounting=accounting,
    )
    env.set_control_step_observers(before=clock.before_robot_step, after=clock.after_robot_step)
    clock.robot_active = True
    return clock, accounting


def _restore(*, env, name):
    path = Path(__file__).parent / "fixtures/toss_settling" / f"{name}.json"
    return env.restore_plain_snapshot(plain=json.loads(path.read_text())["state"])


@pytest.mark.parametrize("fixture", ["falling_front", "falling_fast", "falling_side"])
def test_settling_enables_pick_and_charges_every_tick(*, settling_scene, fixture):
    env = settling_scene
    state = _restore(env=env, name=fixture)
    assert not ON_GROUND.holds(state, (env.cube,))
    clock, accounting = _clock(env=env, budget=500)
    state = env.take_action(action=np.array([1.0, 1.3, 0.0, 140.0, 792.0]))
    ticks = sum(env.last_controller_steps())
    assert 3 <= ticks <= 20
    assert clock.steps == ticks
    assert accounting.total_cost == pytest.approx(ticks * 0.01)
    assert len(env.kinder_backend().drain_substep_states()) == 0
    assert ON_GROUND.holds(state, (env.cube,))
    assert PICK_PLANNABLE.holds(state, (env.robot, env.cube))
    state = env.take_action(action=np.zeros(5))
    assert env.last_skill_error() is None
    assert HOLDING.holds(state, (env.robot, env.cube))
    assert accounting.total_cost == pytest.approx(clock.steps * 0.01)


def test_practice_budget_interrupts_settling_without_free_ticks(*, settling_scene):
    env = settling_scene
    _restore(env=env, name="falling_front")
    clock, accounting = _clock(env=env, budget=2)
    with pytest.raises(ControlStepLimitReached):
        env.take_action(action=np.array([1.0, 1.3, 0.0, 140.0, 792.0]))
    assert clock.steps == 2
    assert accounting.total_cost == pytest.approx(0.02)
