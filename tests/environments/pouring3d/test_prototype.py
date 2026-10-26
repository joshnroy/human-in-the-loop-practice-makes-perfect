"""Physical outcome checks for the pouring proof of concept."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None, reason="MuJoCo not installed"
)


def test_physical_pour_and_both_resets():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0)
    assert sim.locations() == ["pitcher"] * 5
    sim.execute(skill="PickPitcher")
    assert sim.held == "pitcher"
    event = sim.execute(skill="Pour", parameters={"angle": 130.0})
    assert event["success"], sim.locations()
    sim.execute(skill="PlacePitcherAtHome")
    assert sim.locations() == ["tray"] * 5
    event = sim.execute(skill="HumanPourBack")
    assert event["success"], sim.locations()
    assert sim.ready()
    sim.execute(skill="PickPitcher")
    sim.execute(skill="Pour", parameters={"angle": 130.0})
    sim.execute(skill="PlacePitcherAtHome")
    for cube in range(5):
        sim.execute(skill="PickCubeFromTray", cube=cube)
        sim.execute(skill="DropCubeIntoPitcher", cube=cube)
    assert sim.ready(), sim.locations()
    assert sim.human_interventions == 1
    assert sim.elapsed_seconds > 0
    assert np.isfinite(sim.data.qpos).all()


def test_small_tilt_does_not_magically_transfer_cubes():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0)
    sim.execute(skill="PickPitcher")
    event = sim.execute(skill="Pour", parameters={"angle": 10.0})
    assert not event["success"]
    assert sim.locations() == ["pitcher"] * 5
    assert sim.human_interventions == 0


def test_invalid_reset_does_not_change_world_or_charge_cost():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0)
    before = sim.data.qpos.copy()
    elapsed = sim.elapsed_seconds
    with pytest.raises(ValueError, match="Preconditions unmet"):
        sim.execute(skill="HumanPourBack")
    np.testing.assert_array_equal(before, sim.data.qpos)
    assert sim.elapsed_seconds == elapsed
    assert sim.human_interventions == 0


def test_adapter_observes_actions_without_free_resets():
    from hitl_pmp.environments.pouring3d.environment import PouringEnvironment

    env = PouringEnvironment(seed=0)
    env.hard_reset()
    initial = env.get_current_state()
    assert env.world is not None
    np.testing.assert_array_equal(
        env.take_action(action=env.noop_action())[env.world], initial[env.world]
    )
    assert env.last_info["cost_seconds"] == 1.0
    actions = env.get_valid_actions()
    assert {int(a[0]) for a in actions} == {0, 7}
    pick = next(a for a in actions if a[0] == 0)
    env.take_action(action=pick)
    assert env.sim.held == "pitcher"
    env.take_action(action=np.array([1, 0, 130, 0, -0.055, 2, 2]))
    assert env.last_info["success"]
    assert env.last_info["after"]["locations"] == ["tray"] * 5
    before = env.observe()[env.world].copy()
    with pytest.raises(NotImplementedError):
        env.set_state(state=initial)
    np.testing.assert_array_equal(before, env.observe()[env.world])
    with pytest.raises(ValueError):
        env.take_action(action=np.array([1.5, 0, 130, 0, -0.055, 2, 2]))
