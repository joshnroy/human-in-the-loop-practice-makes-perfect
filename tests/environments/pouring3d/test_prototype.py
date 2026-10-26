"""Physical outcomes of native KinDER pouring and persistent resets."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KinDER not installed"
)


def test_physical_pour_and_both_resets():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0, scene_bg=False)
    try:
        assert sim.locations() == ["pitcher"] * 5
        assert sim.execute(skill="PickPitcher")["success"]
        assert sim.execute(skill="Pour")["success"]
        assert sim.execute(skill="PlacePitcherAtHome")["success"]
        assert sim.locations() == ["tray"] * 5
        assert sim.execute(skill="HumanPourBack")["success"]
        assert sim.ready()
        assert sim.execute(skill="PickPitcher")["success"]
        assert sim.execute(skill="Pour")["success"]
        assert sim.execute(skill="PlacePitcherAtHome")["success"]
        for _ in range(5):
            cube = sim.controller.select_reset_cube()
            assert sim.execute(skill="PickCubeFromTray", cube=cube)["success"]
            assert sim.execute(skill="DropCubeIntoPitcher", cube=cube)["success"]
        assert sim.ready(), sim.locations()
        assert sim.human_interventions == 1
        assert sim.elapsed_seconds > 0
        assert np.isfinite(sim.data.qpos).all()
    finally:
        sim.close()


def test_small_tilt_does_not_magically_transfer_cubes():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0, scene_bg=False)
    try:
        assert sim.execute(skill="PickPitcher")["success"]
        event = sim.execute(skill="Pour", parameters={"angle": 10.0})
        assert not event["success"]
        assert sim.locations() == ["pitcher"] * 5
        assert sim.human_interventions == 0
    finally:
        sim.close()


def test_invalid_reset_does_not_change_world_or_charge_cost():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0, scene_bg=False)
    try:
        before = sim.data.qpos.copy()
        with pytest.raises(ValueError, match="Preconditions unmet"):
            sim.execute(skill="HumanPourBack")
        np.testing.assert_array_equal(before, sim.data.qpos)
        assert sim.elapsed_seconds == 0
        assert sim.human_interventions == 0
    finally:
        sim.close()


def test_adapter_has_native_physics_relations_and_no_free_reset():
    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.environment import PouringEnvironment
    from hitl_pmp.environments.pouring3d.objects import PouringObjects as O
    from hitl_pmp.environments.pouring3d.predicates import PouringPredicates

    env = PouringEnvironment(seed=0)
    env.hard_reset()
    try:
        assert isinstance(env.sim.env.unwrapped, PitcherPour3DEnv)
        initial = env.get_current_state()
        assert env.world is not None
        predicates = {p.name: p for p in PouringPredicates.all()}
        assert predicates["ReadyToPour"].holds(initial, ())
        assert all(predicates["InPitcher"].holds(initial, (c, O.pitcher)) for c in O.cubes)
        start_time = float(env.sim.data.time)
        env.take_action(action=env.noop_action())
        assert env.sim.data.time == pytest.approx(start_time + 1)
        assert env.last_info["cost_seconds"] == pytest.approx(1)
        assert {int(a[0]) for a in env.get_valid_actions()} == {0, 7}
        before = env.observe()[env.world].copy()
        with pytest.raises(NotImplementedError):
            env.set_state(state=initial)
        np.testing.assert_array_equal(before, env.observe()[env.world])
        with pytest.raises(ValueError):
            env.take_action(action=np.array([1.5, 0, 145, 0.055, 0, 10, 4]))
    finally:
        env.sim.close()
