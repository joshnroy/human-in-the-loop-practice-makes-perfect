"""Physical outcomes of native KinDER pouring and persistent resets."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KinDER not installed"
)


def test_physical_pour_and_both_resets():
    from hitl_pmp.environments.pouring3d.environment import PouringEnvironment
    from hitl_pmp.humans.oracle import UnconditionalHumanOracle

    env = PouringEnvironment(seed=0, scene_bg=False)
    env.hard_reset()
    sim = env.sim
    try:
        assert sim.locations() == ["pitcher"] * 5
        assert sim.execute(skill="PickPitcher")["success"]
        assert sim.controller.finger_contacts(name="pitcher", handle_only=True) == {"left", "right"}
        assert sim.locations() == ["pitcher"] * 5
        assert sim.execute(skill="Pour")["success"]
        assert sim.execute(skill="PlacePitcherAtHome")["success"]
        assert sim.locations() == ["tray"] * 5
        tray_before = sim.controller.body_pose(name="tray")[0]
        robot_before = sim.data.qpos[sim.controller.arm_q].copy()
        time_before, cost_before = float(sim.data.time), sim.elapsed_seconds
        UnconditionalHumanOracle.execute_movables_reset(env=env, destination="reset_cubes")
        human_reset = env.last_info
        assert human_reset["success"]
        assert human_reset["simulated_human_reset"]
        np.testing.assert_allclose(sim.controller.body_pose(name="tray")[0], tray_before, atol=1e-5)
        np.testing.assert_array_equal(sim.data.qpos[sim.controller.arm_q], robot_before)
        assert sim.data.time == time_before
        assert sim.elapsed_seconds == cost_before
        assert sim.ready()
        assert sim.execute(skill="PickPitcher")["success"]
        assert sim.execute(skill="Pour")["success"]
        assert sim.execute(skill="PlacePitcherAtHome")["success"]
        recovered = 0
        expected_human_interventions = 1
        for _ in range(5):
            cube = sim.controller.select_reset_cube()
            event = sim.execute(skill="PickCubeFromTray", cube=cube)
            if not event["success"]:
                # Contact-rich landings can leave an ungraspable cube. A failed
                # lift must remain a failure, not teleport the cube into recovery.
                assert event["error"] == "Physical cube grasp was not retained during lift"
                assert sim.held is None
                assert sim.locations()[cube] != "pitcher"
                assert not sim.ready()
                assert sim.human_interventions == 1
                before = sim.data.qpos.copy()
                time_before = float(sim.data.time)
                UnconditionalHumanOracle.execute_movables_reset(env=env, destination="reset_cubes")
                assert env.last_info["success"]
                np.testing.assert_array_equal(
                    sim.data.qpos[sim.controller.arm_q], before[sim.controller.arm_q]
                )
                assert sim.data.time == time_before
                expected_human_interventions += 1
                break
            assert sim.execute(skill="DropCubeIntoPitcher", cube=cube)["success"]
            assert sim.locations()[cube] == "pitcher"
            recovered += 1
        assert recovered >= 1, "The test must exercise physical pickup and drop-back"
        assert sim.ready(), sim.locations()
        assert sim.human_interventions == expected_human_interventions
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
    from hitl_pmp.environments.pouring3d.environment import PouringEnvironment

    env = PouringEnvironment(seed=0, scene_bg=False)
    env.hard_reset()
    sim = env.sim
    try:
        before = sim.data.qpos.copy()
        with pytest.raises(ValueError, match="Human reset requires"):
            env.reset_movables()
        np.testing.assert_array_equal(before, sim.data.qpos)
        assert sim.elapsed_seconds == 0
        assert sim.human_interventions == 0
        with pytest.raises(ValueError, match="Human reset uses HumanOracle"):
            env.take_action(action=np.array([5, 0, 145, 0, 0, 10, 4]))
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


def test_requested_count_is_observed_and_human_reset_preserves_request():
    from hitl_pmp.environments.pouring3d.environment import PouringEnvironment

    env = PouringEnvironment(seed=0, scene_bg=False, requested_count=3)
    env.hard_reset()
    try:
        assert env.world is not None
        index = env.world.type.feature_names.index("requested_count_0")
        assert env.get_current_state()[env.world][index] == 3
        scene = env.sim.controller.scene
        scene.get_object("cube_0").set_pose([0.65, 0.2, 0.02], [1, 0, 0, 0])
        import mujoco

        mujoco.mj_forward(env.sim.model, env.sim.data)
        env.reset_movables()
        assert env.get_current_state()[env.world][index] == 3
        assert env.sim.snapshot()["requested_count"] == 3
        assert not env.sim.snapshot()["goal"]
    finally:
        env.sim.close()


def test_physical_requested_two_pour_retains_three():
    from hitl_pmp.environments.pouring3d.simulation import PouringSimulation

    sim = PouringSimulation.create(seed=0, scene_bg=False, requested_count=2)
    try:
        assert sim.execute(skill="PickPitcher")["success"]
        event = sim.execute(skill="Pour", parameters={"angle": 130, "hold_seconds": 0.2})
        assert event["success"], event["after"]
        assert sim.locations().count("tray") == 2
        assert sim.locations().count("pitcher") == 3
        assert sim.settled()
        assert ("RequestedCountInTray", ()) in sim.abstract_atoms()
        assert ("RequestedRemainderInPitcher", ()) in sim.abstract_atoms()
        assert ("Goal", ()) in sim.abstract_atoms()
        assert sim.execute(skill="PlacePitcherAtHome")["success"]
        assert sim.snapshot()["goal"]
    finally:
        sim.close()
