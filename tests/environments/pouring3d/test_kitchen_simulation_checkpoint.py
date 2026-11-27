"""A kitchen practice scene captured between skills carries on identically in a
freshly built scene: the jug welded in the hand, dumplings in the pan, the human's
draws and every piece of controller bookkeeping included."""

import importlib.util
import pickle

import numpy as np
import pytest

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
    pytest.mark.xdist_group("kitchen_simulation_checkpoint"),
]

PICK_PITCHER = np.array([0, 0, 2, 0, 0, 0], dtype=float)
POUR_TWO = np.array([1, 0, 2, 1.0, 0, 0], dtype=float)
PLACE_PITCHER = np.array([2, 0, 2, 0, 0, 0], dtype=float)


def _environment():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    # Half the human's attempts fail, so a wrong generator state changes the scene.
    return KitchenEnvironment(requested_count=2, human_reset_success_probability=0.5)


def _integration(*, env):
    import mujoco

    c = env.controller()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(c.model, spec))
    mujoco.mj_getState(c.model, c.data, state, spec)
    return state


def _plain(*, value):
    if isinstance(value, dict):
        return {key: _plain(value=item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(value=item) for item in value]
    if isinstance(value, np.ndarray | np.generic):
        return np.asarray(value).tolist()
    return value


def _record(*, env):
    """Everything a method, the step accounting or the next skill reads."""
    state = env.get_current_state()
    c = env.controller()
    return {
        "integration": _integration(env=env).tolist(),
        "observation": {obj.name: values.tolist() for obj, values in state.data.items()},
        "atoms": sorted(state.abstract_atoms),
        "info": _plain(value=env.last_info),
        "weld_targets": c.model.eq_data.tolist(),
        "held": c.held,
        "controller_steps": c.steps,
        "human_interventions": env.human_interventions,
        "cost_observation": _plain(value=env.cost_observation()),
        "scene_steps": c.scene.scene_steps,
        "robot_timestep": c.scene._robot_env.timestep,
        "rest_window": [[time, at.tolist()] for time, at in c.scene.payload_rest.samples],
        "scene_rng": _plain(value=c.scene.np_random.bit_generator.state),
    }


def _continue(*, env):
    """A place, then three human attempts (seeded draws at p = 0.5), then a pick."""
    from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET

    records = []
    env.take_action(action=PLACE_PITCHER)
    records.append(_record(env=env))
    for _ in range(3):
        env.reset_movables(destination=HUMAN_RESET)
        records.append(_record(env=env))
    env.take_action(action=PICK_PITCHER)
    records.append(_record(env=env))
    return records


def test_a_held_jug_with_dumplings_in_the_pan_resumes_identically_in_a_fresh_scene():
    original = _environment()
    restored = _environment()
    try:
        original.reset_to_seed(seed=0, requested=2)
        original.take_action(action=PICK_PITCHER)
        assert original.last_info["success"], original.last_info
        original.take_action(action=POUR_TWO)
        assert original.last_info["error"] is None, original.last_info
        c = original.controller()
        assert c.held == "pitcher"
        assert c.scene.pitcher_attached()
        assert "tray" in c.scene.cube_locations()
        # Stands for an earlier human pour-back, which jitters the respawned dumplings
        # from this generator: a scene rebuilt from the seed alone has not drawn it.
        c.scene.np_random.random()

        # Through bytes, as a new process receives it.
        checkpoint = pickle.loads(pickle.dumps(original.simulation_checkpoint()))
        at_checkpoint = _record(env=original)
        expected = _continue(env=original)

        restored.restore_simulation_checkpoint(checkpoint=checkpoint)
        r = restored.controller()
        assert r is not c
        assert r.scene.pitcher_attached()
        robot = r.scene._robot_env
        assert np.shares_memory(robot.qpos["arm"], r.data.qpos)
        assert np.shares_memory(robot.ctrl["arm"], r.data.ctrl)
        assert _record(env=restored) == at_checkpoint
        assert _continue(env=restored) == expected
        # The continuation moved the scene and drew a human failure and a success.
        assert expected[-1]["integration"] != at_checkpoint["integration"]
        assert {record["info"].get("human_reset_failed", False) for record in expected[1:4]} == {
            True,
            False,
        }
    finally:
        original.close()
        restored.close()


def test_a_checkpoint_from_a_different_scene_is_refused():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    original = _environment()
    slippery = KitchenEnvironment(requested_count=2, skillet_mu=0.2)
    try:
        original.reset_to_seed(seed=0, requested=2)
        checkpoint = original.simulation_checkpoint()
        with pytest.raises(ValueError, match="fingerprint"):
            slippery.restore_simulation_checkpoint(checkpoint=checkpoint)
    finally:
        original.close()
        slippery.close()
