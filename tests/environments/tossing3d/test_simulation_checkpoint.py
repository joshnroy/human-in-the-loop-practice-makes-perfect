"""MuJoCo engine checkpoints preserve exact contact-physics continuation.

Keep this test in the physics suite: importing MuJoCo loads the configured GL backend,
and CI installs OSMesa only for physics and script jobs.
"""

import numpy as np
import pytest


def test_engine_checkpoint_restores_float64_fingers_controls_and_solver_state():
    """Tiny contact model: no KINDER, assets, renderer, planner or heavy simulation."""
    import mujoco

    from hitl_pmp.environments.tossing3d.simulation_checkpoint import capture_engine, restore_engine

    xml = """<mujoco><option timestep="0.002"/><worldbody>
    <geom type="plane" size="1 1 .1"/><body pos="0 0 .12">
    <joint name="finger" type="slide" axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="finger"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0123456789012345
    data.qvel[0] = -0.123456789012345
    data.ctrl[0] = 0.333333333333333
    data.qfrc_applied[0] = 0.123456789012345
    for _ in range(25):
        mujoco.mj_step(model, data)
    checkpoint = capture_engine(model=model, data=data)
    expected = []
    for _ in range(20):
        mujoco.mj_step(model, data)
        expected.append(capture_engine(model=model, data=data)["integration"])
    fresh_model = mujoco.MjModel.from_xml_string(xml)
    fresh_data = mujoco.MjData(fresh_model)
    restore_engine(model=fresh_model, data=fresh_data, checkpoint=checkpoint)
    np.testing.assert_array_equal(
        capture_engine(model=fresh_model, data=fresh_data)["integration"], checkpoint["integration"]
    )
    for state in expected:
        mujoco.mj_step(fresh_model, fresh_data)
        np.testing.assert_array_equal(
            capture_engine(model=fresh_model, data=fresh_data)["integration"], state
        )
    fresh_model.body_mass[1] += 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        restore_engine(model=fresh_model, data=fresh_data, checkpoint=checkpoint)
    incompatible = {**checkpoint, "mujoco_version": "incompatible"}
    with pytest.raises(ValueError, match="version"):
        restore_engine(model=model, data=data, checkpoint=incompatible)
