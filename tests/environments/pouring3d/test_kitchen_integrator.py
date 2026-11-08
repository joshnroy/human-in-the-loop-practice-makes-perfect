"""The kitchen steps with MuJoCo's `implicit` integrator, not KinDER's `implicitfast`.

`implicitfast` drops the gyroscopic term from its implicit velocity derivative, so a
spinning dumpling (anisotropic principal inertia, ~3.9/3.5/1.3e-6 kg m^2) gains rotational
energy every step with no torque on it. Landing on the skillet or the floor spins one past
~100 rad/s, the energy runs away, MuJoCo's guard fires and `mj_resetData` wipes the scene
(`test_kitchen_simulator_diverged.py`). `implicit` keeps the full derivative.
"""

import importlib.util

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
    pytest.mark.xdist_group("kitchen_integrator"),
]


@pytest.fixture(scope="module")
def kitchen_model() -> mujoco.MjModel:
    env = KitchenEnvironment(requested_count=3)
    env.reset_to_seed(seed=0, requested=3)
    return env.controller().model


def test_the_compiled_kitchen_model_uses_the_implicit_integrator(*, kitchen_model):
    assert kitchen_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT


def _torque_free_energy_ratio(*, kitchen: mujoco.MjModel, spin: float, seconds: float) -> float:
    """KE_rot(T)/KE_rot(0) of a lone dumpling spinning freely under the kitchen's options."""
    body = kitchen.body("potsticker_1").id
    inertia = kitchen.body_inertia[body]
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><option gravity="0 0 0"/><worldbody><body><freejoint/>
        <inertial pos="0 0 0" mass="{kitchen.body_mass[body]}"
            diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>
        <geom type="sphere" size="0.01" contype="0" conaffinity="0" mass="0"/>
        </body></worldbody></mujoco>"""
    )
    model.opt.timestep = kitchen.opt.timestep
    model.opt.integrator = kitchen.opt.integrator
    data = mujoco.MjData(model)
    axis = np.random.default_rng(0).normal(size=3)
    data.qvel[3:6] = spin * axis / np.linalg.norm(axis)

    def energy() -> float:
        return 0.5 * float(data.qvel[3:6] ** 2 @ inertia)

    start = energy()
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return energy() / start


def test_a_freely_spinning_dumpling_gains_no_energy_under_the_kitchen_integrator(*, kitchen_model):
    # Exact physics conserves it; implicitfast reaches 2.23x in this second, implicit 0.83x.
    ratio = _torque_free_energy_ratio(kitchen=kitchen_model, spin=100.0, seconds=1.0)
    assert ratio <= 1.0
