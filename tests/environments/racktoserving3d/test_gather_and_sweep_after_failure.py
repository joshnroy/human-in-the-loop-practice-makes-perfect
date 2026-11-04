"""GatherAndSweep from the state a failed GatherAndSweep left behind.

The fixture is the measured state at the end of the first GatherAndSweep of Della
dry run 16788008 (determinized Curtis, seed 0, a66dc32): the gather_minus_y push
failed with the scraper held at the -y side stance and its blade lying flat on the
rack top. Every later GatherAndSweep in that run stopped within 15 control steps,
then at 0, at the squeeze_plus_y raise.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "gather_minus_y_failed_on_rack.json"


def _restored_env(*, skill_step_limit):
    import mujoco

    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    plain = json.loads(FIXTURE.read_text())
    env = RackServingEnvironment(skill_step_limit=skill_step_limit)
    RackServingTasks(env=env, seed=0).sample_train_task()
    backend = env.backend()
    model, data = backend.model(), backend.data()
    for joint, value in plain["robot_joints"].items():
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
    for name, pose in plain["free_bodies"].items():
        address = model.jnt_qposadr[model.joint(f"{name}_joint").id]
        data.qpos[address : address + 7] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    controllers = env.rack_controllers()
    shelf = controllers.shelf
    shelf.grip = backend.grip = float(plain["grip"])
    shelf.base_target = data.qpos[shelf.ba].copy()
    for _ in range(20):
        shelf.step(arm=data.qpos[shelf.qa].copy(), label="restore_hold")
    shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ controllers.tool.wiper_pose()
    return env


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")
def test_gather_and_sweep_lifts_a_blade_left_lying_on_the_rack_off_it():
    env = _restored_env(skill_step_limit=150)
    try:
        controllers = env.rack_controllers()
        rack_top = env.backend().support(name="rack_0").top_z
        assert controllers.tool.pads_hold_grip()
        assert controllers.tool.blade_edge()[:, 2].min() == pytest.approx(rack_top, abs=0.002)
        before = env.backend().control_steps()
        env.take_action(action=np.array([1.0, 0.0, 0.10, -0.01, 0.06]))
        error = env._last_skill_error or ""
        assert "_raise" not in error, error
        assert env.backend().control_steps() - before > 20, error
        assert controllers.tool.blade_edge()[:, 2].min() > rack_top + 0.005
    finally:
        env.close()
