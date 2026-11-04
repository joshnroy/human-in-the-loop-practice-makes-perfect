"""Food skills that start already holding food, and the planned food motions.

`held_food_pad_on_plate_rim.json` is the Della dry run's state (5cce1de, EES seed 0)
after PickFood(cube_3) from the plate failed its lift: cube_3 still held, the left pad
10 um into the plate's rim. Every following PlaceFoodOnTray and ReturnFoodToRack then
executed nothing ("Arm transit start configuration blocked"), 0/4 each.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
HELD_ON_RIM = FIXTURES / "held_food_pad_on_plate_rim.json"


def restored(*, fixture: Path):
    """The environment at a recorded motion state, the grasp re-measured."""
    import mujoco

    from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    plain = json.loads(fixture.read_text())
    config = RackServingConfig.model_validate({
        k: v for k, v in plain.get("config", {}).items() if k in RackServingConfig.model_fields
    })
    env = RackServingEnvironment(config=config)
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
    held = [name for name in backend.cube_names if backend.held(name=name)]
    if held:
        controllers.held_food = shelf.held_cube = held[0]
    return env


@pytest.mark.parametrize("skill", [4, 5], ids=["PlaceFoodOnTray", "ReturnFoodToRack"])
def test_a_held_food_resting_a_pad_on_the_rim_still_executes(*, skill):
    """A pad already touching a fixture when the skill starts may leave that contact; it
    must not block the skill's very first configuration."""
    env = restored(fixture=HELD_ON_RIM)
    try:
        assert env.backend().held(name="cube_3")
        env.take_action(action=np.array([float(skill), 3, 0, 0, 0]))
        error = env.last_skill_error() or ""
        assert "start configuration blocked" not in error, error
        assert env._last_controller_steps > 0, error
    finally:
        env.close()
