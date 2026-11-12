"""A held dumpling's carry is planned at a height the arm reaches with the hand down.

DropShrimpIntoPitcher reused the jug's carry pose, the hand 1.10 m up and 0.45 m in front
of the base. Holding the jug by its handle the hand is level there; holding a dumpling it
points down, and a downward hand reaches about 0.97 m at that distance. IKFast offered no
solution at all, for every grasp turn: 299 of the 302 refused drops in the planned-carry
launch failed this way. The starts below are three of them, as recorded.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_refused_shrimp_carries.json"
STARTS = json.loads(FIXTURE.read_text())


def _holding(*, start: dict) -> tuple[KitchenEnvironment, str]:
    env = KitchenEnvironment(requested_count=start["requested"])
    env.reset_to_seed(seed=start["scene_seed"], requested=start["requested"])
    c = env.controller()
    c.data.qpos[:] = start["qpos"]
    c.data.qvel[:] = start["qvel"]
    c.data.time = start["time"]
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()
    name = c.scene.cube_names[int(start["action"][1])]
    c.scene.attach_payload(name=name)
    c.held, c.grip = name, 1.0
    env.current_state = env.observe()
    return env, name


@pytest.mark.parametrize("label", list(STARTS))
def test_a_recorded_refused_carry_now_drops_the_dumpling_in_the_jug(*, label):
    start = STARTS[label]
    env, name = _holding(start=start)
    try:
        env.take_action(action=np.array(start["action"], dtype=float))
        c = env.controller()
        assert env.last_info["success"], env.last_info
        assert c.held is None
        assert c.scene.cube_in_vessel(name, "pitcher")
    finally:
        env.close()
