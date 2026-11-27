"""The arm's unwind before PickPitcher keeps clear of the jug.

After a dumpling drop the hand rests beside the jug with joint 3 wound past half a turn.
PickPitcher first unwinds it by a whole turn; that path was checked against the worktops
only, so the finger swept through the jug and left it 40 mm off home, and the pick refused
"requires a ready scene" (seed 0 of the p = 0 arm, planned-carry launch, as recorded).
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_wound_arm_beside_jug.json"


def _restored() -> KitchenEnvironment:
    start = json.loads(FIXTURE.read_text())
    env = KitchenEnvironment(requested_count=start["requested"])
    env.reset_to_seed(seed=start["scene_seed"], requested=start["requested"])
    c = env.controller()
    c.data.qpos[:] = start["qpos"]
    c.data.qvel[:] = start["qvel"]
    c.data.time = start["time"]
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()
    return env


def test_the_unwinds_path_check_sees_the_jug():
    """The arm as recorded at the unwind's first finger contact, the jug still home: the
    unwind's own path check must reject it."""
    env = _restored()
    try:
        c = env.controller()
        contact = json.loads(FIXTURE.read_text())["first_jug_contact_qpos"]
        c.phase = "unwind continuous arm joints"
        c.scratch.qpos[:] = c.data.qpos
        c.scratch.qpos[c.arm_q] = np.asarray(contact)[c.arm_q]
        c.update_ik_scratch()
        assert np.max(c.obstacle_residual(), initial=0) > 0.005
    finally:
        env.close()
