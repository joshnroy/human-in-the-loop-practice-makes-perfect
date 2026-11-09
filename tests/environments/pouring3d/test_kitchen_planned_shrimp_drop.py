"""DropShrimpIntoPitcher plans its carry (PyBullet), as the jug's carries already do.

In the overnight relaunch pouring-fixed-20261108-76f41dc9c, 14 of 14 failed drops
failed at the scripted "raise held shrimp in aisle": it held the grasp orientation
fixed and tracked the raise with local IK, which pinned the wrist (joint 6) at its
119.7 deg limit about 1 cm short. That carry dates from when a dumpling was held by
friction; since the weld (#703) the hand may take any orientation while carrying one.
The starts below are three of those 14, as recorded (base x, y, yaw; arm degrees).
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_planned_shrimp_drop")

DROP = np.array([SKILL_NAMES.index("DropShrimpIntoPitcher"), 0, 1, 0, 0, 0], dtype=float)

FAILED_STARTS = {
    "tray side, wrist at limit": (
        [0.2, -1.052, 0.0],
        [-251.6, -9.6, 482.1, -69.5, 137.7, 119.7, -1.7],
    ),
    "tray side, wrist free": (
        [-0.349, -1.05, 0.0],
        [-27.5, 87.6, 339.2, -126.3, 10.5, 88.2, -56.6],
    ),
    "skillet side, wrist at limit": (
        [0.3, 0.916, 0.0],
        [57.3, 25.4, 277.1, 42.0, -3.0, 119.7, -23.9],
    ),
}


def _holding_at(*, base: list[float], arm_degrees: list[float]) -> KitchenEnvironment:
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    c.data.qpos[c.base_q] = base
    c.data.qpos[c.arm_q] = np.deg2rad(arm_degrees)
    c.data.qvel[:] = 0.0
    mujoco.mj_forward(c.model, c.data)
    c.base_target = c.data.qpos[c.base_q].copy()
    c.last_q = c.data.qpos[c.arm_q].copy()
    name = c.scene.cube_names[0]
    hand, _ = c.hand_pose()
    c.scene.get_object(name).set_pose(hand, [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(c.model, c.data)
    c.scene.attach_payload(name=name)
    c.held, c.grip = name, 1.0
    return env


@pytest.mark.parametrize("start", list(FAILED_STARTS))
def test_a_drop_from_a_recorded_failed_start_lands_in_the_jug(*, start):
    base, arm = FAILED_STARTS[start]
    env = _holding_at(base=base, arm_degrees=arm)
    try:
        env.take_action(action=DROP)
        c = env.controller()
        assert env.last_info["success"], env.last_info
        assert c.held is None
        assert c.scene.cube_in_vessel(c.scene.cube_names[0], "pitcher")
    finally:
        env.close()


def test_plans_stay_inside_the_executed_arms_joint_limits():
    """PyBullet's arm model is wider than MuJoCo's (joint 6: 127.8 deg against 119.7),
    so an unchecked plan can end where the executed arm cannot follow ("Arm did not
    track the planned path", from the 'tray side, wrist free' start above)."""
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        sim = KitchenPyBulletSim.from_controller(controller=c)
        try:
            assert sim.executed_arm_limits is not None
            lower, upper = sim.executed_arm_limits
            np.testing.assert_allclose(lower, c.lower)
            np.testing.assert_allclose(upper, c.upper)
        finally:
            sim.close()
    finally:
        env.close()
