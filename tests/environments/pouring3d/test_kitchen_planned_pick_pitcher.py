"""PickPitcher's free-space motions are planned in PyBullet and executed clear in MuJoCo."""

import mujoco
import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_planned_pick_pitcher")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)


def test_a_refused_plan_moves_nothing(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills

    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    steps = c.steps

    def refuse(**_):
        raise parameterized_skills.PlanningRefused("No collision-free arm path to pitcher hover")

    monkeypatch.setattr(parameterized_skills.KitchenMotionPlanner, "plan_arm", refuse)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is False
    assert env.last_info["error"] == "No collision-free arm path to pitcher hover"
    assert c.steps == steps
    assert c.held is None


def test_the_planned_pick_lifts_the_jug_without_touching_anything():
    env = KitchenEnvironment(requested_count=1)
    seen = Clearance.record(env=env)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    phases = {row[0] for row in seen}
    assert {"planned base to pitcher", "planned arm to pitcher hover"} <= phases
    Clearance.check(seen=seen)


def test_the_planned_pick_starts_from_a_hand_over_the_jug():
    """After a dumpling drop the hand hangs over the jug's mouth, the case the scripted
    pick needed a hand-tuned aisle detour for."""
    from scipy.spatial.transform import Rotation

    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    # The base as a drop leaves it, nearer the counter, and the hand over the mouth,
    # posed with the scripted controller's own (MuJoCo) IK.
    base = np.array([0.2, c.home[1], 0.0])
    c.data.qpos[c.base_q] = base
    c.base_target = base.copy()
    mujoco.mj_forward(c.model, c.data)
    # The dumpling pick's 30 degree tilted grasp, as DropShrimpIntoPitcher holds it.
    from kinder_models.dynamic3d.utils import inverse_kinematics_fixed_effort
    from pybullet_helpers.geometry import Pose

    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

    tilted = Rotation.from_euler("x", 30, degrees=True) * Rotation.from_euler("y", np.pi)
    sim = KitchenPyBulletSim.from_controller(controller=c)
    above = tuple(float(v) for v in c.home + [0, 0, 0.24])
    q = inverse_kinematics_fixed_effort(
        sim.robot, Pose(above, tuple(tilted.as_quat().tolist())), set_joints=False
    )[:7]
    sim.close()
    c.data.qpos[c.arm_q] = q
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.pause(phase="settle hand over jug", seconds=1, grip=0)
    hand, _ = c.hand_pose()
    assert np.linalg.norm(hand[:2] - c.home[:2]) < 0.03
    seen = Clearance.record(env=env)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    Clearance.check(seen=seen)
