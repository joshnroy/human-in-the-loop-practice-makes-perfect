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


class Family:
    """Which IK branch an arm configuration is in, measured against a fresh pick's."""

    # A repeat of a fresh pick lands within a few degrees of it on every joint; the
    # mirror branch differs by about 175 degrees on the shoulder (joints 1 and 2).
    SAME_BRANCH_DEG = 30.0

    @staticmethod
    def distance_deg(*, a: np.ndarray, b: np.ndarray) -> float:
        """Largest per-joint difference, continuous joints (1, 3, 5, 7) the short way."""
        delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        delta[[0, 2, 4, 6]] = np.arctan2(np.sin(delta[[0, 2, 4, 6]]), np.cos(delta[[0, 2, 4, 6]]))
        return float(np.rad2deg(np.max(np.abs(delta))))


# ees seed 2 (Della 16761243), as the third DropShrimpIntoPitcher left the arm: the
# hand over the jug's mouth, joint 3 wound most of a turn (517 deg), the base 0.2 m
# nearer the counter. PickPitcher from here took the mirror branch, from which no
# upright carry path exists, and every PlacePitcherAtHome after it was refused.
AFTER_DUMPLING_DROPS_ARM = np.array([1.8696, -0.5723, 9.0205, -1.4653, 2.2514, 2.09, -0.3626])
PLACE = np.array([2, 0, 1, 0, 0, 0], dtype=float)


def test_a_pick_after_dumpling_drops_lands_in_the_fresh_picks_branch_and_places():
    fresh = KitchenEnvironment(requested_count=1)
    fresh.take_action(action=PICK_PITCHER)
    assert fresh.last_info["success"] is True, fresh.last_info
    nominal = fresh.controller().data.qpos[fresh.controller().arm_q].copy()

    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    base = np.array([0.2, c.home[1], 0.0])
    c.data.qpos[c.base_q] = base
    c.base_target = base.copy()
    c.data.qpos[c.arm_q] = AFTER_DUMPLING_DROPS_ARM
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.pause(phase="settle hand over jug", seconds=1, grip=0)
    seen = Clearance.record(env=env)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    Clearance.check(seen=seen)
    arm = c.data.qpos[c.arm_q].copy()
    distance = Family.distance_deg(a=arm, b=nominal)
    print(f"post-pick arm {np.round(np.rad2deg(arm))} deg, {distance:.1f} deg from fresh")
    assert distance < Family.SAME_BRANCH_DEG

    env.take_action(action=PLACE)
    assert env.last_info["success"] is True, env.last_info
