"""PickPitcher's free-space motions are planned in PyBullet and executed clear in MuJoCo."""

import mujoco
import numpy as np
import pytest
from kitchen_clearance import Mujoco

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_planned_pick_pitcher")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
OBSTACLES = (
    "prep_table",
    "stovetop",
    "pitcher",
    "tray",
    "potsticker_0",
    "potsticker_1",
    "potsticker_2",
)


class Clearance:
    """Every control tick of a planned phase: the arm's MuJoCo distance to the scene."""

    @staticmethod
    def record(*, env: KitchenEnvironment) -> list[tuple[str, float]]:
        c = env.controller()
        m = c.model
        arm = Mujoco.arm_geoms(model=m)
        obstacles = Mujoco.geoms(model=m, names=OBSTACLES)
        seen: list[tuple[str, float]] = []
        sink = c.callback

        def tick(**kwargs):  # noqa: PLR0917 -- controller callback signature
            if sink is not None:
                sink(**kwargs)
            if c.phase.startswith("planned "):
                distance = Mujoco.distance(model=m, data=c.data, a=arm, b=obstacles, cap=0.05)
                seen.append((c.phase, distance))

        c.callback = tick
        return seen


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
    phases = {phase for phase, _ in seen}
    assert {"planned base to pitcher", "planned arm to pitcher hover"} <= phases
    assert min(d for _, d in seen) > 0, min(seen, key=lambda r: r[1])


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
    assert min(d for _, d in seen) > 0, min(seen, key=lambda r: r[1])
