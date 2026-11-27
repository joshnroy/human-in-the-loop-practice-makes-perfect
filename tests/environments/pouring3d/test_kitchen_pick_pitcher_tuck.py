"""PickPitcher tucks an outstretched empty arm before it drives the base to the jug.

A failed dumpling skill can leave the arm reaching out over the stovetop; the human
pour-back does not move it, and a base drive planned with the arm as it is finds no
collision-free path. PickPitcher's own policy first plans and executes an arm-only move
to the episode-initial (tucked) arm, then drives.
"""

import mujoco
import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_pick_pitcher_tuck")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)

# The 4-cycle check on pouring-integration-20261104b (motion.npz frame 7531): cycle 2's
# second DropShrimpIntoPitcher lost its grasp 1.6 s in, the arm out over the stovetop
# ([126, -68, 138, -59, -220, 100, 234] deg). After the human pour-back, PickPitcher
# refused: "No collision-free base path to the pitcher pick pose".
STRANDED_BASE = np.array([0.3, 0.7871, 0.0])
STRANDED_ARM = np.array([2.2012, -1.187, 2.4001, -1.0261, -3.8371, 1.7362, 4.0888])


def _stranded() -> KitchenEnvironment:
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    c.data.qpos[c.base_q] = STRANDED_BASE
    c.base_target = STRANDED_BASE.copy()
    c.data.qpos[c.arm_q] = STRANDED_ARM
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.pause(phase="settle stranded arm", seconds=1, grip=0)
    return env


def _plan(*, env: KitchenEnvironment, planner_type: type) -> dict:
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

    c = env.controller()
    handle, upright, _ = c.pitcher_grasp_pose()
    sim = KitchenPyBulletSim.from_controller(controller=c)
    try:
        planner = planner_type(sim=sim)
        planner.reset(
            x=c.scene._get_object_centric_state(),
            arm=c.data.qpos[c.arm_q].copy(),
            base=c.data.qpos[c.base_q].copy(),
            base_target=c.base_home.copy(),
            hover=handle + [-c.pitcher_hover_backoff, 0, c.pitcher_approach_clearance],
            rotation=upright,
            home=c.arm_neutral_q.copy(),
        )
        return planner.plans
    finally:
        sim.close()


def test_a_pick_from_the_stranded_reach_tucks_first_and_succeeds():
    # This recorded arm is 14.4 mm from the scene in PyBullet, inside the 15 mm planning
    # margin: the tuck escapes the margin first (`test_kitchen_plan_from_inside_margin`).
    env = _stranded()
    seen = Clearance.record(env=env)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    phases = [row[0] for row in seen]
    tuck = "planned tuck arm before pitcher pick"
    assert tuck in phases
    assert phases.index(tuck) < phases.index("planned base to pitcher")
    Clearance.check(seen=seen)


def test_a_tucked_start_plans_exactly_as_before():
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
        PickPitcherController,
    )

    class Untucked(PickPitcherController):
        """PickPitcher's plan before the tuck: base and hover from the arm as it is."""

        def reset(self, *, x, arm, base, base_target, hover, rotation, home):  # noqa: PLR0917
            sim = self.sim
            sim.set_state(x)
            base_path = KitchenMotionPlanner.plan_base(
                sim=sim, state=x, target=base_target, arm=arm, what="the pitcher pick pose"
            )
            from spatialmath import SE2

            sim.set_robot_base(SE2(*(float(v) for v in base_target)))
            self.plans = {
                self.BASE_MOTION: base_path,
                self.MOVE_ARM_TO_HOVER: KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=arm,
                    position=hover,
                    rotation=rotation,
                    what="pitcher hover",
                    reference=home,
                ),
            }

    # The arm tucked as at the episode start, the base away from the pick pose.
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    # As a dumpling drop leaves the base (`test_kitchen_planned_pick_pitcher.py`).
    start = np.array([0.2, c.home[1], 0.0])
    c.data.qpos[c.base_q] = start
    c.base_target = start.copy()
    mujoco.mj_forward(c.model, c.data)
    now = _plan(env=env, planner_type=PickPitcherController)
    before = _plan(env=env, planner_type=Untucked)
    assert list(now) == list(before)
    for key in before:
        assert len(now[key]) == len(before[key])
        for a, b in zip(now[key], before[key], strict=True):
            np.testing.assert_array_equal(a, b)


def test_an_infeasible_tuck_refuses_the_whole_pick_and_moves_nothing(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills

    original = parameterized_skills.KitchenMotionPlanner.plan_arm

    def refuse_the_tuck(**kwargs):
        if kwargs["what"] == "the tucked arm":
            raise parameterized_skills.PlanningRefused(
                "No collision-free arm path to the tucked arm"
            )
        return original(**kwargs)

    monkeypatch.setattr(
        parameterized_skills.KitchenMotionPlanner, "plan_arm", staticmethod(refuse_the_tuck)
    )
    env = _stranded()
    c = env.controller()
    # The stranded joint 5 (-220 deg) is past half a turn from neutral, so the pick's
    # own unwind runs first (`test_kitchen_joint_windup`); the refused plan moves nothing.
    c.unwind_continuous_joints(threshold=np.pi)
    steps = c.steps
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is False
    assert env.last_info["error"] == "No collision-free arm path to the tucked arm"
    assert c.steps == steps
