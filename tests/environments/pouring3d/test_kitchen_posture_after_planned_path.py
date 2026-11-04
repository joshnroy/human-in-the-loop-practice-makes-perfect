"""A planned arm path re-seats joint 5; the posture attractor must follow it there.

The posture turn is chosen at a skill's start, nearest joint 5. A planned path in the
same skill can then carry joint 5 across half a turn (PickPitcher's tuck took it from
-129 deg to +19 deg in the 20261104c 4-cycle check, with the posture at 1.0 rad). The
scripted IK motions after it (the descent onto the handle) must behave exactly as they
would in a skill that had started where the plan left the arm, not be dragged back
toward the turn chosen before the path.
"""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

pytestmark = pytest.mark.xdist_group("kitchen_posture_after_planned_path")

POSTURE_JOINT = 4


def _descend_after_a_planned_path(*, restart_skill: bool) -> list[np.ndarray]:
    c: KitchenController = KitchenEnvironment(requested_count=1).controller()
    # As a dumpling drop leaves it after a pour: posture at pour_posture, joint 5 at
    # -129 deg, so the skill-start turn is -2*pi.
    c.posture = c.posture_goal = 1.0
    c.data.qpos[c.arm_q[POSTURE_JOINT]] = np.deg2rad(-129.0)
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.choose_posture_turn()
    assert c.posture_turn == pytest.approx(-2 * np.pi)
    start = c.data.qpos[c.arm_q].copy()
    goal = start.copy()
    goal[POSTURE_JOINT] = np.deg2rad(19.0)
    c.follow_arm_path(path=[start, goal], phase="planned test path", grip=0)
    assert float(c.data.qpos[c.arm_q[POSTURE_JOINT]]) == pytest.approx(goal[4], abs=0.02)
    if restart_skill:
        # The reference: a skill that starts where the plan left the arm.
        c.choose_posture_turn()
    hand, rotation = c.hand_pose()
    solved = []
    for step in range(15):
        try:
            solved.append(
                c.inverse_kinematics(
                    position=hand - np.array([0.0, 0.0, 0.001 * step]), rotation=rotation
                ).copy()
            )
        except ValueError as exc:
            pytest.fail(f"descent IK step {step} failed after the planned path: {exc}")
    return solved


def test_ik_after_a_planned_path_solves_as_a_skill_starting_there_would():
    within = _descend_after_a_planned_path(restart_skill=False)
    fresh = _descend_after_a_planned_path(restart_skill=True)
    for step, (a, b) in enumerate(zip(within, fresh, strict=True)):
        np.testing.assert_allclose(
            a, b, atol=1e-6, err_msg=f"descent step {step}: {np.rad2deg(a).round(1).tolist()}"
        )


def test_ik_driven_drift_alone_never_rechooses_the_turn():
    # Unwound at the skill start with the posture at 1.0, then carried by IK alone past
    # posture - pi: the turn stays the one chosen at the start, and joint 5 is pulled
    # back up toward +1.0 rather than on toward 1.0 - 2*pi.
    c: KitchenController = KitchenEnvironment(requested_count=1).controller()
    c.posture = c.posture_goal = 1.0
    c.choose_posture_turn()
    assert c.posture_turn == 0.0
    c.data.qpos[c.arm_q[POSTURE_JOINT]] = 1.0 - np.pi - 0.3
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    before = float(c.last_q[POSTURE_JOINT])
    hand, rotation = c.hand_pose()
    for _ in range(10):
        q = c.inverse_kinematics(position=hand, rotation=rotation)
        assert c.posture_turn == 0.0
    assert float(q[POSTURE_JOINT]) > before
