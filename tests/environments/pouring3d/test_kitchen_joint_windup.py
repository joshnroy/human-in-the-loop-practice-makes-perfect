"""Planned arm motions keep continuous joints inside MuJoCo's range, nearest neutral.

Della sweep, EES p=0.5 seed 0: PickPitcher wound joint 3 to 687.3 deg, 0.7 deg short of
MuJoCo's continuous-joint range (+-12 rad), because the planner's goal and path went the
short way from wherever the joint already was and the unwind before the pick only
acted after a full turn. The carry's tuck then planned past the range, the held-IK
guard tripped mid-tilt, and the jug came to rest on the pan.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
    CONTINUOUS,
    CONTINUOUS_LIMIT_RAD,
    KitchenMotionPlanner,
    PickPitcherController,
    PlanningRefused,
)

pytestmark = pytest.mark.xdist_group("kitchen_joint_windup")

TURN = 2 * np.pi
PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
RECORDED = Path(__file__).parent / "fixtures" / "kitchen_pick_pitcher_joint3_wound_ees_p05.json"


class Stop(Exception):
    """Raised where the pick would read its grasp pose, right after unwinding."""


def _wound(*, joint: int, turns: float | None = None, degrees: float | None = None):
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    if degrees is not None:
        c.data.qpos[c.arm_q[joint]] = np.deg2rad(degrees)
    else:
        c.data.qpos[c.arm_q[joint]] += turns * TURN
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.pause(phase="settle wound arm", seconds=1, grip=0)
    return env


def _plan_hover(*, env: KitchenEnvironment) -> list[np.ndarray]:
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

    c = env.controller()
    handle, upright, _ = c.pitcher_grasp_pose()
    sim = KitchenPyBulletSim.from_controller(controller=c)
    try:
        planner = PickPitcherController(sim=sim)
        planner.reset(
            x=c.scene._get_object_centric_state(),
            arm=c.data.qpos[c.arm_q].copy(),
            base=c.data.qpos[c.base_q].copy(),
            base_target=c.base_home.copy(),
            hover=handle + [-c.pitcher_hover_backoff, 0, c.pitcher_approach_clearance],
            rotation=upright,
            home=c.arm_neutral_q.copy(),
        )
        return planner.plans[planner.MOVE_ARM_TO_HOVER]
    finally:
        sim.close()


def _recorded() -> KitchenEnvironment:
    state = json.loads(RECORDED.read_text())
    env = KitchenEnvironment(requested_count=int(state["request"]))
    c = env.controller()
    c.data.qpos[:] = state["qpos"]
    c.data.qvel[:] = state["qvel"]
    c.data.time = state["time"]
    mujoco.mj_forward(c.model, c.data)
    c.held, c.grip = None, 0
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()
    return env


def test_the_recorded_pick_keeps_joint_3_within_half_a_turn_of_neutral(*, monkeypatch):
    env = _recorded()
    c = env.controller()
    assert np.rad2deg(c.data.qpos[c.arm_q[2]] - c.arm_neutral_q[2]) > 180
    reach: list[float] = []
    original = KitchenController.motor_step

    def record(self, **kwargs):  # noqa: PLR0917 -- patched bound method
        original(self, **kwargs)
        reach.append(float(self.data.qpos[self.arm_q[2]] - self.arm_neutral_q[2]))

    monkeypatch.setattr(KitchenController, "motor_step", record)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    # Before the fix the pick ended 507 deg past neutral (687 deg), 0.7 deg inside the
    # range; within half a turn of neutral at the start, it ends at most a turn past.
    assert max(np.abs(reach)) < 2 * np.pi, np.rad2deg(max(np.abs(reach)))
    assert abs(reach[-1]) <= np.pi + 0.05, np.rad2deg(reach[-1])


def test_the_planners_continuous_joints_are_mujocos():
    c = KitchenEnvironment(requested_count=1).controller()
    assert np.flatnonzero(c.lower == -CONTINUOUS_LIMIT_RAD).tolist() == list(CONTINUOUS)
    assert np.all(c.upper[list(CONTINUOUS)] == CONTINUOUS_LIMIT_RAD)


def test_a_fresh_hover_plan_is_already_nearest_neutral():
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    path = _plan_hover(env=env)
    np.testing.assert_array_equal(path[0], c.data.qpos[c.arm_q])
    offset = path[-1] - c.arm_neutral_q
    assert np.all(np.abs(offset[list(CONTINUOUS)]) <= np.pi)


def test_a_path_unwraps_from_the_start_as_the_follower_drives_it():
    start = np.array([0.0, 0.0, 11.8, 0.0, 0.0, 0.0, 0.0])
    path = [start.copy(), start.copy()]
    path[1][2] = -0.3  # 11.8 + (-0.3 - 11.8 wrapped) = 12.2 rad, the short way.
    unwrapped = KitchenMotionPlanner.unwrapped(path=path)
    np.testing.assert_allclose(unwrapped[1][2], 11.8 + np.arctan2(np.sin(-12.1), np.cos(-12.1)))
    assert unwrapped[1][2] > CONTINUOUS_LIMIT_RAD


@pytest.mark.parametrize(
    ("end", "expected"),
    [(11.99, -2), (1.0, 0), (4.0, -1), (-3.5, 1), (-11.99, 2)],
)
def test_the_chosen_turn_is_the_one_nearest_neutral(*, end, expected):
    q = np.zeros(7)
    q[2] = end
    turns = KitchenMotionPlanner.turns_toward(end=q, neutral=np.zeros(7))
    assert turns.tolist() == [0, 0, expected, 0, 0, 0, 0]


def test_a_path_past_the_range_is_refused():
    start = np.zeros(7)
    start[2] = 11.8
    path = [start.copy(), start.copy()]
    path[1][2] = CONTINUOUS_LIMIT_RAD + 0.1
    with pytest.raises(PlanningRefused, match="joint 3"):
        KitchenMotionPlanner.check_range(path=path, what="the carry pose")


def test_a_path_from_past_the_range_heading_inwards_is_not_refused():
    start = np.zeros(7)
    start[2] = CONTINUOUS_LIMIT_RAD - 0.01
    path = [start.copy(), start.copy()]
    path[1][2] = 10.0
    KitchenMotionPlanner.check_range(path=path, what="the carry pose")


def test_the_pick_unwinds_a_joint_more_than_half_a_turn_from_neutral(*, monkeypatch):
    env = _wound(joint=2, turns=0.6)
    c = env.controller()
    before = c.data.qpos[c.arm_q].copy()
    seen: list[np.ndarray] = []

    def record(self, **_):  # noqa: PLR0917 -- patched bound method
        seen.append(self.data.qpos[self.arm_q].copy())
        raise Stop

    monkeypatch.setattr(KitchenController, "pitcher_grasp_pose", record)
    with pytest.raises(Stop):
        c.pick_pitcher()
    offset = seen[0] - c.arm_neutral_q
    assert np.all(np.abs(offset[list(CONTINUOUS)]) <= np.pi)
    wrapped = np.arctan2(np.sin(seen[0] - before), np.cos(seen[0] - before))
    assert np.max(np.abs(wrapped)) < 0.03
