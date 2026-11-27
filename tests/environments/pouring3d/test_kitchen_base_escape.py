"""A base that starts inside an obstacle's footprint, but not deep in it, is recoverable.

In pouring-threefix-20261109-3586431e0 (EES seed 2, p = 0.1) a failed scripted floor pick
left the base pressed against the stovetop: its 2-D footprint overlapped the stovetop's,
so `run_base_motion_planning` refused its own start, and PickPitcher refused "No
collision-free base path to the pitcher pick pose" at 0 steps until practice ended. A human
reset moves the dumplings and the jug, never the robot. From such a start
`KitchenMotionPlanner.plan_base` first escapes in a short straight line, the footprint
inset by `BASE_ESCAPE_INSET_M` and the body at `ESCAPE_MARGIN_M`, until the base is
outside the full checks, then plans the rest exactly as before.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_base_escape")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_base_pressed_against_the_stove.json"
RECORDED = json.loads(FIXTURE.read_text())
PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)


class Recorded:
    """A static-method container, never instantiated."""

    @staticmethod
    def restored() -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=RECORDED["requested"])
        env.reset_to_seed(seed=RECORDED["scene_seed"], requested=RECORDED["requested"])
        c = env.controller()
        c.data.qpos[:] = RECORDED["qpos"]
        c.data.qvel[:] = RECORDED["qvel"]
        c.data.time = RECORDED["time"]
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        env.current_state = env.observe()
        return env

    @staticmethod
    def planner(*, env: KitchenEnvironment):
        from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

        c = env.controller()
        sim = KitchenPyBulletSim.from_controller(controller=c)
        state = c.scene._get_object_centric_state()
        sim.set_state(state)
        return sim, state


def test_the_recorded_base_is_inside_the_footprint_check_but_clear_of_the_escape_checks():
    from spatialmath import SE2

    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        BASE_ESCAPE_INSET_M,
        ESCAPE_MARGIN_M,
        KitchenMotionPlanner,
    )

    env = Recorded.restored()
    sim, state = Recorded.planner(env=env)
    try:
        c = env.controller()
        base, arm = c.data.qpos[c.base_q].copy(), c.data.qpos[c.arm_q].copy()
        assert KitchenMotionPlanner.base_pose_blocked(sim=sim, state=state, target=base, arm=arm)
        assert KitchenMotionPlanner.footprint_blocked(state=state, target=base)
        assert not KitchenMotionPlanner.footprint_blocked(
            state=state, target=base, inset=BASE_ESCAPE_INSET_M
        )
        escape = KitchenMotionPlanner.body_check(sim=sim, arm=arm, margin=ESCAPE_MARGIN_M)
        assert not escape(SE2(*(float(v) for v in base)))
    finally:
        sim.close()
        env.close()


def test_a_base_path_from_the_recorded_start_escapes_then_passes_every_check():
    from spatialmath import SE2

    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        BASE_ESCAPE_INSET_M,
        BASE_ESCAPE_MAX_M,
        KitchenMotionPlanner,
    )

    env = Recorded.restored()
    sim, state = Recorded.planner(env=env)
    try:
        c = env.controller()
        base, arm = c.data.qpos[c.base_q].copy(), c.data.qpos[c.arm_q].copy()
        path = KitchenMotionPlanner.plan_base(
            sim=sim, state=state, target=c.base_home.copy(), arm=arm, what="home"
        )
        np.testing.assert_allclose(path[0], base)
        np.testing.assert_allclose(path[-1], c.base_home)
        # One straight escape segment, heading held, no longer than allowed...
        assert path[1][2] == pytest.approx(base[2])
        assert np.linalg.norm(path[1][:2] - base[:2]) <= BASE_ESCAPE_MAX_M + 1e-9
        # ...along which the escape checks hold...
        for u in np.linspace(0.0, 1.0, 21):
            pose = base + u * (path[1] - base)
            assert not KitchenMotionPlanner.footprint_blocked(
                state=state, target=pose, inset=BASE_ESCAPE_INSET_M
            )
        # ...and every later waypoint passes the full checks.
        for pose in path[1:]:
            assert not KitchenMotionPlanner.base_pose_blocked(
                sim=sim, state=state, target=pose, arm=arm
            )
        body = KitchenMotionPlanner.body_check(sim=sim, arm=arm)
        assert not any(body(SE2(*(float(v) for v in pose))) for pose in path[1:])
    finally:
        sim.close()
        env.close()


def test_pick_pitcher_from_the_recorded_dead_end_no_longer_refuses_its_base_path():
    env = Recorded.restored()
    try:
        c = env.controller()
        assert c.scene.cube_locations() == ["pitcher", "pitcher", "pitcher"]
        steps = c.steps
        env.take_action(action=PICK_PITCHER)
        assert env.last_info["error"] != "No collision-free base path to the pitcher pick pose"
        assert c.steps > steps
        assert env.last_info["success"] is True, env.last_info
        assert c.held == "pitcher"
    finally:
        env.close()


def test_a_start_deep_in_the_footprint_is_still_refused(*, monkeypatch):
    """Inside the escape inset too: no escape is planned, and the refusal is unchanged."""
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    # The recorded footprint overlaps the stovetop's by 3 to 5 cm.
    monkeypatch.setattr(ps, "BASE_ESCAPE_INSET_M", 0.02)
    env = Recorded.restored()
    sim, state = Recorded.planner(env=env)
    try:
        c = env.controller()
        with pytest.raises(ps.PlanningRefused, match="^No collision-free base path to home$"):
            ps.KitchenMotionPlanner.plan_base(
                sim=sim,
                state=state,
                target=c.base_home.copy(),
                arm=c.data.qpos[c.arm_q].copy(),
                what="home",
            )
    finally:
        sim.close()
        env.close()


def test_a_start_outside_the_checks_plans_exactly_as_before():
    """The path is upstream's `run_base_motion_planning`, unchanged."""
    from kinder_models.dynamic3d.utils import (
        WORLD_X_BOUNDS,
        WORLD_Y_BOUNDS,
        run_base_motion_planning,
    )
    from spatialmath import SE2

    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
    )

    env = Recorded.restored()
    c = env.controller()
    # The episode-initial arm at the aisle pose the floor pick started from: clear.
    c.data.qpos[c.base_q] = [-0.349, -1.05, 0.0]
    c.data.qpos[c.arm_q] = c.arm_neutral_q
    mujoco.mj_forward(c.model, c.data)
    sim, state = Recorded.planner(env=env)
    try:
        arm, base = c.arm_neutral_q.copy(), c.data.qpos[c.base_q].copy()
        assert not KitchenMotionPlanner.base_pose_blocked(
            sim=sim, state=state, target=base, arm=arm
        )
        target = c.base_home.copy()
        now = KitchenMotionPlanner.plan_base(
            sim=sim, state=state, target=target, arm=arm, what="home"
        )
        before = run_base_motion_planning(
            state=state,
            target_base_pose=SE2(*(float(v) for v in target)),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            body_collision_fn=KitchenMotionPlanner.body_check(sim=sim, arm=arm),
        )
        assert before is not None
        assert len(now) == len(before)
        for a, b in zip(now, before, strict=True):
            np.testing.assert_array_equal(a, [b.x, b.y, b.theta()])
    finally:
        sim.close()
        env.close()
