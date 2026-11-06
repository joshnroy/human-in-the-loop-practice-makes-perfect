"""An arm motion may start inside the planning margin, provided it is not in contact.

MuJoCo's drop IK keeps 15 mm in its own robot model; PyBullet's is under a millimetre
off, so a drop can park the arm 14.4 mm from the scene as PyBullet measures it, inside
the 15 mm margin. BiRRT then refuses its own start, every time, and no skill or human
action moves the arm: practice ended `repeated_zero_step` in two Della runs. From such a
start, `KitchenMotionPlanner.plan_arm` plans a short escape at `ESCAPE_MARGIN_M` until
the arm is outside the full margin, then the rest of the motion at the full margin.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_plan_from_inside_margin")

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED = (
    "kitchen_pick_pitcher_start_inside_margin_ees0.json",
    "kitchen_pick_pitcher_start_inside_margin_ees1.json",
)
PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)


def _recorded(*, name: str) -> KitchenEnvironment:
    state = json.loads((FIXTURES / name).read_text())
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


class Tuck:
    """PickPitcher's tuck from the controller's current state, planned directly."""

    @staticmethod
    def plan(*, env: KitchenEnvironment, margin: float | None = None):
        from scipy.spatial.transform import Rotation

        from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps
        from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

        c = env.controller()
        _, upright, _ = c.pitcher_grasp_pose()
        base = c.data.qpos[c.base_q].copy()
        carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler("z", float(base[2])).apply(
            ps.CarryToPourController.CARRY_HAND_IN_BASE
        )
        sim = KitchenPyBulletSim.from_controller(controller=c)
        sim.set_state(c.scene._get_object_centric_state())
        kwargs = {} if margin is None else {"margin": margin}
        return sim, dict(
            sim=sim,
            start=c.data.qpos[c.arm_q].copy(),
            position=carry,
            rotation=upright,
            what="the tucked arm",
            reference=c.arm_neutral_q.copy(),
            **kwargs,
        )

    @staticmethod
    def collides(*, sim, q, margin: float) -> bool:
        from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object

        from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
            KitchenMotionPlanner,
        )

        return bool(
            check_collisions_with_held_object(
                sim.robot,
                sim.get_collision_bodies(),
                sim.physics_client_id,
                None,
                None,
                KitchenMotionPlanner.with_fingers(sim=sim, arm=q),
                distance_threshold=margin,
            )
        )


@pytest.mark.parametrize("name", RECORDED)
def test_the_recorded_start_is_inside_the_margin_but_clear_of_the_escape_margin(*, name):
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        ESCAPE_MARGIN_M,
    )
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import PLANNING_MARGIN_M

    env = _recorded(name=name)
    sim, kwargs = Tuck.plan(env=env)
    try:
        assert Tuck.collides(sim=sim, q=kwargs["start"], margin=PLANNING_MARGIN_M)
        assert not Tuck.collides(sim=sim, q=kwargs["start"], margin=ESCAPE_MARGIN_M)
    finally:
        sim.close()
        env.close()


@pytest.mark.parametrize("name", RECORDED)
def test_a_tuck_from_inside_the_margin_escapes_then_keeps_the_full_margin(*, name):
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        ESCAPE_MARGIN_M,
        KitchenMotionPlanner,
    )
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import PLANNING_MARGIN_M

    env = _recorded(name=name)
    sim, kwargs = Tuck.plan(env=env)
    try:
        path = KitchenMotionPlanner.plan_arm(**kwargs)
        np.testing.assert_allclose(path[0], kwargs["start"])
        inside = [Tuck.collides(sim=sim, q=q, margin=PLANNING_MARGIN_M) for q in path]
        # Only an unbroken run of waypoints from the start lies inside the margin...
        escape = inside.index(False)
        assert escape >= 1
        assert not any(inside[escape:])
        # ...and none of it inside the escape margin.
        assert not any(Tuck.collides(sim=sim, q=q, margin=ESCAPE_MARGIN_M) for q in path)
    finally:
        sim.close()
        env.close()


@pytest.mark.parametrize("name", RECORDED)
def test_pick_pitcher_from_the_recorded_dead_end_now_succeeds(*, name):
    env = _recorded(name=name)
    try:
        c = env.controller()
        seen = Clearance.record(env=env)
        steps = c.steps
        env.take_action(action=PICK_PITCHER)
        assert env.last_info["error"] != "No collision-free arm path to the tucked arm"
        assert c.steps > steps
        assert env.last_info["success"] is True, env.last_info
        assert "planned tuck arm before pitcher pick" in [row[0] for row in seen]
        Clearance.check(seen=seen)
    finally:
        env.close()


def test_a_start_within_the_escape_margin_is_still_refused(*, monkeypatch):
    """In contact, or nearly: no escape is planned, and the refusal is unchanged."""
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    # The recorded start is 14.4 mm away; an escape margin above that puts it "in contact".
    monkeypatch.setattr(ps, "ESCAPE_MARGIN_M", 0.0146)
    env = _recorded(name=RECORDED[0])
    sim, kwargs = Tuck.plan(env=env)
    try:
        with pytest.raises(ps.PlanningRefused, match="No collision-free arm path to the tucked"):
            ps.KitchenMotionPlanner.plan_arm(**kwargs)
    finally:
        sim.close()
        env.close()


def test_a_start_outside_the_margin_plans_exactly_as_before():
    """The full-margin plan is upstream's `plan_checked_arm_motion`, unchanged."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import plan_checked_arm_motion

    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
    )
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import PLANNING_MARGIN_M

    env = _recorded(name=RECORDED[0])
    c = env.controller()
    # The episode-initial (tucked) arm at the recorded base: outside the margin.
    c.data.qpos[c.arm_q] = c.arm_neutral_q
    mujoco.mj_forward(c.model, c.data)
    sim, kwargs = Tuck.plan(env=env)
    try:
        assert not Tuck.collides(sim=sim, q=kwargs["start"], margin=PLANNING_MARGIN_M)
        now = KitchenMotionPlanner.plan_arm(**kwargs)
        start = KitchenMotionPlanner.with_fingers(sim=sim, arm=kwargs["start"])
        sim.robot.set_joints(start)
        goal = KitchenMotionPlanner.with_fingers(sim=sim, arm=now[-1])
        before = plan_checked_arm_motion(
            sim.robot,
            start,
            goal,
            collision_bodies=sim.get_collision_bodies(),
            physics_client_id=sim.physics_client_id,
            distance_threshold=PLANNING_MARGIN_M,
        )
        assert before is not None
        assert len(now) == len(before[1])
        for a, b in zip(now, before[1], strict=True):
            # Rounding only: the goal is passed back in through `now[-1]`.
            np.testing.assert_allclose(a, np.asarray(b[:7], dtype=float), rtol=0, atol=1e-12)
    finally:
        sim.close()
        env.close()
