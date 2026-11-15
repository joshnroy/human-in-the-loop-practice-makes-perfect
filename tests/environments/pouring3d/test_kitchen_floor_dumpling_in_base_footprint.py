"""A dumpling on the floor under the base, clear of its 3-D body, does not trap it.

In pouring-cost-10seed-20261112 (human-reset cost 10, Curtis, seed 8) a pour flung a
dumpling to the floor between the casters of the base, which held the jug. The 3-D body
check is clear there, but the dumpling lies inside the base's 2-D footprint, and inside
the footprint inset by `BASE_ESCAPE_INSET_M` too, so `base_escape` refused its own start
and PlacePitcherAtHome refused "No collision-free base path to placement" at 0 steps
until practice ended. Only when that escape finds nothing does `plan_base` try again
with floor dumplings the body clears left out of the escape's inset footprint; the
escape still ends where the full checks, that dumpling's footprint included, pass.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_floor_dumpling_in_base_footprint")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_floor_dumpling_in_base_footprint.json"
RECORDED = json.loads(FIXTURE.read_text())
FLOOR_DUMPLING = "potsticker_1"


class Recorded:
    """A static-method container, never instantiated."""

    @staticmethod
    def restored() -> KitchenEnvironment:
        """The recorded state, the jug welded in the hand as the run held it."""
        env = KitchenEnvironment(requested_count=RECORDED["requested"])
        env.reset_to_seed(seed=RECORDED["scene_seed"], requested=RECORDED["requested"])
        c = env.controller()
        c.data.qpos[:] = RECORDED["qpos"]
        c.data.qvel[:] = RECORDED["qvel"]
        c.data.time = RECORDED["time"]
        mujoco.mj_forward(c.model, c.data)
        c.scene.attach_pitcher()
        mujoco.mj_forward(c.model, c.data)
        c.held = "pitcher"
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        env.current_state = env.observe()
        return env

    @staticmethod
    def planner(*, env: KitchenEnvironment):
        """The placement's own planning inputs: the scene, the jug as grasped and the
        arm tucked into the carry pose, exactly as `PlacePitcherController.reset`."""
        from spatialmath import SE2

        from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps
        from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

        c = env.controller()
        sim = KitchenPyBulletSim.from_controller(controller=c)
        state = c.scene._get_object_centric_state()
        base, arm = c.data.qpos[c.base_q].copy(), c.data.qpos[c.arm_q].copy()
        sim.set_state(state)
        sim.set_robot_base(SE2(*(float(v) for v in base)))
        riders = tuple(
            n
            for n, at in zip(c.scene.cube_names, c.scene.cube_locations(), strict=True)
            if at == "pitcher"
        )
        held = ps.HeldObject.grasped(sim=sim, arm=arm, name="pitcher", riders=riders)
        carry = np.array([base[0], base[1], 0.0]) + Rotation.from_euler("z", float(base[2])).apply(
            ps.CarryToPourController.CARRY_HAND_IN_BASE
        )
        tuck = ps.KitchenMotionPlanner.plan_arm(
            sim=sim,
            start=arm,
            position=carry,
            rotation=c.level_hand_rotation(),
            what="the carry pose",
            held=held,
            neutral=c.arm_neutral_q.copy(),
        )
        return sim, state, held, tuck[-1]


def test_the_recorded_floor_dumpling_is_in_the_footprint_but_clear_of_the_body():
    from spatialmath import SE2

    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    env = Recorded.restored()
    c = env.controller()
    assert c.scene.cube_locations()[1] == "floor"
    sim, state, held, carried = Recorded.planner(env=env)
    try:
        base = c.data.qpos[c.base_q].copy()
        P = ps.KitchenMotionPlanner
        assert P.footprint_blocked(state=state, target=base)
        assert P.footprint_blocked(state=state, target=base, inset=ps.BASE_ESCAPE_INSET_M)
        assert not P.footprint_blocked(
            state=state, target=base, inset=ps.BASE_ESCAPE_INSET_M, ignore=(FLOOR_DUMPLING,)
        )
        pose = SE2(*(float(v) for v in base))
        assert not P.body_check(sim=sim, arm=carried, held=held)(pose)
        assert P.floor_dumplings(state=state) == (FLOOR_DUMPLING,)
    finally:
        sim.close()
        env.close()


def test_placement_from_the_recorded_stall_escapes_then_passes_every_check():
    from spatialmath import SE2

    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    env = Recorded.restored()
    c = env.controller()
    sim, state, held, carried = Recorded.planner(env=env)
    try:
        P = ps.KitchenMotionPlanner
        base, target = c.data.qpos[c.base_q].copy(), c.base_home.copy()
        path = P.plan_base(
            sim=sim, state=state, target=target, arm=carried, what="placement", held=held
        )
        np.testing.assert_allclose(path[0], base)
        np.testing.assert_allclose(path[-1], target)
        # One straight escape segment, heading held, no longer than allowed...
        assert path[1][2] == pytest.approx(base[2])
        assert np.linalg.norm(path[1][:2] - base[:2]) <= ps.BASE_ESCAPE_MAX_M + 1e-9
        escape_body = P.body_check(sim=sim, arm=carried, held=held, margin=ps.ESCAPE_MARGIN_M)
        for u in np.linspace(0.0, 1.0, 21):
            pose = base + u * (path[1] - base)
            # ...clear of every other footprint inset, and of the body at the escape margin...
            assert not P.footprint_blocked(
                state=state, target=pose, inset=ps.BASE_ESCAPE_INSET_M, ignore=(FLOOR_DUMPLING,)
            )
            assert not escape_body(SE2(*(float(v) for v in pose)))
        # ...and every later waypoint passes the full checks, the dumpling's footprint too.
        body = P.body_check(sim=sim, arm=carried, held=held)
        for pose in path[1:]:
            assert not P.footprint_blocked(state=state, target=pose)
            assert not body(SE2(*(float(v) for v in pose)))
    finally:
        sim.close()
        env.close()


def test_the_dumpling_is_still_an_obstacle_when_the_ordinary_escape_succeeds(*, monkeypatch):
    """The fallback runs only where the escape it extends found nothing."""
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    calls: list[tuple[str, ...]] = []
    escape = ps.KitchenMotionPlanner.base_escape

    def recorded(**kwargs):
        calls.append(tuple(kwargs.get("escape_ignore", ())))
        return escape(**kwargs)

    monkeypatch.setattr(ps.KitchenMotionPlanner, "base_escape", staticmethod(recorded))
    env = Recorded.restored()
    c = env.controller()
    sim, state, held, carried = Recorded.planner(env=env)
    try:
        ps.KitchenMotionPlanner.plan_base(
            sim=sim,
            state=state,
            target=c.base_home.copy(),
            arm=carried,
            what="placement",
            held=held,
        )
        assert calls == [(), (FLOOR_DUMPLING,)]
    finally:
        sim.close()
        env.close()


def test_a_dumpling_off_the_floor_is_never_left_out():
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills as ps

    env = Recorded.restored()
    try:
        c = env.controller()
        state = c.scene._get_object_centric_state()
        locations = c.scene.cube_locations()
        floor = ps.KitchenMotionPlanner.floor_dumplings(state=state)
        assert floor == tuple(
            n for n, at in zip(c.scene.cube_names, locations, strict=True) if at == "floor"
        )
    finally:
        env.close()
