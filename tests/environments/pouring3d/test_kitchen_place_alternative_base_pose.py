"""PlacePitcherAtHome finds a reachable base pose when a dumpling occupies the nominal one.

Della 16949734 expectimax/pomdp/2: a closed-loop Pour1 flung potsticker_0 onto the floor
at about (0.02, -0.83), inside the robot's 2-D footprint at the placement's one fixed
base pose (`base_home`). Every base path must end there, so PlacePitcherAtHome was
refused 100 times at 0 ticks and practice ended with `repeated_zero_step`. The fixture is
that run's recorded state after the Pour (recording tick seq 159).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_place_alternative_base_pose")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_holding_jug_dumpling_on_floor.json"
PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
POUR1 = np.array([1, 0, 1, 1.0, 0, 0], dtype=float)
PLACE = np.array([2, 0, 1, 0, 0, 0], dtype=float)


def _rebuild_holding_jug(*, env: KitchenEnvironment) -> None:
    import mujoco

    state = json.loads(FIXTURE.read_text())
    c = env.controller()
    c.data.qpos[:] = state["qpos"]
    c.data.qvel[:] = state["qvel"]
    c.data.time = state["time"]
    mujoco.mj_forward(c.model, c.data)
    # The recording carries MuJoCo state only; the run held the jug, closed, here.
    c.held, c.grip = "pitcher", 1.0
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()


def test_from_the_recorded_dead_end_the_place_succeeds_from_a_nearby_base_pose():
    env = KitchenEnvironment(requested_count=1)
    try:
        _rebuild_holding_jug(env=env)
        c = env.controller()
        assert c.scene.cube_locations() == ["floor", "pitcher", "pitcher"]
        steps = c.steps
        seen = Clearance.record(env=env)
        env.take_action(action=PLACE)
        ticks = c.steps - steps
        print(f"PlacePitcherAtHome from seq 159: {ticks} ticks, info {env.last_info}")
        assert env.last_info.get("error") is None, env.last_info
        assert env.last_info["success"] is True, env.last_info
        assert c.held is None
        assert c.vessel_home(name="pitcher")
        position, rotation = c.body_pose(name="pitcher")
        assert np.linalg.norm(position[:2] - c.home[:2]) < 0.01
        assert rotation.magnitude() < np.deg2rad(5)
        # The dumpling on the floor was not pushed into the jug or anywhere else.
        assert c.scene.cube_locations() == ["floor", "pitcher", "pitcher"]
        Clearance.check(seen=seen)
    finally:
        env.close()


def test_a_free_nominal_base_pose_never_consults_the_alternatives(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills

    env = KitchenEnvironment(requested_count=1)
    try:
        env.take_action(action=PICK_PITCHER)
        assert env.last_info["success"] is True, env.last_info
        env.take_action(action=POUR1)
        assert env.last_info["error"] is None, env.last_info

        def never(**_):
            raise AssertionError("alternative base poses consulted for a free nominal pose")

        monkeypatch.setattr(
            parameterized_skills.PlacePitcherController, "alternative_base_targets", never
        )
        env.take_action(action=PLACE)
        assert env.last_info["success"] is True, env.last_info
        assert env.controller().vessel_home(name="pitcher")
    finally:
        env.close()


def test_the_alternatives_are_deterministic_nearest_first_and_exclude_the_nominal_pose():
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        PlacePitcherController,
    )

    nominal = np.array([0.0, -1.05, 0.0])
    first = PlacePitcherController.alternative_base_targets(nominal=nominal)
    again = PlacePitcherController.alternative_base_targets(nominal=nominal)
    assert len(first) == len(again) > 0
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not any(np.allclose(t, nominal) for t in first)
    distances = [
        PlacePitcherController.alternative_distance(nominal=nominal, target=t) for t in first
    ]
    assert distances == sorted(distances)
    # Small: within 0.25 m and 0.2 rad of the nominal pose.
    for t in first:
        assert np.linalg.norm(t[:2] - nominal[:2]) <= 0.25 + 1e-9
        assert abs(t[2] - nominal[2]) <= 0.2 + 1e-9
