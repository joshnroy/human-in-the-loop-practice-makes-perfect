"""The robot's batch return takes a dumpling out of a pile without throwing its neighbours.

Robot batch return after a GatherAndSweep, seed 2 (2026-11-13): the second pick from
the five on the plate snapped `cube_1` into the hand 13 mm inside `cube_2`; one physics
step later `cube_2` was moving at 2.3 m/s, then 4.7 m/s, and came to rest on the floor
1.2 m away, and `cube_4` was pushed off the plate onto the counter. A welded pick turns
the dumpling to the hand (`WeldedFood.snap_into_hand`), whatever lies beside it.

Here three dumplings lie side by side on the plate, long axes along y, and the middle
one is picked from the counter's front, which turns it a quarter turn into both others.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.layout import RackLayout

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)
SETUP_A = RackServingConfig(
    start_layout="middle_three",
    side_offset=0.13,
    plate_tuck=0.035,
    surface_kind="wire_solid",
    scraper_attachment="grasp_weld",
    two_route_rewards=True,
    bulk_transfer_only=True,
    robot_batch_return=True,
)
MIDDLE, NEIGHBOURS = "cube_1", ("cube_2", "cube_3")
SIDE_BY_SIDE_M = 0.036


def _three_side_by_side_on_the_plate():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=SETUP_A)
    env.reset_to_seed(seed=0)
    backend = env.backend()
    plate = backend.support(name="box_0")
    x, y = (plate.x_min + plate.x_max) / 2, (plate.y_min + plate.y_max) / 2
    height = plate.top_z + backend.config.dumpling_height / 2 + 0.002
    backend._commit(
        placements={
            name: (np.array([x + offset, y, height]), RackLayout.quaternion(yaw=0.0))
            for name, offset in zip(
                (MIDDLE, *NEIGHBOURS), (0.0, SIDE_BY_SIDE_M, -SIDE_BY_SIDE_M), strict=True
            )
        }
    )
    env.rack_controllers().shelf.hold(count=60, label="food_onto_plate")
    return env


def _centres(*, env):
    backend = env.backend()
    return {name: np.array(backend.food_centre(name=name)) for name in backend.cube_names}


def _moved(*, env, before):
    """How far each of the middle dumpling's neighbours has moved, once at rest."""
    env.rack_controllers().shelf.hold(count=60, label="food_settles")
    after = _centres(env=env)
    return {name: float(np.linalg.norm(after[name] - before[name])) for name in NEIGHBOURS}


def test_a_plain_welded_pick_from_between_two_dumplings_throws_them():
    """The defect, as PickFood still has it: the scene this file's fix is tested on."""
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController, WeldedFoodSkills

    env = _three_side_by_side_on_the_plate()
    try:
        before = _centres(env=env)
        WeldedFoodSkills(food=RackFoodController(controllers=env.rack_controllers())).pick(
            name=MIDDLE
        )
        assert max(_moved(env=env, before=before).values()) > 0.02
    finally:
        env.close()


def test_the_robot_returns_pick_leaves_the_neighbours_where_they_lie():
    from hitl_pmp.environments.racktoserving3d.food import (
        SNAP_INTO_FOOD,
        RackFoodController,
        WeldedFoodSkills,
    )
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = _three_side_by_side_on_the_plate()
    try:
        backend = env.backend()
        before = _centres(env=env)
        skills = WeldedFoodSkills(food=RackFoodController(controllers=env.rack_controllers()))
        try:
            skills.pick(name=MIDDLE, clear_of_food=True)
        except SkillFailed as refused:
            # No height cleared the others: the dumpling is back where it lay, unheld.
            assert SNAP_INTO_FOOD in str(refused)
            assert not backend.held(name=MIDDLE)
            assert np.linalg.norm(_centres(env=env)[MIDDLE] - before[MIDDLE]) < 0.001
        else:
            assert backend.held(name=MIDDLE)
        moved = _moved(env=env, before=before)
        assert max(moved.values()) < 0.002, moved
        for name in NEIGHBOURS:
            assert backend.location(name=name) == "OnTray"
    finally:
        env.close()
