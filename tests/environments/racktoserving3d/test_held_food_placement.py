"""A held dumpling the placement skills could not put down anywhere.

`held_food_plate_holds_four_ees0.json` is the Della rack-80k-3d0cb30 EES seed 0 state
at practice step 5929, after PickFood(cube_0): cube_0 held with the pads reaching 15 mm
below its lowest point, cube_3 and cube_4 served in the middle of the plate's well,
cube_1 and cube_2 on its rim. Every following PlaceFoodOnTray refused at 0 steps ("No
unoccupied food placement at destination") and every ReturnFoodToRack at 0 steps (the
pads in the rack's wires at each pre-placement's descent), 0/9 between them.
"""

from pathlib import Path

import numpy as np

from tests.environments.racktoserving3d.test_planned_food import restored

FIXTURE = Path(__file__).parent / "fixtures" / "held_food_plate_holds_four_ees0.json"


def test_the_plate_has_room_for_the_held_food_at_some_hand_yaw():
    """The as-held footprint is diagonal and a third wider than one set down square; a
    7 x 7 grid of that footprint found no spot between the served food and the rim."""
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController

    env = restored(fixture=FIXTURE)
    try:
        food = RackFoodController(controllers=env.rack_controllers())
        here = food.shelf.site_pose()
        placements = food._placements(name="cube_0", destination="box_0", rotation=here[:3, :3])
        assert len(placements) >= 10
    finally:
        env.close()


def test_food_held_low_in_the_hand_is_released_above_the_pads_tips():
    """Released by the food's own lowest point, the pads would end 12 mm into the
    support; the release rises by how far they reach below it."""
    from hitl_pmp.environments.racktoserving3d.food import (
        RELEASE_CLEARANCE_M,
        RackFoodController,
    )

    env = restored(fixture=FIXTURE)
    try:
        food = RackFoodController(controllers=env.rack_controllers())
        release = food._release_height(name="cube_0")
        assert release > RELEASE_CLEARANCE_M + 0.01
    finally:
        env.close()


def test_food_held_low_in_the_hand_returns_to_the_rack():
    env = restored(fixture=FIXTURE)
    try:
        assert env.backend().held(name="cube_0")
        env.take_action(action=np.array([5.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.backend().location(name="cube_0") == "OnRack"
    finally:
        env.close()
