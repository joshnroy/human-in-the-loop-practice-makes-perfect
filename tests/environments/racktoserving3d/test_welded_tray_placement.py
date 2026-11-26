"""A welded dumpling the robot could not set down on a plate that had room for it.

Two Della rack-prior2x2-b406ad8-20261106 EES runs ended on the repeated zero-step stop,
each holding cube_1 after a successful PickFood, every PlaceFoodOnTray refused with
"No unoccupied food placement at destination; retaining grasp":

- `tray_place_ees_p080_1_cube_1_held_two_served.json` (ees-p080 seed 1): two dumplings
  served. The welded pick had snapped cube_1 22 mm above the pads' tips, past the 20 mm
  drop the release height allows, so every yaw's search raised and the refusal was
  reported as a full plate; ReturnFoodToRack refused with the drop itself.
- `tray_place_ees_p025_2_cube_1_held_four_served.json` (ees-p025 seed 2): four served in
  a cross. The search kept the footprint 20 mm inside the well's inner edge, a margin
  for a dropped dumpling's landing scatter, which left no spot between the cross and
  the edge.

A welded placement snaps the dumpling upright and at rest onto its spot: it is never
dropped, so neither the drop limit nor the scatter margin applies to it.

With both lifted the four-served plate has free spots, but only on its far half (+x,
toward the rack): cube_1 was picked straight down off the floor, so it is held under a
vertical hand, and a placement keeps the grasp's lean. No stance reaches a far-half
hover with a vertical hand. A hand leaned 30 degrees reaches one, and the set-down
snaps the food upright whatever the lean, but each lean candidate whose base or arm
path fails costs 60-90 s of the planner's 60 s deadline, so whether a leaned search
plans depends on which spot comes first. That is a follow-up.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from .test_welded_food import restored, upright

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)

FIXTURES = Path(__file__).parent / "fixtures"
TWO_SERVED = FIXTURES / "tray_place_ees_p080_1_cube_1_held_two_served.json"
FOUR_SERVED = FIXTURES / "tray_place_ees_p025_2_cube_1_held_four_served.json"
PLACE_ON_TRAY, RETURN_TO_RACK = 4.0, 5.0


def _held(*, fixture: Path):
    from hitl_pmp.environments.racktoserving3d.welded_food import HAND, GraspWeld

    env, _ = restored(fixture=fixture)
    held = json.loads(fixture.read_text())["held"]
    backend = env.backend()
    GraspWeld.attach(model=backend.model(), data=backend.data(), hand=HAND, name=held)
    controllers = env.rack_controllers()
    controllers.held_food = controllers.shelf.held_cube = held
    assert backend.held(name=held)
    return env, held


@pytest.mark.parametrize("fixture", [TWO_SERVED, FOUR_SERVED], ids=["two_served", "four_served"])
def test_the_plate_has_a_free_spot_for_the_held_food(*, fixture):
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController

    env, held = _held(fixture=fixture)
    try:
        food = RackFoodController(controllers=env.rack_controllers())
        here = food.shelf.site_pose()
        assert food._placements(name=held, destination="box_0", rotation=here[:3, :3])
    finally:
        env.close()


@pytest.mark.parametrize(
    "fixture",
    [
        TWO_SERVED,
        pytest.param(
            FOUR_SERVED,
            marks=pytest.mark.xfail(
                strict=True,
                reason="free spots only on the far half, beyond a vertical hand's reach",
            ),
        ),
    ],
    ids=["two_served", "four_served"],
)
def test_the_held_food_is_served(*, fixture):
    env, held = _held(fixture=fixture)
    try:
        backend = env.backend()
        served = [n for n in backend.cube_names if backend.location(name=n) == "OnTray"]
        env.take_action(action=np.array([PLACE_ON_TRAY, float(held[-1]), 0, 0, 0]))
        assert env.last_skill_error() is None
        assert not backend.held(name=held)
        assert upright(env=env, name=held) > 0.98
        for name in [*served, held]:
            assert backend.location(name=name) == "OnTray", name
    finally:
        env.close()


def test_food_welded_high_in_the_hand_returns_to_the_rack():
    env, held = _held(fixture=TWO_SERVED)
    try:
        env.take_action(action=np.array([RETURN_TO_RACK, float(held[-1]), 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.backend().location(name=held) == "OnRack"
    finally:
        env.close()
