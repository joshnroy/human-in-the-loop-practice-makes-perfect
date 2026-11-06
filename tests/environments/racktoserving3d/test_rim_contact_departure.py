"""A pad resting against the welded plate's rim must not strand a held-food skill.

Della rack-80k-55d61a4, EES seed 0: PickScraper, GatherAndSweep, ReturnScraper, then
PickFood(cube_0) from the plate closed on a dumpling beside the rim, the right pad 14 um
into `box_0_rim_4`. Its lift failed at its first control step, 457 steps in: "Scraper
motion would collide, cost 0.00292: [('box_0_rim_4', 'robot_right_pad1', -0.00047),
...]". Every one of the next 20 PlaceFoodOnTray / ReturnFoodToRack calls then refused
at 0 steps with the same contact, and the zero-step guard ended practice at step 5,734.

Before the weld the plate was free, so the pad nudged it. Welded, it keeps the contact,
and the motion leaving it, straight up or back along the approach, first presses the
pad's corner up to 0.24 mm deeper into the thin, tilted rim box before clearing it: a
persist-not-deepen guard rejects that line. A departure on which no starting contact
deepens exists (up and away from the rim), so the skills must take it.

`stuck_ees0_pick_food_rim_start.json`: recording row 5305, the PickFood's start.
`stuck_ees0_held_food_pad_in_rim.json`: recording row 5764, the tick before the first
zero-step call, restored without settling (as that call saw it).
"""

from pathlib import Path

import numpy as np
import pytest

from tests.environments.racktoserving3d.test_planned_food import restored
from tests.environments.racktoserving3d.test_stuck_states import (
    done_or_refused,
    restored_frozen,
)

FIXTURES = Path(__file__).parent / "fixtures"
PICK_START = FIXTURES / "stuck_ees0_pick_food_rim_start.json"
HELD_IN_RIM = FIXTURES / "stuck_ees0_held_food_pad_in_rim.json"
# The same sequence replayed on agni from the first version of this fix (ceb22b7): the
# left pad on `box_0_rim_5`. That version checked one 5 mm move but executed it as a
# Cartesian line with its own IK, whose midpoint touched the rim; 14 zero-step calls
# followed.
LOCAL_PICK_START = FIXTURES / "local_ees0_pick_food_rim_start.json"
LOCAL_HELD_IN_RIM = FIXTURES / "local_ees0_held_food_pad_in_rim.json"
PICK_STARTS = pytest.mark.parametrize("start", [PICK_START, LOCAL_PICK_START], ids=["della", "agni"])
HELD_STARTS = pytest.mark.parametrize(
    "held", [HELD_IN_RIM, LOCAL_HELD_IN_RIM], ids=["della", "agni"]
)


@PICK_STARTS
def test_a_pick_food_whose_closed_pad_touches_the_rim_lifts_the_food(*, start):
    env = restored(fixture=start)
    try:
        steps, error = done_or_refused(env=env, action=np.array([3.0, 0, 0, 0, 0]))
        assert error is None, f"{steps} steps, then: {error}"
        assert env.backend().held(name="cube_0")
    finally:
        env.close()


@pytest.mark.parametrize("skill", [4, 5], ids=["PlaceFoodOnTray", "ReturnFoodToRack"])
@HELD_STARTS
def test_a_held_food_skill_starting_with_a_pad_in_the_rim_leaves_it(*, skill, held):
    env = restored_frozen(fixture=held)
    try:
        controllers = env.rack_controllers()
        assert env.backend().held(name="cube_0")
        controllers.held_food = controllers.shelf.held_cube = "cube_0"
        steps, error = done_or_refused(env=env, action=np.array([float(skill), 0, 0, 0, 0]))
        assert steps > 0, error
        assert "box_0_rim" not in (error or ""), f"{steps} steps, then: {error}"
        assert env.backend().held(name="cube_0") or error is None
    finally:
        env.close()


@HELD_STARTS
def test_the_departure_never_presses_a_pad_deeper_into_the_rim(*, held):
    """Measured at every control step of the departure: each starting pad-fixture
    contact is never deeper than it began and no new one appears, the food still held.
    (The commanded end clears the rim; the tracked hand may still graze it by microns,
    and the rise that follows leaves it.)"""
    from hitl_pmp.environments.racktoserving3d.food import DEPARTURE_TOLERANCE, RackFoodController

    env = restored_frozen(fixture=held)
    try:
        controllers = env.rack_controllers()
        controllers.held_food = controllers.shelf.held_cube = "cube_0"
        food = RackFoodController(controllers=controllers)
        food.carried = "cube_0"
        shelf = food.shelf
        fixed = shelf.cupboard_geoms | shelf.box_geoms

        def live() -> dict[tuple[int, int], float]:
            pairs: dict[tuple[int, int], float] = {}
            for contact in shelf.data.contact[: shelf.data.ncon]:
                for pad, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                    if pad in shelf.pad_geoms and other in fixed and contact.dist <= 0:
                        key = (int(pad), int(other))
                        pairs[key] = min(pairs.get(key, np.inf), float(contact.dist))
            return pairs

        seen: list[dict[tuple[int, int], float]] = []
        stepped = shelf.step

        def step(**kwargs):
            stepped(**kwargs)
            seen.append(live())

        shelf.step = step
        with controllers.feedback(), food.motion.departing_from_contacts():
            start = dict(food.motion.departing)
            assert start, "the fixture's pad no longer touches the rim"
            food.motion.depart(label="food_depart")
        assert seen, "the departure moved nothing"
        for index, contacts in enumerate(seen):
            for pair, distance in contacts.items():
                assert pair in start, (index, pair, distance)
                assert distance >= start[pair] - DEPARTURE_TOLERANCE, (index, pair, distance)
        assert env.backend().held(name="cube_0")
    finally:
        env.close()
