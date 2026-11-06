"""The food guard's vectorised contact test answers every query as the per-contact loop
it replaced."""

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

pytestmark = pytest.mark.xdist_group("rack_food_guard_scene")


@pytest.fixture(scope="module")
def food_motion():
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController

    env = RackServingEnvironment(config=RackServingConfig(food_handling="physical"))
    env.reset_to_seed(seed=0)
    food = RackFoodController(controllers=env.rack_controllers())
    food.target_geoms = food.backend.body_geoms(name="cube_0")
    yield food
    env.close()


def touches_ungrasped_food_reference(*, food, foods):
    """The per-contact loop the vectorised check replaced, verbatim."""
    shelf = food.shelf
    scratch = shelf.ik_data
    for contact in scratch.contact[: scratch.ncon]:
        if contact.dist > 0:
            continue
        for robot, other in (
            (int(contact.geom1), int(contact.geom2)),
            (int(contact.geom2), int(contact.geom1)),
        ):
            if (
                robot in shelf.robot_geoms
                and other in foods
                and not (robot in shelf.pad_geoms and other in food.target_geoms)
            ):
                return True
    return False


def test_the_food_guard_finds_food_contacts_as_the_per_contact_loop(*, food_motion):
    """Arms reaching into the rack from a stance in front of it: the vectorised
    contact test and the loop it replaced agree, and both outcomes occur."""
    food = food_motion
    motion, shelf, backend = food.motion, food.shelf, food.backend
    centre = backend.data().xpos[backend.model().body("cube_0").id]
    base = np.array([centre[0] - 0.55, centre[1], 0.0])
    arm = shelf.data.qpos[shelf.qa].copy()
    foods = motion._food_geoms(carried=None)
    assert foods == {g for n in backend.cube_names for g in backend.body_geoms(name=n)}
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(1500):
        shelf.site_at(base=base + rng.uniform(-0.1, 0.1, 3), arm=arm + rng.uniform(-1.5, 1.5, 7))
        shelf.collision_cost(d=shelf.ik_data, holding=False)
        expected = touches_ungrasped_food_reference(food=food, foods=foods)
        assert motion._touches_ungrasped_food(scratch=shelf.ik_data, foods=foods) == expected
        hits += int(expected)
    assert 0 < hits < 1500
