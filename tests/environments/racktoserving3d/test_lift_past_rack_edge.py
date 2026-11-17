"""A welded pick beside the rack's front edge plans a lift that clears it, in execution too.

Robot batch return after a Sweep, seed 10 (2026-11-13): the second pick of the return,
`cube_2` on the plate tucked under the rack, was planned from the rack's far side, the
hand leaned 30 degrees away from the base. The lift back to the hover then runs up and
toward the rack, and the dumpling, snapped upright into the hand, passes the front edge
of the rack's support slab with no clearance at all: checked clear in planning, refused
in execution, "Carried food would collide with rack_0_support: overlap=0.0000,
bottom=0.9219, top=0.9250". The plan never checked the lift with the dumpling in the
hand, so it had no reason to prefer a grasp whose lift moves away from the rack.

The same return after a Sweep, seed 14, and after a GatherAndSweep, seed 16, failed
the same way on the same edge. Each fixture is the state its failing pick started from.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from .test_welded_food import restored

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)
FIXTURES = Path(__file__).parent / "fixtures"
STARTS = sorted(FIXTURES.glob("robot_return_*_lift_past_rack_front.json"))


def _closest_to_the_rack(*, env, name: str) -> float:
    """The dumpling's closest distance, in the live simulation, to the rack's support.

    MuJoCo's mesh-box distance returns exactly 0.0 at some poses millimetres apart; the
    same query 10 µm off decides, as `_FoodMotion.held_food_gaps` does.
    """
    import mujoco

    backend = env.backend()
    model, data = backend.model(), backend.data()
    slab = model.geom("rack_0_support").id

    def distance(*, geom: int) -> float:
        value = float(mujoco.mj_geomDistance(model, data, geom, slab, 0.05, None))
        if value != 0.0:
            return value
        saved = data.geom_xpos[geom].copy()
        nudged = []
        for offset in 1e-5 * np.eye(3):
            data.geom_xpos[geom] = saved + offset
            nudged.append(float(mujoco.mj_geomDistance(model, data, geom, slab, 0.05, None)))
        data.geom_xpos[geom] = saved
        return min((v for v in nudged if v != 0.0), default=0.0)

    return min(distance(geom=geom) for geom in backend.body_geoms(name=name))


def test_there_is_one_start_per_failed_return():
    assert len(STARTS) == 3


@pytest.mark.parametrize("fixture", STARTS, ids=[path.stem for path in STARTS])
def test_a_pick_beside_the_racks_front_edge_lifts_clear_of_it(*, fixture):
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController, WeldedFoodSkills

    env, target = restored(fixture=fixture)
    try:
        backend = env.backend()
        closest = []

        def observe() -> None:
            if backend.held(name=target):
                closest.append(_closest_to_the_rack(env=env, name=target))

        backend.set_tick_observer(observer=observe)
        food = RackFoodController(controllers=env.rack_controllers())
        WeldedFoodSkills(food=food).pick(name=target, clear_of_food=True)
        backend.set_tick_observer(observer=None)
        assert backend.held(name=target)
        assert closest
        assert min(closest) > 0.001
    finally:
        env.close()
