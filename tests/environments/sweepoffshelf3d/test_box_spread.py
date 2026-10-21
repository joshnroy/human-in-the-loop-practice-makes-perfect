"""Cubes let go over the box are spread across it, and a cube with a neighbour on one
side has grasps shifted away from it. Before, every PlaceCubeInBox dropped at the box's
centre: five placements left the cubes in one touching row, and in launch ea9b45b15
(ees/ees/0, the 2026-10-21 refusal diagnosis's case 1) no cube in such a row had a
single grasp, so ReturnCubeToShelfFromBox was refused for the rest of the run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet

STATES = Path(__file__).resolve().parent / "data/refusal_states.jsonl"
CUBES = [f"cube_{i}" for i in range(5)]

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_box_spread"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _kinder(*, env):
    kinder = env.controllers()._kinder  # noqa: SLF001
    kinder.sync()
    return kinder


def _act(*, env, skill, cube):
    action = np.array([float(SKILL_IDS[skill]), float(cube.split("_")[1]), 0.0, 0.0])
    env.take_action(action=action)
    return env.get_current_state().abstract_atoms


def _put_in_box(*, env, cube, x, y):
    core = env.backend().core()
    floor = float(core.get_object("box_0").get_pose()[0][2])
    obj = core.get_object(cube)
    obj.set_pose([x, y, floor + 0.01 + O.cube_half_size], [1.0, 0.0, 0.0, 0.0])
    obj.set_velocity([0.0] * 3, [0.0] * 3)
    env.backend().sim().forward()
    assert core.cube_in_box(cube)


def _touching_pairs(*, env):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    pairs = set()
    for contact in data.contact[: data.ncon]:
        a = model.body(model.geom_bodyid[contact.geom1]).name
        b = model.body(model.geom_bodyid[contact.geom2]).name
        if a in CUBES and b in CUBES and contact.dist < 1e-3:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _region(*, env):
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
        BOX_DROP_X_MARGINS,
        BOX_DROP_Y_MARGIN,
    )

    (x0, y0, x1, y1), _ = env.backend().core().box_interior()
    near, far = BOX_DROP_X_MARGINS
    return x0 + near, y0 + BOX_DROP_Y_MARGIN, x1 - far, y1 - BOX_DROP_Y_MARGIN


def test_an_empty_box_takes_its_first_cube_at_the_drop_regions_centre(*, env):
    env.reset_to_seed(seed=0)
    x0, y0, x1, y1 = _region(env=env)
    first = _kinder(env=env).box_drop_points()[0]
    assert np.allclose(first, ((x0 + x1) / 2, (y0 + y1) / 2), atol=1e-9)


def test_each_drop_point_is_as_far_as_the_region_allows_from_the_cubes_in_the_box(*, env):
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import BOX_DROP_TARGET_SPACING

    env.reset_to_seed(seed=0)
    core = env.backend().core()
    x0, y0, x1, y1 = _region(env=env)
    kinder = _kinder(env=env)
    for cube in CUBES:
        points = kinder.box_drop_points()
        assert points == kinder.box_drop_points()  # deterministic
        for x, y in points:
            assert x0 - 1e-9 <= x <= x1 + 1e-9
            assert y0 - 1e-9 <= y <= y1 + 1e-9
        for i, (ax, ay) in enumerate(points):
            for bx, by in points[i + 1 :]:
                assert np.hypot(ax - bx, ay - by) >= BOX_DROP_TARGET_SPACING - 1e-9
        _put_in_box(env=env, cube=cube, x=points[0][0], y=points[0][1])
    positions = [core.get_object(c).get_pose()[0][:2] for c in CUBES]
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(positions) for b in positions[i + 1 :]]
    # The region's centre and its four corners: 8 cm apart at the closest.
    assert min(gaps) > 0.075, gaps


def test_five_cubes_placed_in_turn_lie_apart_and_can_each_be_picked_again(*, env):
    """Seed 0, every cube picked off the shelf and placed in the box in turn. Dropped
    at the box's centre (before) they ended with 4 touching pairs, one cube on top of
    another, and 0/5 with a single grasp candidate. Now: no touching pair, 5/5."""
    from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks

    env.reset_to_seed(seed=0)
    for cube in CUBES:
        for skill in ("PickCubeFromShelf", "PlaceCubeInBox"):
            _act(env=env, skill=skill, cube=cube)
            assert env.last_skill_error() is None, (skill, cube, env.last_skill_error())
    atoms = env.get_current_state().abstract_atoms
    assert all(("InBox", (c, "box_0")) in atoms for c in CUBES)
    assert _touching_pairs(env=env) == []
    kinder = _kinder(env=env)
    pickable = [c for c in CUBES if next(FloorPicks(kinder=kinder).candidates(cube=c), None)]
    assert pickable == CUBES


def test_the_end_cube_of_a_touching_row_has_an_off_centre_grasp(*, env, states):
    """Case 1's row, cube_4 | cube_0 | cube_1 | cube_3 | cube_2 along x: 0/576 centred
    candidates for every cube. The diagnosis found 2 for end cube cube_4 with the hand
    shifted 6 mm away from cube_0."""
    from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks

    StateCapture.restore(env=env, state=states["refusal-case1-ees0-cube-row-in-box"])
    floor = FloorPicks(kinder=_kinder(env=env))
    grasp = next(floor.candidates(cube="cube_4"), None)
    assert grasp is not None
    centre = env.controllers()._skills.cube_position(cube="cube_4")  # noqa: SLF001
    shift = float(np.linalg.norm(grasp.hand[:2, 3] - centre[:2]))
    assert shift > 0.005  # an off-centre grasp, not a centred one


@pytest.mark.parametrize(("cube", "shifted"), [("cube_4", True), ("cube_1", False)])
def test_only_a_cube_with_a_neighbour_on_one_side_gets_off_centre_grasps(
    *, env, states, cube, shifted
):
    """cube_4 ends the row; cube_1 has a neighbour on both sides, so no shift frees it."""
    from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FLOOR_PICK_LEANS_DEG, FloorPicks

    StateCapture.restore(env=env, state=states["refusal-case1-ees0-cube-row-in-box"])
    kinder = _kinder(env=env)
    centre = kinder.shelf.cube_position(cube=cube)
    hands = FloorPicks(kinder=kinder).hands(cube=cube, centre=centre)
    centred = 4 * len(FLOOR_PICK_LEANS_DEG)  # four faces per lean
    assert (len(hands) > centred) == shifted


def test_the_end_cube_of_a_touching_row_is_returned_to_the_shelf(*, env, states):
    StateCapture.restore(env=env, state=states["refusal-case1-ees0-cube-row-in-box"])
    atoms = _act(env=env, skill="ReturnCubeToShelfFromBox", cube="cube_4")
    assert ("OnShelf", ("cube_4",)) in atoms, env.last_skill_error()
