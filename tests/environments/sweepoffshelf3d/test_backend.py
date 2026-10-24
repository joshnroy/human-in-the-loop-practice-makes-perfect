"""SweepOffShelf3D's live classifier and human resets, against the real scene."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
    pytest.mark.xdist_group("sweepoffshelf3d_test_backend"),
]

_IDENTITY = [1.0, 0.0, 0.0, 0.0]
_START = frozenset(
    {("OnShelf", (f"cube_{i}",)) for i in range(5)}
    | {("WiperHome", ("wiper_0",)), ("HandEmpty", ("robot",)), ("AnyCubeOnShelf", ())}
)


def _settle(*, backend, steps=1500):
    for _ in range(steps):
        backend.sim().step()


def _place(*, backend, name, position):
    obj = backend.core().get_object(name)
    obj.set_pose(position, _IDENTITY)
    obj.set_velocity([0.0] * 3, [0.0] * 3)


@pytest.fixture(name="backend", scope="module")
def _backend():
    backend = SweepOffShelf3DBackend(scene_bg=False)
    try:
        yield backend
    finally:
        backend.close()


def test_the_start_state_is_every_cube_on_the_shelf_and_the_wiper_home(*, backend):
    backend.reset(seed=0)
    assert backend.abstract_atoms() == _START


def test_the_box_is_the_narrow_30_cm_box_centred_under_the_shelf_edge(*, backend):
    """KINDER's `SweepOffShelf3D-o5-narrow_box`: 30 cm along the shelf, 25 cm deep,
    20 cm tall, centred at y = 0, its far wall under the shelf's open front."""
    backend.reset(seed=0)
    core = backend.core()
    box = core.get_object("box_0")
    assert (box.length, box.width, box.height) == pytest.approx((0.25, 0.30, 0.20))
    (x0, y0, x1, y1), _ = core.box_interior()
    assert (x0, y0, x1, y1) == pytest.approx((1.12, -0.14, 1.35, 0.14), abs=1e-3)
    assert x0 < core.shelf_front_x() <= x1 + 0.03


def test_cubes_in_the_box_and_on_the_floor_classify(*, backend):
    backend.reset(seed=0)
    (x0, y0, x1, y1), _ = backend.core().box_interior()
    _place(backend=backend, name="cube_0", position=[(x0 + x1) / 2, (y0 + y1) / 2, 0.3])
    _place(backend=backend, name="cube_1", position=[x0 - 0.15, 0.0, 0.05])
    _settle(backend=backend)
    atoms = backend.abstract_atoms()
    assert ("InBox", ("cube_0", "box_0")) in atoms
    assert ("OnFloor", ("cube_1",)) in atoms
    assert ("OnShelf", ("cube_2",)) in atoms


def test_a_wiper_on_the_floor_is_not_home(*, backend):
    backend.reset(seed=0)
    _place(backend=backend, name="wiper_0", position=[0.6, 0.6, 0.05])
    _settle(backend=backend)
    assert {("WiperOnFloor", ("wiper_0",)), ("WiperAway", ("wiper_0",))} <= (
        backend.abstract_atoms()
    )
    backend.reset_wiper()
    _settle(backend=backend, steps=500)
    assert ("WiperHome", ("wiper_0",)) in backend.abstract_atoms()
    assert ("WiperAway", ("wiper_0",)) not in backend.abstract_atoms()


def _cubes_in_the_box(*, backend):
    (x0, y0, x1, y1), _ = backend.core().box_interior()
    for i in range(5):
        _place(
            backend=backend,
            name=f"cube_{i}",
            position=[(x0 + x1) / 2, y0 + 0.06 + 0.08 * i, 0.3],
        )
    _settle(backend=backend)


def test_a_successful_reset_cubes_returns_all_five_cubes_to_the_shelf():
    backend = SweepOffShelf3DBackend(scene_bg=False, reset_cubes_success_probability=1.0)
    try:
        backend.reset(seed=0)
        _cubes_in_the_box(backend=backend)
        assert ("AnyCubeOffShelf", ()) in backend.abstract_atoms()
        moved = backend.reset_cubes()
        _settle(backend=backend, steps=500)
        atoms = backend.abstract_atoms()
        assert moved == [f"cube_{i}" for i in range(5)]
        assert {objects[0] for name, objects in atoms if name == "OnShelf"} == set(moved)
        assert ("AnyCubeOffShelf", ()) not in atoms
    finally:
        backend.close()


def test_a_failed_reset_cubes_moves_nothing():
    backend = SweepOffShelf3DBackend(scene_bg=False, reset_cubes_success_probability=0.0)
    try:
        backend.reset(seed=0)
        _cubes_in_the_box(backend=backend)
        before = backend.features()
        atoms = backend.abstract_atoms()
        assert backend.reset_cubes() == []
        after = backend.features()
        assert backend.abstract_atoms() == atoms
        for name in backend.cube_names:
            assert after[name] == pytest.approx(before[name])
    finally:
        backend.close()


def _yaw_box(*, backend, yaw):
    box = backend.core().get_object("box_0")
    position, _ = box.get_pose()
    box.set_pose(list(position), [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    box.set_velocity([0.0] * 3, [0.0] * 3)
    backend.sim().forward()


def test_a_turned_box_still_scores_a_cube_inside_it_and_steps(*, backend):
    backend.reset(seed=0)
    _yaw_box(backend=backend, yaw=0.35)
    centre = np.asarray(backend.core().get_object("box_0").get_pose()[0], dtype=float)
    _place(backend=backend, name="cube_0", position=[centre[0], centre[1], 0.15])
    for _ in range(20):
        backend.env().step(np.zeros(11))
    atoms = backend.abstract_atoms()
    assert ("InBox", ("cube_0", "box_0")) in atoms
    assert backend.features()["box_0"].shape == (5,)


def test_a_turned_box_scores_by_its_own_walls_not_its_axis_aligned_bound(*, backend):
    backend.reset(seed=0)
    _yaw_box(backend=backend, yaw=np.pi / 4)
    core = backend.core()
    centre = np.asarray(core.get_object("box_0").get_pose()[0], dtype=float)
    box = core._objects_dict["box_0"]  # noqa: SLF001
    half_x = box.length / 2 - box.wall_thickness
    # Along the box's own x axis just inside its wall; and 0.2 m along world x, which
    # at 45 deg lies past the box's x walls though well inside its axis-aligned bound.
    inside = centre + (half_x - 0.03) * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    outside = centre + np.array([0.2, 0.0, 0.0])
    _place(backend=backend, name="cube_0", position=[inside[0], inside[1], centre[2] + 0.05])
    _place(backend=backend, name="cube_1", position=[outside[0], outside[1], centre[2] + 0.05])
    backend.sim().forward()
    assert core.cube_in_box("cube_0")
    assert not core.cube_in_box("cube_1")


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_an_axis_aligned_box_scores_exactly_as_upstream(*, backend, yaw):
    from kinder.envs.dynamic3d.sweep_off_shelf3d import ObjectCentricSweepOffShelf3DEnv

    backend.reset(seed=0)
    _yaw_box(backend=backend, yaw=yaw)
    core = backend.core()
    centre = np.asarray(core.get_object("box_0").get_pose()[0], dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(40):
        offset = rng.uniform(-0.3, 0.3, size=2)
        _place(backend=backend, name="cube_0", position=[*(centre[:2] + offset), 0.1])
        backend.sim().forward()
        assert core.cube_in_box("cube_0") == ObjectCentricSweepOffShelf3DEnv.cube_in_box(
            core, "cube_0"
        )
        (ours, rim), (theirs, upstream_rim) = (
            core.box_interior(),
            ObjectCentricSweepOffShelf3DEnv.box_interior(core),
        )
        assert ours == pytest.approx(theirs, abs=1e-9)
        assert rim == pytest.approx(upstream_rim, abs=1e-9)


@pytest.mark.parametrize("tilt_deg", [0.0, 3.0, 10.0])
def test_in_box_and_the_goal_match_upstream_for_a_tipped_box(*, backend, tilt_deg):
    """A cube inside a box tipped about its x axis: `InBox` and the goal answer as
    upstream's `cube_in_box` does, which holds nothing in a box tipped past 5 deg."""
    from kinder.envs.dynamic3d.sweep_off_shelf3d import ObjectCentricSweepOffShelf3DEnv
    from scipy.spatial.transform import Rotation

    backend.reset(seed=0)
    core = backend.core()
    box = core.get_object("box_0")
    centre = np.asarray(box.get_pose()[0], dtype=float)
    tilt = Rotation.from_euler("x", tilt_deg, degrees=True)
    x, y, z, w = tilt.as_quat()
    box.set_pose(list(centre), [w, x, y, z])
    box.set_velocity([0.0] * 3, [0.0] * 3)
    for i, name in enumerate(backend.cube_names):
        cube = core.get_object(name)
        offset = [-0.06 + 0.03 * i, 0.0, 0.05]
        cube.set_pose(list(tilt.apply(offset) + centre), [w, x, y, z])
        cube.set_velocity([0.0] * 3, [0.0] * 3)
    backend.sim().forward()
    upstream = ObjectCentricSweepOffShelf3DEnv.cube_in_box(core, "cube_0")
    assert upstream == (tilt_deg < 5.0)
    assert (("InBox", ("cube_0", "box_0")) in backend.abstract_atoms()) == upstream
    assert core._check_goals() == upstream  # noqa: SLF001


# --- where a cube can lodge beyond every robot skill's reach -------------------
#
# A cube is 2 cm on a side. Two openings in this scene are larger: the one above the
# box's far wall, below the cube shelf, into the cupboard's bottom compartment (a cube
# that goes over the far rim lands on the bottom board), and the box's 1 cm rims, on
# which a cube balances. Both classify `OnFloor`; no floor pick reaches either, and
# `reset_cubes` recovers both. The geometry is kept as it is unless the traps show up
# as a problem in practice; these tests pin it so a scene change is noticed.

_CUBE_EDGE = 0.02
_TRAPS = {
    "over_the_far_rim": ([1.355, 0.10, 0.235], [0.10, 0.0, 0.0]),
    "on_the_far_rim": ([1.3551, 0.0, 0.2105], [0.0, 0.0, 0.0]),
    "on_a_side_rim": ([1.235, 0.245, 0.2105], [0.0, 0.0, 0.0]),
}


def _far_wall_outer_x(*, backend):
    box = backend.core()._objects_dict["box_0"]  # noqa: SLF001
    return float(box.get_pose()[0][0]) + box.length / 2


def _cube_shelf_underside_z(*, backend):
    core = backend.core()
    board = core._shelf_geoms()[1]  # noqa: SLF001
    return float(core._world_corners(board)[:, 2].min())  # noqa: SLF001


def _release(*, backend, position, velocity):
    obj = backend.core().get_object("cube_0")
    obj.set_pose(position, _IDENTITY)
    obj.set_velocity(velocity, [0.0] * 3)
    backend.sim().forward()
    _settle(backend=backend, steps=3000)
    return np.asarray(backend.core().get_object("cube_0").get_pose()[0], dtype=float)


def test_a_cube_cannot_drop_between_the_box_and_the_cupboard(*, backend):
    backend.reset(seed=0)
    gap = backend.core().shelf_front_x() - _far_wall_outer_x(backend=backend)
    assert 0.0 < gap < _CUBE_EDGE


def test_the_opening_above_the_far_wall_is_wider_than_a_cube(*, backend):
    """66.7 mm at the 64ce45c pin, from the far rim up to the cube shelf's underside."""
    backend.reset(seed=0)
    _, rim = backend.core().box_interior()
    opening = _cube_shelf_underside_z(backend=backend) - rim
    assert opening == pytest.approx(0.0667, abs=1e-3)
    assert opening > _CUBE_EDGE


@pytest.mark.parametrize("trap", sorted(_TRAPS))
def test_a_trapped_cube_is_on_the_floor_to_the_planner_and_the_human_recovers_it(*, backend, trap):
    """Reproducible today: placed, a cube comes to rest on the bottom board or a rim,
    classifies `OnFloor`, and `reset_cubes` (p = 1 here) puts it back on the shelf."""
    backend.reset(seed=0)
    position, velocity = _TRAPS[trap]
    x, _, z = _release(backend=backend, position=position, velocity=velocity)
    if trap == "over_the_far_rim":
        assert x > backend.core().shelf_front_x()
        assert z == pytest.approx(backend.core().shelf_top_z(shelf=0) + 0.01, abs=2e-3)
    else:
        assert z == pytest.approx(backend.core().box_interior()[1] + 0.01, abs=2e-3)
    assert ("OnFloor", ("cube_0",)) in backend.abstract_atoms()
    assert backend.reset_cubes() == list(backend.cube_names)
    _settle(backend=backend, steps=500)
    assert ("OnShelf", ("cube_0",)) in backend.abstract_atoms()
