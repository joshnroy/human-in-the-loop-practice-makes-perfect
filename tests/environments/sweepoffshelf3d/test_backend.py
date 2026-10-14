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
