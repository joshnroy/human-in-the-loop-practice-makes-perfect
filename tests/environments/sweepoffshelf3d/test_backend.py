"""SweepOffShelf3D's live classifier and human resets, against the real scene."""

import importlib.util

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
    | {("WiperHome", ("wiper_0",)), ("HandEmpty", ("robot",))}
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
    assert ("WiperOnFloor", ("wiper_0",)) in backend.abstract_atoms()
    backend.reset_wiper()
    _settle(backend=backend, steps=500)
    assert ("WiperHome", ("wiper_0",)) in backend.abstract_atoms()


@pytest.mark.parametrize("left_behind", [0, 2, 5])
def test_reset_cubes_returns_five_minus_k_cubes_to_the_shelf(*, left_behind):
    backend = SweepOffShelf3DBackend(scene_bg=False, left_behind=left_behind)
    try:
        backend.reset(seed=0)
        (x0, y0, x1, y1), _ = backend.core().box_interior()
        for i in range(5):
            _place(
                backend=backend,
                name=f"cube_{i}",
                position=[(x0 + x1) / 2, y0 + 0.06 + 0.08 * i, 0.3],
            )
        _settle(backend=backend)
        moved = backend.reset_cubes()
        _settle(backend=backend, steps=500)
        atoms = backend.abstract_atoms()
        assert len(moved) == 5 - left_behind
        on_shelf = {objects[0] for name, objects in atoms if name == "OnShelf"}
        assert on_shelf == set(moved)
    finally:
        backend.close()
