"""Cubes that ride the drawer's front: on top of its face, or on its handle."""

import numpy as np

from hitl_pmp.environments.sweep_drawer3d.session import CubePlaces


def _rides(*, x: float, y: float, z: float, drawer: float) -> bool:
    return CubePlaces.rides_drawer_front(position=np.array([x, y, z]), drawer_pos=drawer)


def test_a_cube_on_top_of_the_drawers_face_rides_it() -> None:
    """Seed 22 leaves two there, a centimetre from the countertop's edge."""
    assert _rides(x=0.8852, y=0.0498, z=0.4539, drawer=0.0003)
    assert _rides(x=0.8853, y=0.0245, z=0.4539, drawer=0.0003)


def test_a_cube_on_the_handle_rides_it() -> None:
    """Seed 1 leaves one there, a centimetre above the bar."""
    assert _rides(x=0.9429, y=0.0459, z=0.3379, drawer=0.0147)


def test_the_cube_is_followed_out_with_the_drawer() -> None:
    assert _rides(x=0.8852 + 0.25, y=0.05, z=0.4539, drawer=0.25)
    assert not _rides(x=0.8852, y=0.05, z=0.4539, drawer=0.25)


def test_cubes_on_the_counter_in_the_drawer_or_on_the_floor_do_not() -> None:
    assert not _rides(x=0.832, y=0.063, z=0.4698, drawer=0.0)
    assert not _rides(x=1.044, y=0.046, z=0.2373, drawer=0.25)
    assert not _rides(x=0.9118, y=0.0046, z=0.0099, drawer=0.0)


def test_a_cube_beside_the_drawer_on_a_neighbouring_face_does_not() -> None:
    """The island's other drawers have faces and handles of the same shape."""
    assert not _rides(x=0.885, y=0.60, z=0.4539, drawer=0.0)
