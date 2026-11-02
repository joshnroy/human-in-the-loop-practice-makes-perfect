"""World AABB math from measured transforms; no native geometry or simulator."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.semantics import SupportRegion


def _backend(*, monkeypatch, kind, size, rotation, center):
    model = SimpleNamespace(geom_type=np.array([kind]), geom_size=np.array([size]))
    data = SimpleNamespace(geom_xmat=np.array([rotation.reshape(-1)]), geom_xpos=np.array([center]))
    monkeypatch.setattr(RackServingBackend, "model", lambda self: model)
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    monkeypatch.setattr(RackServingBackend, "body_geoms", lambda self, **kwargs: {0})
    return RackServingBackend()


@pytest.mark.parametrize("angles", [(45, 0, 0), (0, 37, 0), (41, 23, 17), (90, 0, 0)])
def test_ellipsoid_bounds_use_exact_world_extents_without_rotating_twice(*, monkeypatch, angles):
    rotation = Rotation.from_euler("zyx", angles, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    center = np.array([1.2, -0.04, 0.94])
    backend = _backend(monkeypatch=monkeypatch, kind=4, size=size, rotation=rotation, center=center)
    corners = backend.corners(name="cube_0")
    extent = np.sqrt(rotation**2 @ size**2)
    expected = np.array([center + np.array(sign) * extent for sign in product((-1, 1), repeat=3)])
    np.testing.assert_allclose(corners, expected, atol=1e-14)
    np.testing.assert_allclose(corners.min(axis=0), center - extent, atol=1e-14)
    np.testing.assert_allclose(corners.max(axis=0), center + extent, atol=1e-14)


@pytest.mark.parametrize("kind", [2, 3, 5, 6])
@pytest.mark.parametrize("angles", [(0, 0, 0), (41, 23, 17)])
def test_non_ellipsoid_bounds_are_unchanged(*, monkeypatch, kind, angles):
    rotation = Rotation.from_euler("zyx", angles, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    center = np.array([1.2, -0.04, 0.94])
    backend = _backend(
        monkeypatch=monkeypatch, kind=kind, size=size, rotation=rotation, center=center
    )
    if kind == 3:
        extent = np.array([size[0], size[0], size[0] + size[1]])
    elif kind == 2:
        extent = np.full(3, size[0])
    else:
        extent = size
    expected = np.array([
        center + np.array(sign) * extent @ rotation.T for sign in product((-1, 1), repeat=3)
    ])
    np.testing.assert_allclose(backend.corners(name="rack_0"), expected, atol=1e-14)


def test_rotated_ellipsoid_fully_on_tray_is_not_rejected_by_box_overestimate(*, monkeypatch):
    rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    center = np.array([1.2, 0.0, 0.9245])
    backend = _backend(monkeypatch=monkeypatch, kind=4, size=size, rotation=rotation, center=center)
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: False)
    monkeypatch.setattr(RackServingBackend, "supporting_bodies", lambda self, **kwargs: {"box_0"})
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, **kwargs: SupportRegion(
            x_min=1.18, x_max=1.22, y_min=-0.02, y_max=0.02, top_z=0.912
        ),
    )
    assert backend.location(name="cube_0") == "OnTray"


@pytest.mark.parametrize(
    "kind, tilt, separation, expected",
    [(0, 0, 0, "SpilledFloor"), (0, 90, 0, None), (6, 0, 0, None), (0, 0, 0.01, None)],
)
def test_background_floor_uses_contact_geometry_not_scene_body_name(
    *, monkeypatch, kind, tilt, separation, expected
):
    bodies = {"cube_0": SimpleNamespace(id=1), 2: SimpleNamespace(name="scene")}
    model = SimpleNamespace(
        body=lambda key: bodies[key],
        geom_bodyid=np.array([1, 2]),
        geom_type=np.array([4, kind]),
    )
    data = SimpleNamespace(
        xpos=np.array([[0, 0, 0], [0, 0, 0.0125], [0, 0, 0]]),
        geom_xmat=np.array([
            np.eye(3).ravel(),
            Rotation.from_euler("x", tilt, degrees=True).as_matrix().ravel(),
        ]),
        contact=[SimpleNamespace(geom1=0, geom2=1, dist=separation, pos=np.zeros(3))],
        ncon=1,
    )
    monkeypatch.setattr(RackServingBackend, "model", lambda self: model)
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: False)
    monkeypatch.setattr(
        RackServingBackend,
        "corners",
        lambda self, **kwargs: np.array([[-0.015, -0.0225, 0], [0.015, 0.0225, 0.025]]),
    )
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, **kwargs: SupportRegion(x_min=1, x_max=2, y_min=-0.5, y_max=0.5, top_z=0.9),
    )
    assert RackServingBackend().location(name="cube_0") == expected
