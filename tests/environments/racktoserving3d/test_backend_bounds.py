"""World AABB math from measured transforms; no native geometry or simulator."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.semantics import (
    CircularSupport,
    RectangularSupport,
    SupportRegion,
)


def _backend(*, monkeypatch, kind, size, rotation, center, config=None):
    model = SimpleNamespace(geom_type=np.array([kind]), geom_size=np.array([size]))
    data = SimpleNamespace(geom_xmat=np.array([rotation.reshape(-1)]), geom_xpos=np.array([center]))
    monkeypatch.setattr(RackServingBackend, "model", lambda self: model)
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    monkeypatch.setattr(RackServingBackend, "body_geoms", lambda self, **kwargs: {0})
    return RackServingBackend() if config is None else RackServingBackend(config=config)


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
    elif kind == 5:
        extent = np.array([size[0], size[0], size[1]])
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


@pytest.mark.parametrize("angles", [(0, 0, 0), (45, 0, 0), (41, 23, 17), (10, 80, 5)])
def test_ellipsoid_footprint_encloses_its_exact_horizontal_shadow(*, monkeypatch, angles):
    rotation = Rotation.from_euler("zyx", angles, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    center = np.array([1.2, -0.04, 0.94])
    backend = _backend(monkeypatch=monkeypatch, kind=4, size=size, rotation=rotation, center=center)
    footprint = backend.footprint(name="cube_0")
    assert footprint.ndim == 2 and footprint.shape[1] == 2
    shadow = (rotation @ np.diag(size**2) @ rotation.T)[:2, :2]
    for theta in np.linspace(0, 2 * np.pi, 361):
        direction = np.array([np.cos(theta), np.sin(theta)])
        exact = center[:2] @ direction + np.sqrt(direction @ shadow @ direction)
        reach = float(np.max(footprint @ direction))
        assert exact - 1e-12 <= reach <= exact + 1e-4


def test_box_footprint_is_its_horizontal_corners(*, monkeypatch):
    rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    backend = _backend(
        monkeypatch=monkeypatch, kind=6, size=size, rotation=rotation, center=np.zeros(3)
    )
    np.testing.assert_allclose(
        backend.footprint(name="rack_0"), backend.corners(name="rack_0")[:, :2]
    )


def _plate_backend(*, monkeypatch, tilt=0.0, serving_target="round_plate"):
    rotation = Rotation.from_euler("x", tilt, degrees=True).as_matrix()
    return _backend(
        monkeypatch=monkeypatch,
        kind=5,
        size=np.array([0.13, 0.006, 0.0]),
        rotation=rotation,
        center=np.array([1.2325, 0.01, 0.906]),
        config=RackServingConfig(serving_target=serving_target),
    )


def test_round_plate_support_is_the_live_cylinder_circle(*, monkeypatch):
    plate = _plate_backend(monkeypatch=monkeypatch).support(name="box_0")
    assert isinstance(plate, CircularSupport)
    assert (plate.center_x, plate.center_y, plate.radius) == pytest.approx((1.2325, 0.01, 0.13))
    assert plate.top_z == pytest.approx(0.912)
    assert plate.x_max == pytest.approx(1.3625)


def test_tilted_round_plate_support_shrinks_conservatively(*, monkeypatch):
    plate = _plate_backend(monkeypatch=monkeypatch, tilt=10).support(name="box_0")
    assert plate.radius == pytest.approx(0.13 * np.cos(np.deg2rad(10)))


def test_slate_target_keeps_rectangular_support(*, monkeypatch):
    rotation = np.eye(3)
    backend = _backend(
        monkeypatch=monkeypatch,
        kind=6,
        size=np.array([0.15, 0.1875, 0.006]),
        rotation=rotation,
        center=np.array([1.2125, 0, 0.906]),
        config=RackServingConfig(serving_target="slate"),
    )
    assert isinstance(backend.support(name="box_0"), SupportRegion)


@pytest.mark.parametrize(("food_y", "expected"), [(0.0, "OnTray"), (0.10, None)])
def test_food_inside_plate_box_but_outside_circle_is_not_on_tray(*, monkeypatch, food_y, expected):
    size = np.array([0.015, 0.0225, 0.0125])
    backend = _backend(
        monkeypatch=monkeypatch,
        kind=4,
        size=size,
        rotation=np.eye(3),
        center=np.array([1.2325 + 0.095, food_y, 0.9245]),
        config=RackServingConfig(),
    )
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: False)
    monkeypatch.setattr(RackServingBackend, "supporting_bodies", lambda self, **kwargs: {"box_0"})
    plate = CircularSupport(center_x=1.2325, center_y=0.0, radius=0.13, top_z=0.912)
    rack = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, *, name: plate if name == "box_0" else rack,
    )
    assert plate.x_min < 1.2325 + 0.095 - 0.015 and plate.x_max > 1.2325 + 0.095 + 0.015
    assert backend.location(name="cube_0") == expected


PLATTER_GEOMS = (
    "box_0_platter_floor",
    "box_0_lip_front",
    "box_0_lip_back",
    "box_0_lip_left",
    "box_0_lip_right",
)


def _platter_backend(*, monkeypatch, yaw=0.0, tilt=0.0, centre=(1.2125, 0.0)):
    """Five live box geoms of a 0.30 x 0.45 platter: 6 mm floor and 8 mm x 12 mm lip."""
    rotation = Rotation.from_euler("zx", [yaw, tilt], degrees=True).as_matrix()
    local = {
        "box_0_platter_floor": ((0, 0, 0.003), (0.15, 0.225, 0.003)),
        "box_0_lip_front": ((-0.146, 0, 0.012), (0.004, 0.217, 0.006)),
        "box_0_lip_back": ((0.146, 0, 0.012), (0.004, 0.217, 0.006)),
        "box_0_lip_left": ((0, -0.221, 0.012), (0.15, 0.004, 0.006)),
        "box_0_lip_right": ((0, 0.221, 0.012), (0.15, 0.004, 0.006)),
    }
    origin = np.array([*centre, 0.9])
    model = SimpleNamespace(
        geom_type=np.full(5, 6),
        geom_size=np.array([local[name][1] for name in PLATTER_GEOMS]),
        geom=lambda name: SimpleNamespace(id=PLATTER_GEOMS.index(name)),
    )
    data = SimpleNamespace(
        geom_xmat=np.array([rotation.reshape(-1)] * 5),
        geom_xpos=np.array([
            origin + rotation @ np.array(local[name][0]) for name in PLATTER_GEOMS
        ]),
    )
    monkeypatch.setattr(RackServingBackend, "model", lambda self: model)
    monkeypatch.setattr(RackServingBackend, "data", lambda self: data)
    monkeypatch.setattr(RackServingBackend, "body_geoms", lambda self, **kwargs: set(range(5)))
    return RackServingBackend(config=RackServingConfig(serving_target="lipped_platter"))


def test_platter_support_is_the_live_lip_inner_rectangle_at_floor_height(*, monkeypatch):
    platter = _platter_backend(monkeypatch=monkeypatch).support(name="box_0")
    assert isinstance(platter, RectangularSupport)
    assert (platter.center_x, platter.center_y) == pytest.approx((1.2125, 0.0))
    assert (platter.half_x, platter.half_y) == pytest.approx((0.142, 0.217))
    assert platter.top_z == pytest.approx(0.906)
    assert platter.x_max == pytest.approx(1.2125 + 0.142)


def test_yawed_platter_support_rotates_with_the_body(*, monkeypatch):
    platter = _platter_backend(monkeypatch=monkeypatch, yaw=20).support(name="box_0")
    assert platter.yaw == pytest.approx(np.deg2rad(20))
    assert (platter.half_x, platter.half_y) == pytest.approx((0.142, 0.217))


def test_tilted_platter_support_shrinks_conservatively(*, monkeypatch):
    platter = _platter_backend(monkeypatch=monkeypatch, tilt=10).support(name="box_0")
    assert platter.half_y == pytest.approx(0.217 * np.cos(np.deg2rad(10)))
    assert platter.half_x == pytest.approx(0.142 * np.cos(np.deg2rad(10)))


@pytest.mark.parametrize(
    ("food_x", "food_z", "expected"),
    [
        (1.2125, 0.9185, "OnTray"),
        (1.2125 + 0.142 - 0.010, 0.9185, None),
        (1.2125 + 0.146, 0.9305, None),
    ],
)
def test_platter_location_tests_the_food_footprint_against_the_lip(
    *, monkeypatch, food_x, food_z, expected
):
    platter = _platter_backend(monkeypatch=monkeypatch).support(name="box_0")
    food = _backend(
        monkeypatch=monkeypatch,
        kind=4,
        size=np.array([0.015, 0.0225, 0.0125]),
        rotation=np.eye(3),
        center=np.array([food_x, 0.0, food_z]),
        config=RackServingConfig(serving_target="lipped_platter"),
    )
    rack = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)
    monkeypatch.setattr(RackServingBackend, "held", lambda self, **kwargs: False)
    monkeypatch.setattr(RackServingBackend, "supporting_bodies", lambda self, **kwargs: {"box_0"})
    monkeypatch.setattr(
        RackServingBackend,
        "support",
        lambda self, *, name: platter if name == "box_0" else rack,
    )
    assert food.location(name="cube_0") == expected
