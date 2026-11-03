import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.stroke import plan_stroke

GEOMETRY = {
    "rack_back": 1.6275,
    "rack_front": 1.3725,
    "rack_center_y": 0.0,
    "rack_top": 0.925,
    "blade_width": 0.25,
    "tray_back": 1.3625,
}


def _food(*, x, y, half=(0.015, 0.0225)):
    return np.array([
        [x + sx * half[0], y + sy * half[1], 0.94] for sx in (-1, 1) for sy in (-1, 1)
    ])


def _blade_x_at(*, plan, y):
    return plan.start[0] + (y - plan.start[1]) * plan.axis[0] / plan.axis[1]


@pytest.mark.parametrize("dy", [-0.04, 0.0, 0.04])
@pytest.mark.parametrize("phi", [-30.0, 0.0, 15.0])
def test_travel_is_always_straight_toward_the_plate(*, dy, phi):
    plan = plan_stroke(
        food_corners=np.stack([_food(x=1.50, y=0.0)]), dy=dy, phi_deg=phi, **GEOMETRY
    )
    np.testing.assert_allclose(plan.direction, [-1.0, 0.0, 0.0])
    assert plan.start[1] == pytest.approx(dy)
    assert plan.end[1] == pytest.approx(dy)
    assert plan.start[2] == pytest.approx(0.926)


@pytest.mark.parametrize("phi", [-30.0, 0.0, 30.0])
def test_plow_angle_yaws_the_blade_relative_to_travel(*, phi):
    plan = plan_stroke(
        food_corners=np.stack([_food(x=1.50, y=0.0)]), dy=0.0, phi_deg=phi, **GEOMETRY
    )
    radians = np.deg2rad(phi)
    np.testing.assert_allclose(plan.axis, [np.sin(radians), np.cos(radians), 0.0], atol=1e-12)
    assert float(plan.axis @ plan.direction) == pytest.approx(-np.sin(radians))


@pytest.mark.parametrize("phi", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("dy", [-0.04, 0.04])
def test_start_is_behind_every_food_in_the_swept_band(*, phi, dy):
    foods = [_food(x=1.556, y=-0.092), _food(x=1.556, y=0.092), _food(x=1.50, y=0.0)]
    plan = plan_stroke(food_corners=np.stack(foods), dy=dy, phi_deg=phi, **GEOMETRY)
    for corners in foods:
        for corner in corners:
            assert _blade_x_at(plan=plan, y=corner[1]) - corner[0] >= 0.035 - 1e-9


def test_food_outside_the_band_is_left_alone_not_refused():
    far = _food(x=1.60, y=0.30)
    plan = plan_stroke(
        food_corners=np.stack([_food(x=1.45, y=0.0), far]), dy=0.0, phi_deg=0.0, **GEOMETRY
    )
    assert plan.start[0] == pytest.approx(1.6275 - 0.02)


def test_no_food_in_band_still_plans_an_unretreated_stroke():
    plan = plan_stroke(
        food_corners=np.stack([_food(x=1.60, y=0.30)]), dy=0.0, phi_deg=20.0, **GEOMETRY
    )
    assert plan.start[0] == pytest.approx(1.6275 - 0.02)
    assert plan.length > 0


def test_food_partly_inside_the_band_counts():
    edge = _food(x=1.60, y=0.125 + 0.02)
    plan = plan_stroke(food_corners=np.stack([edge]), dy=0.0, phi_deg=0.0, **GEOMETRY)
    assert plan.start[0] >= edge[:, 0].max() + 0.035 - 1e-9


@pytest.mark.parametrize("phi", [-20.0, 0.0, 20.0])
@pytest.mark.parametrize("dy", [-0.04, 0.0, 0.04])
def test_entire_blade_passes_the_plate_near_edge_by_release_clearance(*, phi, dy):
    plan = plan_stroke(
        food_corners=np.stack([_food(x=1.50, y=0.0)]),
        dy=dy,
        phi_deg=phi,
        release_clearance=0.055,
        **GEOMETRY,
    )
    edges = np.stack([plan.end + 0.125 * plan.axis, plan.end - 0.125 * plan.axis])
    assert edges[:, 0].max() == pytest.approx(1.3625 - 0.055)


@pytest.mark.parametrize(
    "overrides",
    [
        {"phi_deg": float("nan")},
        {"phi_deg": -81.0},
        {"phi_deg": 81.0},
        {"dy": float("inf")},
        {"tray_back": 1.40},
        {"blade_width": 0.0},
    ],
)
def test_invalid_geometry_rejected(*, overrides):
    arguments = {**GEOMETRY, "dy": 0.0, "phi_deg": 0.0, **overrides}
    with pytest.raises(ValueError):
        plan_stroke(food_corners=np.stack([_food(x=1.5, y=0.0)]), **arguments)


@pytest.mark.parametrize("corners", [np.zeros((0, 4, 3)), np.zeros((4, 3)), np.zeros((1, 4, 2))])
def test_food_corners_must_be_grouped_per_food(*, corners):
    with pytest.raises(ValueError):
        plan_stroke(food_corners=corners, dy=0.0, phi_deg=0.0, **GEOMETRY)
