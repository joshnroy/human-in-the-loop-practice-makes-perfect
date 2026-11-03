import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner as Planner

RACK = {
    "rack_back": 1.6275,
    "rack_front": 1.3725,
    "rack_y_min": -0.1875,
    "rack_y_max": 0.1875,
    "rack_top": 0.925,
}
PLATE = {"plate_center_x": 1.2323, "plate_center_y": 0.0, "plate_radius": 0.1302}


def _food(*, x, y, half=(0.015, 0.0225)):
    return np.array([
        [x + sx * half[0], y + sy * half[1], 0.94] for sx in (-1, 1) for sy in (-1, 1)
    ])


SEED0 = np.stack([
    _food(x=1.444, y=-0.092),
    _food(x=1.444, y=0.092),
    _food(x=1.500, y=0.000),
    _food(x=1.556, y=-0.092),
    _food(x=1.556, y=0.092),
])


def _gather(**overrides):
    arguments = {"food_corners": SEED0, "delta": 0.0, "blade_width": 0.25, **RACK, **PLATE}
    return Planner.gather(**{**arguments, **overrides})


def _sweep(**overrides):
    arguments = {
        "food_corners": SEED0,
        "d": 0.10,
        "phi_deg": 0.0,
        "blade_width": 0.25,
        **RACK,
        **PLATE,
    }
    return Planner.sweep(**{**arguments, **overrides})


def test_gather_pushes_from_plus_y_then_minus_y_with_the_blade_spanning_x():
    first, second = _gather()
    np.testing.assert_allclose(first.direction, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(second.direction, [0.0, 1.0, 0.0])
    for stroke in (first, second):
        np.testing.assert_allclose(np.abs(stroke.axis), [1.0, 0.0, 0.0])
        # The blade is centred on the rack's depth, so its 0.25 m span covers the rack.
        assert stroke.start[0] == pytest.approx((1.3725 + 1.6275) / 2)
        assert stroke.end[0] == pytest.approx(stroke.start[0])
        assert stroke.start[2] == pytest.approx(0.926)


@pytest.mark.parametrize("delta", [-0.02, 0.0, 0.015, 0.03])
@pytest.mark.parametrize("centre_y", [0.0, 0.02])
def test_each_gather_stroke_stops_delta_past_the_plate_centre_line(*, delta, centre_y):
    first, second = _gather(delta=delta, plate_center_y=centre_y)
    assert first.end[1] == pytest.approx(centre_y - delta)
    assert second.end[1] == pytest.approx(centre_y + delta)


def test_gather_starts_outside_the_rack_and_behind_every_food():
    first, second = _gather()
    assert first.start[1] >= 0.1875 + 0.03 - 1e-9
    assert second.start[1] <= -0.1875 - 0.03 + 1e-9
    assert first.start[1] >= SEED0[:, :, 1].max() + 0.035 - 1e-9
    assert second.start[1] <= SEED0[:, :, 1].min() - 0.035 + 1e-9


def test_gather_retreats_further_for_food_overhanging_the_rack_side():
    overhang = np.stack([_food(x=1.5, y=0.19)])
    first, _ = _gather(food_corners=overhang)
    assert first.start[1] == pytest.approx(0.19 + 0.0225 + 0.035)


def test_sweep_travels_straight_toward_the_plate_on_its_centre_line():
    plan = _sweep(plate_center_y=0.02)
    np.testing.assert_allclose(plan.direction, [-1.0, 0.0, 0.0])
    assert plan.start[1] == pytest.approx(0.02)
    assert plan.end[1] == pytest.approx(0.02)
    assert plan.start[2] == pytest.approx(0.926)


@pytest.mark.parametrize("d", [0.05, 0.12, 0.25])
@pytest.mark.parametrize("phi", [-15.0, 0.0, 15.0])
def test_sweep_ends_with_the_blade_edge_d_past_the_plate_near_edge(*, d, phi):
    plan = _sweep(d=d, phi_deg=phi)
    assert plan.end[0] == pytest.approx(1.2323 + 0.1302 - d)


@pytest.mark.parametrize("phi", [-15.0, 0.0, 15.0])
def test_plow_angle_yaws_the_sweep_blade_relative_to_travel(*, phi):
    plan = _sweep(phi_deg=phi)
    radians = np.deg2rad(phi)
    np.testing.assert_allclose(plan.axis, [np.sin(radians), np.cos(radians), 0.0], atol=1e-12)


@pytest.mark.parametrize("phi", [-15.0, 0.0, 15.0])
def test_sweep_starts_behind_every_food_in_its_band(*, phi):
    gathered = np.stack([_food(x=1.45 + 0.031 * k, y=0.03) for k in range(5)])
    plan = _sweep(food_corners=gathered, phi_deg=phi)
    for corners in gathered:
        for corner in corners:
            blade_x = plan.start[0] + (corner[1] - plan.start[1]) * plan.axis[0] / plan.axis[1]
            assert blade_x - corner[0] >= 0.035 - 1e-9


@pytest.mark.parametrize(
    "overrides",
    [
        {"delta": float("nan")},
        {"blade_width": 0.0},
        {"plate_center_y": 0.30},
        {"rack_y_min": 0.2},
    ],
)
def test_invalid_gather_geometry_rejected(*, overrides):
    with pytest.raises(ValueError):
        _gather(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"d": float("inf")},
        {"phi_deg": 81.0},
        {"plate_radius": 0.0},
        {"plate_center_x": 1.40},
        {"d": 0.60},
    ],
)
def test_invalid_sweep_geometry_rejected(*, overrides):
    with pytest.raises(ValueError):
        _sweep(**overrides)


@pytest.mark.parametrize("corners", [np.zeros((0, 4, 3)), np.zeros((4, 3)), np.zeros((1, 4, 2))])
def test_food_corners_must_be_grouped_per_food(*, corners):
    with pytest.raises(ValueError):
        _gather(food_corners=corners)
    with pytest.raises(ValueError):
        _sweep(food_corners=corners)
