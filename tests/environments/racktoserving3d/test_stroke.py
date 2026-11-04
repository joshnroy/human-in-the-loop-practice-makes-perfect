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
FLOOR = 0.921


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


def _squeeze(**overrides):
    arguments = {"s": 0.07, "blade_width": 0.25, **RACK, **PLATE}
    return Planner.squeeze(**{**arguments, **overrides})


def _gather(**overrides):
    arguments = {
        "food_corners": SEED0,
        "delta": 0.0,
        "food_width": 0.045,
        "blade_width": 0.25,
        **RACK,
        **PLATE,
    }
    return Planner.gather(**{**arguments, **overrides})


def _sweep(**overrides):
    arguments = {"d": 0.10, "plate_clear_z": FLOOR, "blade_width": 0.25, **RACK, **PLATE}
    return Planner.sweep(**{**arguments, **overrides})


def test_squeeze_is_two_passes_toward_the_plate_with_the_blade_spanning_y():
    passes = _squeeze()
    assert len(passes) == 2
    for stroke in passes:
        np.testing.assert_allclose(stroke.direction, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(stroke.axis), [0.0, 1.0, 0.0])
        assert stroke.start[2] == pytest.approx(0.928)
        assert stroke.end[2] == pytest.approx(0.928)


@pytest.mark.parametrize("centre_y", [0.0, 0.02])
def test_squeeze_passes_are_centred_so_the_narrow_blade_covers_each_rack_side(*, centre_y):
    first, second = _squeeze(plate_center_y=centre_y)
    half = 0.375 / 2 - 0.25 / 2
    assert first.start[1] == pytest.approx(centre_y + half)
    assert second.start[1] == pytest.approx(centre_y - half)
    for stroke in (first, second):
        assert stroke.end[1] == pytest.approx(stroke.start[1])


@pytest.mark.parametrize("s", [0.03, 0.07, 0.14])
def test_squeeze_starts_behind_the_rack_and_stops_s_behind_its_front_edge(*, s):
    for stroke in _squeeze(s=s):
        assert stroke.start[0] == pytest.approx(1.6275 + 0.03)
        assert stroke.end[0] == pytest.approx(1.3725 + s)


def test_gather_pushes_from_plus_y_then_minus_y_with_the_blade_spanning_x():
    first, second = _gather()
    np.testing.assert_allclose(first.direction, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(second.direction, [0.0, 1.0, 0.0])
    for stroke in (first, second):
        np.testing.assert_allclose(np.abs(stroke.axis), [1.0, 0.0, 0.0])
        assert stroke.start[0] == pytest.approx((1.3725 + 1.6275) / 2)
        assert stroke.end[0] == pytest.approx(stroke.start[0])
        assert stroke.start[2] == pytest.approx(0.928)


@pytest.mark.parametrize("delta", [-0.03, 0.0, 0.015, 0.045])
@pytest.mark.parametrize("centre_y", [0.0, 0.02])
def test_each_gather_stops_one_food_width_minus_delta_from_the_centre_line(*, delta, centre_y):
    first, second = _gather(delta=delta, plate_center_y=centre_y)
    assert first.end[1] == pytest.approx(centre_y + 0.045 - delta)
    assert second.end[1] == pytest.approx(centre_y - 0.045 + delta)


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
    np.testing.assert_allclose(np.abs(plan.axis), [0.0, 1.0, 0.0])
    assert plan.start[1] == pytest.approx(0.02)
    assert plan.end[1] == pytest.approx(0.02)
    assert plan.start[0] == pytest.approx(1.6275 + 0.03)
    assert plan.start[2] == pytest.approx(0.928)


@pytest.mark.parametrize("d", [0.04, 0.10, 0.12])
def test_sweep_ends_d_before_the_plates_outer_near_edge(*, d):
    plan = _sweep(d=d)
    assert plan.end[0] == pytest.approx(1.2323 + 0.1302 - d)


def test_sweep_height_follows_rack_then_gap_then_plate():
    plan = _sweep(d=0.10)
    # On the rack it glides at rack height.
    assert plan.point(distance=0.0)[2] == pytest.approx(0.928)
    over_rack = 1.6275 + 0.03 - (1.3725 + 0.01)
    assert plan.point(distance=over_rack)[2] == pytest.approx(0.928)
    # Past the rack's front edge it descends, but never below the plate floor clearance.
    heights = [plan.point(distance=t)[2] for t in np.linspace(0, plan.length, 60)]
    assert np.all(np.diff(heights) <= 1e-12)
    assert min(heights) == pytest.approx(FLOOR)
    assert plan.end[2] == pytest.approx(FLOOR)
    np.testing.assert_allclose(plan.point(distance=plan.length), plan.end)


def test_straight_strokes_keep_constant_height():
    for stroke in (*_squeeze(), *_gather()):
        for t in np.linspace(0, stroke.length, 5):
            assert stroke.point(distance=t)[2] == pytest.approx(stroke.start[2])


@pytest.mark.parametrize(
    "overrides",
    [
        {"delta": float("nan")},
        {"blade_width": 0.0},
        {"plate_center_y": 0.30},
        {"rack_y_min": 0.2},
        {"food_width": 0.0},
    ],
)
def test_invalid_gather_geometry_rejected(*, overrides):
    with pytest.raises(ValueError):
        _gather(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"s": float("nan")},
        {"s": -0.01},
        {"s": 0.30},
        {"blade_width": 0.5},
        {"plate_center_y": 0.30},
    ],
)
def test_invalid_squeeze_geometry_rejected(*, overrides):
    with pytest.raises(ValueError):
        _squeeze(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"d": float("inf")},
        {"plate_radius": 0.0},
        {"plate_center_x": 1.40},
        {"d": 0.60},
        {"plate_clear_z": 0.95},
    ],
)
def test_invalid_sweep_geometry_rejected(*, overrides):
    with pytest.raises(ValueError):
        _sweep(**overrides)


@pytest.mark.parametrize("corners", [np.zeros((0, 4, 3)), np.zeros((4, 3)), np.zeros((1, 4, 2))])
def test_food_corners_must_be_grouped_per_food(*, corners):
    with pytest.raises(ValueError):
        _gather(food_corners=corners)
