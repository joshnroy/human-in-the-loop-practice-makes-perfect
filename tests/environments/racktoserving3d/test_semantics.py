"""Offline physical-classification and clock/authorization contracts."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.racktoserving3d.semantics import (
    HUMAN_BATCH_RETURN,
    HUMAN_TOOL_RECOVERY,
    CircularSupport,
    FoodMeasurement,
    RackServingLedger,
    SupportRegion,
    classify_food,
)


def _region(*, top_z=1.0, x_min=-1.0, x_max=1.0):
    return SupportRegion(x_min=x_min, x_max=x_max, y_min=-1.0, y_max=1.0, top_z=top_z)


def _classify(*, held=False, centre_z=1.02, contacts=frozenset({"rack"}), **kwargs):
    return classify_food(
        food=FoodMeasurement(
            position=(0.0, 0.0, centre_z), half_extents=(0.1, 0.1, 0.02), held=held
        ),
        rack=_region(),
        tray=_region(),
        counter=_region(),
        support_contacts=contacts,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("contacts", "expected"),
    [
        (frozenset({"tray", "rack", "counter", "floor"}), "OnTray"),
        (frozenset({"rack", "counter", "floor"}), "OnRack"),
        (frozenset({"counter", "floor"}), "SpilledCounter"),
        (frozenset(), None),
    ],
)
def test_support_overlap_priority(*, contacts, expected):
    assert _classify(contacts=contacts, floor_z=1.0) == expected
    assert _classify(held=True, contacts=contacts, floor_z=1.0) == "HoldingFood"


def test_floor_spill_requires_actual_floor_support():
    assert _classify(centre_z=0.02, contacts=frozenset({"floor"})) == "SpilledFloor"
    assert _classify(centre_z=0.02, contacts=frozenset()) is None
    assert _classify(centre_z=0.5, contacts=frozenset()) is None
    assert _classify(centre_z=0.5, contacts=frozenset({"floor"})) is None


def test_live_bounds_and_full_footprint_not_centre_only():
    food = FoodMeasurement(position=(0.95, 0.0, 1.02), half_extents=(0.1, 0.1, 0.02))
    assert not _region().supports(food=food, tolerance=0.005)
    assert _region(x_max=1.1).supports(food=food, tolerance=0.005)
    assert not _region(top_z=0.8).supports(food=food, tolerance=0.005)


@pytest.mark.parametrize("elapsed", [0.0, 299.0, math.nextafter(300.0, 0.0)])
def test_cooling_before_boundary(*, elapsed):
    assert not RackServingLedger(elapsed_seconds=elapsed).ready


@pytest.mark.parametrize("elapsed", [300.0, math.nextafter(300.0, math.inf), 600.0])
def test_cooling_at_and_after_boundary(*, elapsed):
    assert RackServingLedger(elapsed_seconds=elapsed).ready


def test_wait_chunks_and_physical_time_share_one_monotone_clock():
    initial = RackServingLedger(elapsed_seconds=240.0)
    interrupted = initial.wait()
    assert interrupted.elapsed_seconds == 270.0
    assert not interrupted.ready
    complete = interrupted.advance(seconds=30.0)
    assert complete.ready
    assert initial.elapsed_seconds == 240.0


@pytest.mark.parametrize("intervention", [HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY])
@pytest.mark.parametrize("elapsed", [35.0, 300.0, 900.0])
def test_human_attempt_is_costed_without_restarting_cooling(*, intervention, elapsed):
    before = RackServingLedger(elapsed_seconds=elapsed, human_interruption_cost=7.5)
    after = before.charge_human(intervention=intervention)
    assert after.elapsed_seconds == elapsed
    assert after.ready == before.ready
    assert after.human_cost == 7.5
    assert after.human_interventions == 1
    assert before.human_cost == 0.0
    assert after.charge_human(intervention=intervention).human_cost == 15.0


@pytest.mark.parametrize("intervention", [HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY])
def test_deployment_rejection_precedes_backend_writes(*, intervention):
    ledger = RackServingLedger(mode="deployment", elapsed_seconds=300.0)
    writes = []
    with pytest.raises(PermissionError, match="training-only"):
        ledger = ledger.charge_human(intervention=intervention)
        writes.append("backend mutation")
    assert not writes
    assert ledger.elapsed_seconds == 300.0
    assert ledger.human_cost == 0.0
    assert ledger.human_interventions == 0


@pytest.mark.parametrize("cost", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_human_cost_must_be_positive_finite(*, cost):
    with pytest.raises(ValidationError):
        RackServingLedger(human_interruption_cost=cost)


@pytest.mark.parametrize("seconds", [-1.0, math.nan, math.inf, -math.inf])
def test_clock_rejects_invalid_increments(*, seconds):
    with pytest.raises(ValidationError):
        RackServingLedger().advance(seconds=seconds)


def test_invalid_measurements_are_rejected():
    with pytest.raises(ValidationError):
        FoodMeasurement(position=(math.nan, 0, 0), half_extents=(0.1, 0.1, 0.1))
    with pytest.raises(ValidationError):
        FoodMeasurement(position=(0, 0, 0), half_extents=(0.1, 0, 0.1))
    with pytest.raises(ValidationError):
        _region(x_min=1, x_max=-1)
    with pytest.raises(ValidationError):
        _classify(support_tolerance=math.inf)


def _plate(*, radius=0.13, top_z=1.0):
    return CircularSupport(center_x=0.0, center_y=0.0, radius=radius, top_z=top_z)


def test_round_plate_bounding_box_matches_circle():
    plate = _plate()
    assert (plate.x_min, plate.x_max, plate.y_min, plate.y_max) == (-0.13, 0.13, -0.13, 0.13)


def test_round_plate_rejects_footprint_inside_bounding_box_but_outside_circle():
    corner = FoodMeasurement(
        position=(0.1, 0.1, 1.02),
        half_extents=(0.01, 0.01, 0.02),
        footprint=((0.09, 0.09), (0.11, 0.09), (0.11, 0.11), (0.09, 0.11)),
    )
    assert _plate().x_min <= 0.09 and _plate().x_max >= 0.11
    assert not _plate().supports(food=corner, tolerance=0.005)


def test_round_plate_uses_actual_footprint_not_centre_only():
    footprint = ((0.115, 0.0), (0.135, 0.0), (0.125, 0.01), (0.125, -0.01))
    food = FoodMeasurement(
        position=(0.125, 0.0, 1.02), half_extents=(0.01, 0.01, 0.02), footprint=footprint
    )
    assert np.hypot(*food.position[:2]) < 0.13
    assert not _plate().supports(food=food, tolerance=0.005)
    assert _plate(radius=0.136).supports(food=food, tolerance=0.005)


def test_round_plate_without_footprint_falls_back_to_conservative_box_corners():
    food = FoodMeasurement(position=(0.09, 0.09, 1.02), half_extents=(1e-4, 1e-4, 0.02))
    assert _plate().supports(food=food, tolerance=0.005)
    boxed = FoodMeasurement(position=(0.085, 0.085, 1.02), half_extents=(0.01, 0.01, 0.02))
    assert not _plate().supports(food=boxed, tolerance=0.005)


def test_round_plate_still_requires_resting_height():
    food = FoodMeasurement(position=(0.0, 0.0, 1.05), half_extents=(0.01, 0.01, 0.02))
    assert not _plate().supports(food=food, tolerance=0.005)


def test_classify_food_scores_on_round_plate():
    food = FoodMeasurement(
        position=(0.12, 0.0, 1.02),
        half_extents=(0.02, 0.02, 0.02),
        footprint=((0.10, -0.02), (0.14, -0.02), (0.14, 0.02), (0.10, 0.02)),
    )
    common = {
        "food": food,
        "rack": _region(x_min=0.2, x_max=0.6),
        "counter": _region(x_min=-2, x_max=2),
        "support_contacts": frozenset({"tray", "counter"}),
    }
    assert classify_food(tray=_plate(), **common) == "SpilledCounter"
    assert classify_food(tray=_plate(radius=0.15), **common) == "OnTray"


@pytest.mark.parametrize(
    ("points", "margin", "expected"),
    [
        (((0.0, 0.0), (0.12, 0.0)), 0.0, True),
        (((0.0, 0.0), (0.12, 0.0)), 0.02, False),
        (((0.092, 0.092),), 0.0, False),
    ],
)
def test_contains_xy_for_both_support_shapes(*, points, margin, expected):
    assert _plate().contains_xy(points=points, margin=margin) is expected
    box = SupportRegion(x_min=-0.13, x_max=0.13, y_min=-0.13, y_max=0.13, top_z=1.0)
    assert box.contains_xy(points=points, margin=margin) is all(
        abs(x) <= 0.13 - margin and abs(y) <= 0.13 - margin for x, y in points
    )


def test_round_plate_rejects_nonpositive_radius():
    with pytest.raises(ValidationError):
        CircularSupport(center_x=0.0, center_y=0.0, radius=0.0, top_z=1.0)
