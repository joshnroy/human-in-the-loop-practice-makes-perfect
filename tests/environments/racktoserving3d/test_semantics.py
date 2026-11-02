"""Offline physical-classification and clock/authorization contracts."""

import math

import pytest
from pydantic import ValidationError

from hitl_pmp.environments.racktoserving3d.semantics import (
    HUMAN_BATCH_RETURN,
    HUMAN_TOOL_RECOVERY,
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
