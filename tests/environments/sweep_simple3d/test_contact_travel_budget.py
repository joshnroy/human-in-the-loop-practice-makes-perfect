"""Clearance lead-in cannot enlarge the measured loaded stroke budget."""

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import ContactTravelBudget


def test_near_miss_clearance_allows_reaching_selected_cube() -> None:
    travel = ContactTravelBudget(length=0.5, stroke=0.10, step=0.003, behind=0.06112)
    targets = travel.targets()
    # Recorded narrow anchor was 261.12mm behind a cube; blade half-length150mm.
    first_contact = 0.26112 - 0.15 - 0.01
    assert targets[-1] == pytest.approx(0.16112)
    assert targets[-1] > first_contact > 0.10
    assert not travel.observe(projection=first_contact, loaded=True)
    assert travel.onset == first_contact


def test_early_contact_clips_every_later_command_and_stops_at_loaded_limit() -> None:
    travel = ContactTravelBudget(length=0.5, stroke=0.10, step=0.003, behind=0.06112)
    assert not travel.observe(projection=0.006, loaded=True)
    # Contact loss does not reset the origin or grant a second loaded allowance.
    assert not travel.observe(projection=0.05, loaded=False)
    assert max(travel.target(proposed=float(x)) for x in travel.targets()) == pytest.approx(0.106)
    assert travel.observe(projection=0.107, loaded=False)
    assert travel.onset == 0.006


def test_absent_contact_is_bounded_and_zero_behind_preserves_original_targets() -> None:
    travel = ContactTravelBudget(length=0.02, stroke=0.10, step=0.003, behind=0.3)
    assert travel.targets()[-1] == pytest.approx(0.40)
    for target in travel.targets():
        assert not travel.observe(projection=float(target), loaded=False)
    original = ContactTravelBudget(length=0.5, stroke=0.10, step=0.003, behind=0)
    np.testing.assert_array_equal(original.targets(), np.arange(0.003, 0.103, 0.003))


@pytest.mark.parametrize("behind", [-0.01, np.nan, np.inf])
def test_invalid_clearance_never_enters_motion(*, behind: float) -> None:
    with pytest.raises(ValueError):
        ContactTravelBudget(length=0.5, stroke=0.1, step=0.003, behind=behind)


def test_recorded_late_reverse_gap_keeps_clearance_outside_remaining_travel() -> None:
    travel = ContactTravelBudget(length=0.241994, stroke=0.10, step=0.003, behind=0.310769)
    assert travel.targets()[-1] == pytest.approx(0.410769)
    # An early physical contact still limits all commands to 100mm loaded travel.
    assert not travel.observe(projection=0.02, loaded=True)
    assert max(travel.target(proposed=float(x)) for x in travel.targets()) == pytest.approx(0.12)
    assert travel.observe(projection=0.02 + 0.10, loaded=True)
