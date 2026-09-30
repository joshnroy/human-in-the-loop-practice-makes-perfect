"""Clearance lead-in cannot enlarge the measured loaded stroke budget."""

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import ContactTravelBudget


def test_near_miss_clearance_allows_reaching_selected_cube() -> None:
    travel = ContactTravelBudget(length=.5, stroke=.10, step=.003, behind=.06112)
    targets = travel.targets()
    # Recorded narrow anchor was 261.12mm behind a cube; blade half-length150mm.
    first_contact = .26112 - .15 - .01
    assert targets[-1] == pytest.approx(.16112)
    assert targets[-1] > first_contact > .10
    assert not travel.observe(projection=first_contact, loaded=True)
    assert travel.onset == first_contact


def test_early_contact_clips_every_later_command_and_stops_at_loaded_limit() -> None:
    travel = ContactTravelBudget(length=.5, stroke=.10, step=.003, behind=.06112)
    assert not travel.observe(projection=.006, loaded=True)
    # Contact loss does not reset the origin or grant a second loaded allowance.
    assert not travel.observe(projection=.05, loaded=False)
    assert max(travel.target(proposed=float(x)) for x in travel.targets()) == pytest.approx(.106)
    assert travel.observe(projection=.107, loaded=False)
    assert travel.onset == .006


def test_absent_contact_is_bounded_and_zero_behind_preserves_original_targets() -> None:
    travel = ContactTravelBudget(length=.02, stroke=.10, step=.003, behind=.3)
    assert travel.targets()[-1] == pytest.approx(.12)
    for target in travel.targets():
        assert not travel.observe(projection=float(target), loaded=False)
    original = ContactTravelBudget(length=.5, stroke=.10, step=.003, behind=0)
    np.testing.assert_array_equal(original.targets(), np.arange(.003, .103, .003))


@pytest.mark.parametrize("behind", [-.01, np.nan, np.inf])
def test_invalid_clearance_never_enters_motion(*, behind: float) -> None:
    with pytest.raises(ValueError):
        ContactTravelBudget(length=.5, stroke=.1, step=.003, behind=behind)
