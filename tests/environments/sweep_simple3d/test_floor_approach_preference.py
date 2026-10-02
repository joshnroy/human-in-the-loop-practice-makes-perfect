"""Joint reserve ranks checked floor paths without changing validity constraints."""

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorApproachPreference


def test_later_checked_candidate_with_margin_replaces_first_valid_fallback() -> None:
    preference = FloorApproachPreference()
    first = object()
    preferred = object()
    assert not preference.consider(candidate=first, margin=.0001)
    assert preference.candidate is first
    assert preference.consider(candidate=preferred, margin=.017)
    assert preference.candidate is preferred
    assert preference.margin == .017


def test_no_preferred_candidate_retains_original_checked_fallback() -> None:
    preference = FloorApproachPreference()
    first = object()
    preference.consider(candidate=first, margin=.0001)
    assert not preference.consider(candidate=object(), margin=.009)
    assert preference.candidate is first
    assert preference.margin == .0001
    assert not preference.preferred


def test_native_margin_ignores_continuous_joints_but_not_elbow_limit() -> None:
    limits = np.array([[-np.inf, np.inf], [-2.24, 2.24], [-np.inf, np.inf],
                       [-2.57, 2.57], [-np.inf, np.inf], [-2.09, 2.09], [-np.inf, np.inf]])
    arm = np.array([100., 2.2399, -100., -1., 200., 1., -200.])
    before = limits.copy()
    assert FloorApproachPreference.native_margin(arm=arm, limits=limits) == pytest.approx(.0001)
    arm[1] = 2.22
    assert FloorApproachPreference.native_margin(arm=arm, limits=limits) == pytest.approx(.02)
    np.testing.assert_array_equal(limits, before)


@pytest.mark.parametrize("margin", [-.00009, np.nan])
def test_invalid_endpoint_cannot_be_accepted_as_fallback(*, margin: float) -> None:
    preference = FloorApproachPreference()
    with pytest.raises(ValueError):
        preference.consider(candidate=object(), margin=margin)
    assert preference.candidate is None
