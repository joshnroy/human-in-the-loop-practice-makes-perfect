"""Stale hold recovery preserves exact native bounds and does not clamp commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def hold_fixture(*, current: float):
    records = []
    observed = np.zeros(7)
    observed[1] = current
    primitive = SimpleNamespace(
        scene=SimpleNamespace(within_arm_limits=lambda *, arm: bool(abs(arm[1]) <= 2.24)),
        session=SimpleNamespace(arm=lambda: observed, ticks=5830,
                                _write=lambda *, record: records.append(record)),
    )
    return primitive, observed, records


def test_valid_cached_hold_remains_identical_even_if_observed_is_invalid() -> None:
    primitive, _, records = hold_fixture(current=2.2401)
    cached = np.zeros(7)
    cached[1] = 2.24
    assert FloorPrimitives.valid_contact_hold(primitive, cached=cached) is cached
    assert records == []


@pytest.mark.parametrize("cached_value,current", [(2.24008679, 2.23993802), (-2.2401, -2.2399)])
def test_invalid_cached_hold_uses_unmodified_valid_observation(
    *, cached_value: float, current: float
) -> None:
    primitive, observed, records = hold_fixture(current=current)
    cached = np.zeros(7)
    cached[1] = cached_value
    result = FloorPrimitives.valid_contact_hold(primitive, cached=cached)
    np.testing.assert_array_equal(result, observed)
    assert result is not observed
    assert cached[1] == cached_value
    assert records[0]["observed_within_native_limits"] is True


def test_both_invalid_fail_without_clamping() -> None:
    primitive, observed, records = hold_fixture(current=2.240001)
    cached = np.zeros(7)
    cached[1] = 2.24008679
    with pytest.raises(ExecutionError, match="violate native joint limits"):
        FloorPrimitives.valid_contact_hold(primitive, cached=cached)
    assert observed[1] == 2.240001
    assert records[0]["observed_within_native_limits"] is False
