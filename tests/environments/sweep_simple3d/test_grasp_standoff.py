"""Floor insertion uses its validated default without changing drawer pickup."""

import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.physical.primitives import Primitives


def test_grasp_standoff_defaults_and_explicit_diagnostic_override() -> None:
    drawer = Primitives.model_construct()
    floor = FloorPrimitives.model_construct()
    assert drawer.wiper_grasp_standoff() == 0.035
    assert floor.wiper_grasp_standoff() == 0.020
    floor.diagnostic_grasp_standoff = 0.010
    assert floor.wiper_grasp_standoff() == 0.010
    assert drawer.wiper_grasp_standoff() == 0.035


@pytest.mark.parametrize("offset", [-0.001, 0.036, float("nan")])
def test_invalid_diagnostic_standoff_is_rejected(*, offset: float) -> None:
    floor = FloorPrimitives.model_construct(diagnostic_grasp_standoff=offset)
    with pytest.raises(ValueError, match="Diagnostic grasp standoff"):
        floor.wiper_grasp_standoff()
