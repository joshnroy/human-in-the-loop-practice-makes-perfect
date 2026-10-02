"""Only fixed reverse .70 routes can use the checked .65 fallback."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import ExecutionError, FloorPrimitives


@pytest.mark.parametrize(
    "region,distance", [("sweep_region", 0.7), ("sweep_region", 0.55), ("blocks_init_region", 0.55)]
)
def test_other_stance_parameters_remain_exact(*, region: str, distance: float) -> None:
    primitive = SimpleNamespace(scene=None)
    original = (1.0, 2.0, 0.0)
    assert (
        FloorPrimitives.fixed_reverse_stance(
            primitive,
            region=region,
            distance=distance,
            wiper_start=np.zeros(2),
            bearing=0.0,
            original=original,
        )
        is original
    )


@pytest.mark.parametrize("original_clear", [True, False])
def test_fixed_reverse_checks_original_before_alternative(*, original_clear: bool) -> None:
    original = (1.0, 1.7, -np.pi / 2)
    requests, events = [], []

    def plan(*, target):
        requests.append(target)
        return [target] if target != original or original_clear else None

    primitive = SimpleNamespace(
        scene=SimpleNamespace(plan_base=plan),
        session=SimpleNamespace(ticks=9, _write=lambda **kw: events.append(kw)),
    )
    result = FloorPrimitives.fixed_reverse_stance(
        primitive,
        region="blocks_init_region",
        distance=0.7,
        wiper_start=np.array([1.0, 1.0]),
        bearing=np.pi / 2,
        original=original,
    )
    assert result == pytest.approx(original if original_clear else (1.0, 1.65, -np.pi / 2))
    assert requests[0] == original
    assert len(requests) == (1 if original_clear else 2)
    assert len(events) == (0 if original_clear else 1)
    primitive.scene.plan_base = lambda **kwargs: None
    with pytest.raises(ExecutionError, match="either fixed reverse stance"):
        FloorPrimitives.fixed_reverse_stance(
            primitive,
            region="blocks_init_region",
            distance=0.7,
            wiper_start=np.array([1.0, 1.0]),
            bearing=np.pi / 2,
            original=original,
        )
