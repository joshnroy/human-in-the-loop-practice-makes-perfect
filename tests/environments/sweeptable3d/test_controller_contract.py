"""Production angle/speed dispatch and single-attempt bookkeeping."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweeptable3d.controllers import SweepTable3DControllers


@pytest.mark.parametrize("fail", [False, True])
def test_sweep_uses_contact_controller_speed_and_consumes_attempt(*, monkeypatch, fail):
    from hitl_pmp.environments.sweeptable3d import contact

    stroke = Mock(metrics={"controller_revision": 3})
    if fail:
        stroke.run.side_effect = SkillFailed("blocked")
    factory = Mock(return_value=stroke)
    monkeypatch.setattr(contact, "ContactTableSweep", factory)
    backend = SimpleNamespace(begin_sweep=Mock(), finish_sweep=Mock(), table_center_y=lambda: 0.0)
    controller = SweepTable3DControllers.model_construct(backend=backend)
    controller._wiper = Mock()
    monkeypatch.setattr(
        SweepTable3DControllers, "_run_table", lambda self, **kwargs: kwargs["skill"]()
    )
    try:
        controller.sweep_table(angle=-27, speed=0.15)
    except SkillFailed:
        assert fail
    assert stroke.SPEED == 0.15
    stroke.run.assert_called_once_with(offset=0, yaw=-27)
    backend.begin_sweep.assert_called_once()
    backend.finish_sweep.assert_called_once_with(completed=not fail)


@pytest.mark.parametrize("angle,speed", [(46, 0.15), (0, 0), (0, 0.31), (float("nan"), 0.1)])
def test_invalid_parameters_never_start_motion(*, angle, speed):
    backend = SimpleNamespace(begin_sweep=Mock())
    controller = SweepTable3DControllers.model_construct(backend=backend)
    with pytest.raises(SkillFailed, match="parameters"):
        controller.sweep_table(angle=angle, speed=speed)
    backend.begin_sweep.assert_not_called()
