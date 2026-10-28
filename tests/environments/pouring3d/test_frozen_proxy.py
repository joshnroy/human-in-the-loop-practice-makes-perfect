"""Freezing labels must not silently introduce a different collision model."""

import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourParameters
from hitl_pmp.environments.pouring3d.frozen_native import (
    ExtendedFeedbackPourParameters,
    FrozenShrimp,
)
from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp


def test_frozen_label_preserves_the_existing_rigid_body() -> None:
    options = {"size": [0.017, 0.018, 0.0055], "mass": 0.003}
    cooked = CookedShrimp(name="shrimp_0", options=options)
    frozen = FrozenShrimp(name="shrimp_0", options=options)
    assert ET.tostring(cooked.xml_element) == ET.tostring(frozen.xml_element)


def test_longer_budget_preserves_all_other_controls_and_old_validation() -> None:
    baseline = FeedbackPourParameters(tilt_degrees=101, amplitude_m=0.005, duration_seconds=20)
    extended = ExtendedFeedbackPourParameters(**baseline.model_dump())
    assert extended.model_dump() == baseline.model_dump()
    extended = ExtendedFeedbackPourParameters(duration_seconds=60)
    assert extended.duration_seconds == 60
    with pytest.raises(ValidationError):
        FeedbackPourParameters(duration_seconds=60)
    with pytest.raises(ValidationError):
        ExtendedFeedbackPourParameters(duration_seconds=61)
