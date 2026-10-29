"""Task-local capsule XML works independently of the installed KINDER patch."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweeptable3d.geometry import CapsuleWiperGeometry


def test_capsule_has_the_requested_total_width_height_and_mass():
    wiper = SimpleNamespace(
        name="wiper_0",
        joint_name="wiper_0_joint",
        head_height=0.02,
        head_length=0.20,
        handle_width=0.015,
        handle_height=0.08,
        handle_rgba="1 1 1 1",
        head_rgba="0 0 0 1",
        mass=0.1,
        upright=True,
    )
    body = CapsuleWiperGeometry.body(wiper=wiper)
    assert body.attrib["name"] == wiper.name
    assert body.find("freejoint").attrib["name"] == wiper.joint_name
    handle, head = body.findall("geom")
    assert handle.attrib["type"] == "box"
    assert head.attrib["type"] == "capsule"
    radius, half_length = np.fromstring(head.attrib["size"], sep=" ")
    assert 2 * radius == pytest.approx(0.02)
    assert 2 * (half_length + radius) == pytest.approx(0.20)
    assert float(handle.attrib["mass"]) + float(head.attrib["mass"]) == pytest.approx(0.1)
    assert np.fromstring(head.attrib["pos"], sep=" ")[2] == pytest.approx(radius)


@pytest.mark.parametrize(
    "height,length,upright", [(0, 0.2, True), (0.02, 0.01, True), (0.02, 0.2, False)]
)
def test_invalid_capsule_configuration_is_rejected(*, height, length, upright):
    with pytest.raises(ValueError):
        CapsuleWiperGeometry.body(
            wiper=SimpleNamespace(
                head_height=height,
                head_length=length,
                upright=upright,
            )
        )
