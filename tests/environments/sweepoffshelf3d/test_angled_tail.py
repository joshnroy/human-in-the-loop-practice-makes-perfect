"""The angled stroke's tail geometry and the box drop region on a small box."""

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills


@pytest.mark.parametrize("angle", [-0.8, -0.3, 0.3, 0.8])
def test_turning_the_base_about_the_head_keeps_the_head_still(*, angle: float) -> None:
    base = np.array([0.45, -0.2, -0.5])
    head = np.array([1.25, 0.1])
    turned = ShelfSkills.rotated_base(base=base, pivot=head, angle=angle)
    assert np.isclose(np.linalg.norm(turned[:2] - head), np.linalg.norm(base[:2] - head))
    assert np.isclose(turned[2], base[2] + angle)
    # The head, fixed in the base's frame, lands where it was.
    c0, s0 = np.cos(base[2]), np.sin(base[2])
    local = np.array([[c0, s0], [-s0, c0]]) @ (head - base[:2])
    c1, s1 = np.cos(turned[2]), np.sin(turned[2])
    assert np.allclose(turned[:2] + np.array([[c1, -s1], [s1, c1]]) @ local, head)


@pytest.mark.parametrize("deg", [-45.0, -10.0, 0.0, 30.0])
def test_turning_by_the_stem_yaw_brings_the_stem_to_minus_x(*, deg: float) -> None:
    a = np.deg2rad(deg)
    stem = np.array([-np.cos(a), np.sin(a)])
    yaw = ShelfSkills.stem_yaw(stem=stem)
    c, s = np.cos(yaw), np.sin(yaw)
    assert np.allclose(np.array([[c, -s], [s, c]]) @ stem, [-1.0, 0.0])


def test_the_drop_region_falls_back_to_the_interior_centre() -> None:
    # A 15 cm box's 13 cm interior cannot hold the 11 cm + 4 cm margins.
    assert KinderSkills.drop_axis(low=1.22, high=1.35, margins=(0.11, 0.04)) == [
        pytest.approx(1.285)
    ]
    wide = KinderSkills.drop_axis(low=-0.14, high=0.14, margins=(0.12, 0.12))
    assert min(wide) >= -0.02 - 1e-9 and max(wide) <= 0.02 + 1e-9
