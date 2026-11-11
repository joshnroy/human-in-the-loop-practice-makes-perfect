"""The box's drop point is read from the live scene, not a fixed coordinate: a box
moved or resized in the task JSON is dropped into where it is."""

from types import SimpleNamespace

import pytest

from hitl_pmp.environments.sweepoffshelf3d import scripted_skills
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills


@pytest.mark.parametrize(
    ("interior", "centre"),
    [
        ((1.12, -0.24, 1.35, 0.24), (1.235, 0.0)),  # today's 25 x 50 cm box
        ((1.22, -0.065, 1.35, 0.065), (1.285, 0.0)),  # a 15 cm box further in
        ((1.10, 0.05, 1.30, 0.25), (1.20, 0.15)),
    ],
)
def test_the_box_drop_point_is_the_live_interior_centre(*, interior, centre) -> None:
    skills = object.__new__(ShelfSkills)
    skills.core = SimpleNamespace(box_interior=lambda: (interior, 0.2))
    assert skills.box_drop_xy() == pytest.approx(centre)


def test_no_fixed_drop_coordinates_remain() -> None:
    assert not hasattr(scripted_skills, "BOX_DROP_XY")
    assert not hasattr(scripted_skills, "DROP_XY")
