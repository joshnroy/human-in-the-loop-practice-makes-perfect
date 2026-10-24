"""`ShelfSkills.sweep_shelf`: one angled stroke aimed at one cube (Josh, 2026-10-24).

The sampled parameters are [offset, angle]; the pull speed and the head height are
fixed constants. Each call runs `ShelfSkills.angled_stroke` exactly once on the line
through the target cube's live position, and succeeds iff the target cube ends in the
box. `angled_stroke` and the scene reads are replaced by recorders, so these tests pin
the skill's logic, not the physics (`test_angled_lane.py` pins the line's geometry).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.sweepoffshelf3d import scripted_skills
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    SWEEP_ANGLE_BOUNDS,
    SWEEP_HEIGHT,
    SWEEP_OFFSET_BOUNDS,
    SWEEP_SPEED,
    ShelfSkills,
    SkillFailed,
    SweepParams,
    WiperLost,
)

CUBE_XY = {
    "cube_0": (1.55, -0.12),
    "cube_1": (1.50, -0.04),
    "cube_2": (1.52, 0.03),
    "cube_3": (1.45, 0.09),
    "cube_4": (1.50, 0.15),
}


def _skills(*, lands_in_box=(), on_shelf=tuple(CUBE_XY), stroke_raises=None):
    """`lands_in_box`: the cubes the stroke leaves in the box."""
    skills = object.__new__(ShelfSkills)
    skills._in_hand = np.eye(4)  # noqa: SLF001
    skills.watch_wiper, skills.wiper_unheld_steps = False, 0
    skills.notes = []
    skills.phase = "start"
    strokes = []
    boxed: set[str] = set()

    def angled_stroke(*, cube_xy, offset, angle, name):
        strokes.append({
            "cube_xy": tuple(float(v) for v in cube_xy),
            "offset": offset,
            "angle": angle,
            "name": name,
            "watched": skills.watch_wiper,
        })
        if stroke_raises is not None:
            raise stroke_raises
        boxed.update(lands_in_box)

    skills.angled_stroke = angled_stroke
    skills.cube_position = lambda *, cube: np.array([*CUBE_XY[cube], 0.29])
    skills.shelf_cubes = lambda: [c for c in on_shelf if c not in boxed]
    skills.core = SimpleNamespace(cube_in_box=lambda name: name in boxed)
    return skills, strokes


def test_the_sampled_parameters_are_offset_and_angle_and_the_rest_is_fixed():
    assert SweepParams.model_fields.keys() == {"offset", "angle"}
    assert SWEEP_OFFSET_BOUNDS == (-0.10, 0.10)
    assert SWEEP_ANGLE_BOUNDS == (-45.0, 45.0)
    assert (SWEEP_SPEED, SWEEP_HEIGHT) == (0.05, 0.01)
    # No speed or height parameter survives anywhere in the controller.
    for gone in ("SweepSettings", "SweepLane", "SWEEP_SPEED_BOUNDS", "Strokes"):
        assert not hasattr(scripted_skills, gone), gone
    for gone in ("stroke", "pull", "lower_head"):
        assert not hasattr(ShelfSkills, gone), gone


def test_one_call_runs_exactly_one_angled_stroke_through_the_target_cube():
    skills, strokes = _skills(lands_in_box=("cube_2",))
    skills.sweep_shelf(cube="cube_2", offset=-0.05, angle=15.0)
    assert len(strokes) == 1
    (stroke,) = strokes
    assert stroke["cube_xy"] == pytest.approx(CUBE_XY["cube_2"])
    assert (stroke["offset"], stroke["angle"]) == (-0.05, 15.0)
    # The per-step HoldingWiper check is on during the stroke and off after it.
    assert stroke["watched"]
    assert not skills.watch_wiper


def test_the_sweep_fails_iff_its_cube_is_not_in_the_box():
    """Hard success per call: other cubes landing in the box do not count."""
    skills, strokes = _skills(lands_in_box=("cube_1", "cube_3"))
    with pytest.raises(SkillFailed, match="cube_2"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, angle=0.0)
    assert len(strokes) == 1
    skills, _ = _skills(lands_in_box=("cube_2",))
    skills.sweep_shelf(cube="cube_2", offset=0.05, angle=0.0)


def test_a_cube_not_on_the_shelf_is_refused_before_any_motion():
    skills, strokes = _skills(on_shelf=("cube_0", "cube_1"))
    with pytest.raises(SkillFailed, match="not on the shelf"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, angle=0.0)
    assert strokes == []


def test_no_wiper_in_hand_is_refused_before_any_motion():
    skills, strokes = _skills()
    skills._in_hand = None  # noqa: SLF001
    with pytest.raises(SkillFailed, match="wiper is not in hand"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, angle=0.0)
    assert strokes == []


def test_a_wiper_lost_mid_stroke_is_a_skill_failure():
    skills, _ = _skills(stroke_raises=WiperLost("the wiper left the hand during probe"))
    with pytest.raises(SkillFailed, match="left the hand"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, angle=0.0)
    assert not skills.watch_wiper


@pytest.mark.parametrize(
    ("offset", "angle"), [(0.11, 0.0), (-0.11, 0.0), (0.0, 46.0), (0.0, -46.0)]
)
def test_parameters_outside_their_bounds_are_refused(*, offset, angle):
    with pytest.raises(ValidationError):
        SweepParams(offset=offset, angle=angle)
    skills, strokes = _skills()
    with pytest.raises(ValidationError):
        skills.sweep_shelf(cube="cube_2", offset=offset, angle=angle)
    assert strokes == []
