"""`ShelfSkills.sweep_shelf`: one front-to-back stroke aimed at one cube (Josh,
2026-10-23).

The lane is the target cube's live y plus the offset, clipped to +/- STROKE_LIMIT; the
stroke is the existing `ShelfSkills.stroke`, run exactly once at the fixed head height;
the call succeeds iff the target cube ends in the box. `stroke` and the scene reads
are replaced by recorders, so these tests pin the skill's logic, not the physics.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    STROKE_LIMIT,
    SWEEP_HEIGHT,
    ShelfSkills,
    SkillFailed,
    SweepLane,
    SweepParams,
    WiperLost,
)

CUBE_YS = {"cube_0": -0.12, "cube_1": -0.04, "cube_2": 0.03, "cube_3": 0.09, "cube_4": 0.15}


def _skills(*, lands_in_box=(), on_shelf=tuple(CUBE_YS), stroke_raises=None):
    """`lands_in_box`: the cubes the stroke leaves in the box."""
    skills = object.__new__(ShelfSkills)
    skills._in_hand = np.eye(4)  # noqa: SLF001
    skills.watch_wiper, skills.wiper_unheld_steps = False, 0
    skills.notes = []
    strokes = []
    boxed: set[str] = set()

    def stroke(*, y, settings, name):
        strokes.append({"y": y, "settings": settings, "name": name, "watched": skills.watch_wiper})
        if stroke_raises is not None:
            raise stroke_raises
        boxed.update(lands_in_box)

    skills.stroke = stroke
    skills.cube_position = lambda *, cube: np.array([1.5, CUBE_YS[cube], 0.29])
    skills.shelf_cubes = lambda: [c for c in on_shelf if c not in boxed]
    skills.core = SimpleNamespace(cube_in_box=lambda name: name in boxed)
    return skills, strokes


@pytest.mark.parametrize(
    ("cube_y", "offset", "lane"),
    [
        (0.03, 0.06, 0.09),
        (0.03, -0.06, -0.03),
        (0.15, 0.10, STROKE_LIMIT),
        (-0.12, -0.10, -STROKE_LIMIT),
        (0.0, 0.0, 0.0),
    ],
)
def test_the_lane_is_the_cube_y_plus_the_offset_clipped_to_the_stroke_limit(
    *, cube_y, offset, lane
):
    assert SweepLane.y(cube_y=cube_y, offset=offset) == pytest.approx(lane)


def test_one_call_runs_exactly_one_stroke_on_the_target_lane_at_the_fixed_height():
    skills, strokes = _skills(lands_in_box=("cube_2",))
    skills.sweep_shelf(cube="cube_2", offset=-0.05, speed=0.07)
    assert len(strokes) == 1
    (stroke,) = strokes
    assert stroke["y"] == pytest.approx(CUBE_YS["cube_2"] - 0.05)
    assert stroke["settings"].height == SWEEP_HEIGHT == 0.01
    assert stroke["settings"].speed == pytest.approx(0.07)
    # The per-step HoldingWiper check is on during the stroke and off after it.
    assert stroke["watched"]
    assert not skills.watch_wiper


def test_the_sweep_fails_iff_its_cube_is_not_in_the_box():
    """Hard success per call: other cubes landing in the box do not count."""
    skills, strokes = _skills(lands_in_box=("cube_1", "cube_3"))
    with pytest.raises(SkillFailed, match="cube_2"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, speed=0.05)
    assert len(strokes) == 1
    skills, _ = _skills(lands_in_box=("cube_2",))
    skills.sweep_shelf(cube="cube_2", offset=0.05, speed=0.05)


def test_a_cube_not_on_the_shelf_is_refused_before_any_motion():
    skills, strokes = _skills(on_shelf=("cube_0", "cube_1"))
    with pytest.raises(SkillFailed, match="not on the shelf"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, speed=0.05)
    assert strokes == []


def test_no_wiper_in_hand_is_refused_before_any_motion():
    skills, strokes = _skills()
    skills._in_hand = None  # noqa: SLF001
    with pytest.raises(SkillFailed, match="wiper is not in hand"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, speed=0.05)
    assert strokes == []


def test_a_wiper_lost_mid_stroke_is_a_skill_failure():
    skills, _ = _skills(stroke_raises=WiperLost("the wiper left the hand during probe"))
    with pytest.raises(SkillFailed, match="left the hand"):
        skills.sweep_shelf(cube="cube_2", offset=0.05, speed=0.05)
    assert not skills.watch_wiper


@pytest.mark.parametrize(("offset", "speed"), [(0.11, 0.1), (-0.11, 0.1), (0.0, 0.04), (0.0, 0.26)])
def test_parameters_outside_their_bounds_are_refused(*, offset, speed):
    with pytest.raises(ValidationError):
        SweepParams(offset=offset, speed=speed)
    skills, strokes = _skills()
    with pytest.raises(ValidationError):
        skills.sweep_shelf(cube="cube_2", offset=offset, speed=speed)
    assert strokes == []
