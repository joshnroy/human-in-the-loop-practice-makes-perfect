"""A stroke whose base move leaves the wiper head over the cupboard brings the head
straight back out to the mouth before it descends.

On the one-grasp scene the stroke's `_base` move can turn the base while the head is
still over the cupboard's top board; the descent to above the box then pressed the head
onto that board and turned the wiper about 20 deg in the grasp, after which no base pose
cleared `cupboard_1_shelf3`. A head that is clear of the cupboard takes the old path.

`ShelfSkills.angled_stroke` is driven here with every motion primitive replaced by a
recorder, and stopped at its first reach-in move, so the test pins the order of the
moves before the reach-in, not the physics.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    CARRY_Z,
    CUBE_SHELF,
    HOVER,
    ShelfSkills,
)

FRONT_X = 1.373
BOARD = 0.2794
HEAD_HALF = (0.015, 0.1, 0.0075)
HEAD = 0
CUBE = (1.5, -0.12)
OFFSET = -0.05
Y = CUBE[1] + OFFSET  # the straight (angle 0) stroke's line


class _ReachedReachIn(Exception):
    pass


def _skills(*, head_after_base, head_yaw_deg=0.0):
    """A ShelfSkills with only the geometry `angled_stroke` reads up to its reach-in,
    and recording primitives."""
    skills = object.__new__(ShelfSkills)
    skills.front_x = FRONT_X
    skills.board = {CUBE_SHELF: BOARD}
    skills.head_geom = HEAD
    skills.model = SimpleNamespace(geom_size=np.array([HEAD_HALF]))
    skills.data = SimpleNamespace(geom_xpos=np.zeros((1, 3)), geom_xmat=np.zeros((1, 9)))
    skills.slip_deg = 0.0
    skills.notes = []
    skills.phase = "start"
    calls = []

    def set_head(*, position, yaw_deg):
        skills.data.geom_xpos[HEAD] = position
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        skills.data.geom_xmat[HEAD] = rotation.reshape(9)

    def raise_hand(*, z, label, holding):
        calls.append(("raise_hand", label))

    def move_base(*, target, label="move_base"):
        calls.append(("move_base", label))
        if label.endswith("_base"):
            set_head(position=head_after_base, yaw_deg=head_yaw_deg)

    def move_wiper(*, head, label, stem=None):
        calls.append(("move_wiper", label, tuple(round(float(v), 4) for v in head)))
        set_head(position=head, yaw_deg=0.0)

    def wiper_along(*, head, stem, base, label):
        raise _ReachedReachIn

    skills.raise_hand = raise_hand
    skills.move_base = move_base
    skills.move_wiper = move_wiper
    skills.wiper_along = wiper_along
    skills.angled_base = lambda **kwargs: np.array([0.5, Y, 0.0])
    skills.recalibrate_grasp = lambda: calls.append(("recalibrate_grasp", None))
    return skills, calls


def _stroke(*, skills):
    with pytest.raises(_ReachedReachIn):
        skills.angled_stroke(cube_xy=CUBE, offset=OFFSET, angle=0.0, name="s")


def _hover():
    return BOARD + HEAD_HALF[2] + HOVER


def _moves(*, calls):
    return [call[:2] for call in calls]


def test_a_head_over_the_top_board_comes_straight_back_to_the_mouth_first():
    # As diagnosed: after the base turned, the head sat at x = 1.35 with the wiper
    # turned 20 deg, so its footprint reached past the cupboard's front face.
    head_z = 0.825
    skills, calls = _skills(head_after_base=[1.35, Y, head_z], head_yaw_deg=20.0)
    _stroke(skills=skills)
    assert _moves(calls=calls) == [
        ("raise_hand", "s_raise"),
        ("move_base", "s_base"),
        ("move_wiper", "s_to_mouth"),
        ("move_wiper", "s_above_box"),
        ("move_wiper", "s_mouth"),
        ("recalibrate_grasp", None),
    ]
    out = next(c for c in calls if c[1] == "s_to_mouth")
    # Straight back: to the mouth, on the stroke's line, at the height it is at now.
    assert out[2] == (round(FRONT_X - 0.12, 4), round(Y, 4), head_z)


def test_a_head_inside_the_cupboard_comes_straight_back_to_the_mouth_too():
    skills, calls = _skills(head_after_base=[1.45, Y, _hover()])
    _stroke(skills=skills)
    assert ("move_wiper", "s_to_mouth") in _moves(calls=calls)


def test_a_head_clear_of_the_cupboard_takes_the_old_path_unchanged():
    # A stroke after the previous one's `_out`: the head at the mouth, square.
    skills, calls = _skills(head_after_base=[FRONT_X - 0.12, Y, _hover() + 0.12])
    _stroke(skills=skills)
    assert _moves(calls=calls) == [
        ("raise_hand", "s_raise"),
        ("move_base", "s_base"),
        ("move_wiper", "s_above_box"),
        ("move_wiper", "s_mouth"),
        ("recalibrate_grasp", None),
    ]


def test_a_square_head_just_short_of_the_front_face_is_clear():
    # The footprint, not the centre, decides: 1 mm short of the face is clear ...
    skills, _ = _skills(head_after_base=[0.0, 0.0, 0.0])
    skills.data.geom_xmat[HEAD] = np.eye(3).reshape(9)
    skills.data.geom_xpos[HEAD] = [FRONT_X - HEAD_HALF[0] - 0.001, Y, CARRY_Z]
    assert not skills.head_over_cupboard()


def test_a_turned_head_whose_centre_is_short_of_the_front_face_is_over_it():
    # ... but the same centre with the head turned 20 deg reaches 1.36 + 0.1 sin 20 deg
    # past it.
    skills, _ = _skills(head_after_base=[0.0, 0.0, 0.0])
    rotation = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    skills.data.geom_xmat[HEAD] = rotation.reshape(9)
    skills.data.geom_xpos[HEAD] = [FRONT_X - HEAD_HALF[0] - 0.001, Y, CARRY_Z]
    assert skills.head_over_cupboard()
