"""A stroke whose base move leaves the wiper head over the cupboard brings the head
straight back out to the mouth before it descends.

On the one-grasp scene the stroke's `_base` move can turn the base while the head is
still over the cupboard's top board; the descent to above the box then pressed the head
onto that board and turned the wiper about 20 deg in the grasp, after which no base pose
cleared `cupboard_1_shelf3`. A head that is clear of the cupboard takes the old path.

`ShelfSkills.stroke` is driven here with every motion primitive replaced by a recorder,
so the test pins the order of the moves, not the physics.
"""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import (
    CARRY_Z,
    CUBE_SHELF,
    HOVER,
    ShelfSkills,
    SweepSettings,
)

FRONT_X = 1.373
BOARD = 0.2794
HEAD_HALF = (0.015, 0.1, 0.0075)
HEAD = 0
Y = -0.17


def _skills(*, head_after_base, head_yaw_deg=0.0):
    """A ShelfSkills with only the geometry `stroke` reads, and recording primitives."""
    skills = object.__new__(ShelfSkills)
    skills.front_x = FRONT_X
    skills.board = {CUBE_SHELF: BOARD}
    skills.head_geom = HEAD
    skills.model = SimpleNamespace(geom_size=np.array([HEAD_HALF]))
    skills.data = SimpleNamespace(geom_xpos=np.zeros((1, 3)), geom_xmat=np.zeros((1, 9)))
    skills.slip_deg = 0.0
    skills.notes = []
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

    def wiper_to(*, head, label):
        calls.append(("wiper_to", label, tuple(round(float(v), 4) for v in head)))
        set_head(position=head, yaw_deg=0.0)

    def record(*, name):
        return lambda **kwargs: calls.append((name, kwargs.get("label")))

    skills.raise_hand = raise_hand
    skills.move_base = move_base
    skills.wiper_to = wiper_to
    skills.recalibrate_grasp = lambda: calls.append(("recalibrate_grasp", None))
    skills.lower_head = record(name="lower_head")
    skills.mark = record(name="mark")
    skills.pull = record(name="pull")
    skills.hold = record(name="hold")
    skills.hand_to = record(name="hand_to")
    skills.reseat_wiper = record(name="reseat_wiper")
    skills.level_target = lambda **kwargs: np.eye(4)
    skills.counts = lambda: {"in_box": 0}
    return skills, calls


def _stroke(*, skills):
    skills.stroke(y=Y, settings=SweepSettings(height=0.01, speed=0.18), name="s")


def _hover():
    return BOARD + HEAD_HALF[2] + HOVER


def _moves_before_the_reach_in(*, calls):
    out = []
    for call in calls:
        if call[1] == "s_reach_in":
            break
        out.append(call[:2])
    return out


def test_a_head_over_the_top_board_comes_straight_back_to_the_mouth_first():
    # As diagnosed: after the base turned, the head sat at x = 1.35 with the wiper
    # turned 20 deg, so its footprint reached past the cupboard's front face.
    head_z = 0.825
    skills, calls = _skills(head_after_base=[1.35, Y, head_z], head_yaw_deg=20.0)
    _stroke(skills=skills)
    assert _moves_before_the_reach_in(calls=calls) == [
        ("raise_hand", "s_raise"),
        ("move_base", "s_base"),
        ("wiper_to", "s_to_mouth"),
        ("wiper_to", "s_above_box"),
        ("wiper_to", "s_mouth"),
        ("recalibrate_grasp", None),
    ]
    out = next(c for c in calls if c[1] == "s_to_mouth")
    # Straight back: to the mouth, at the stroke's y, at the height it is at now.
    assert out[2] == (round(FRONT_X - 0.12, 4), Y, head_z)


def test_a_head_inside_the_cupboard_comes_straight_back_to_the_mouth_too():
    skills, calls = _skills(head_after_base=[1.45, Y, _hover()])
    _stroke(skills=skills)
    assert ("wiper_to", "s_to_mouth") in _moves_before_the_reach_in(calls=calls)


def test_a_head_clear_of_the_cupboard_takes_the_old_path_unchanged():
    # A stroke after the previous one's `_out`: the head at the mouth, square.
    skills, calls = _skills(head_after_base=[FRONT_X - 0.12, Y, _hover() + 0.12])
    _stroke(skills=skills)
    assert _moves_before_the_reach_in(calls=calls) == [
        ("raise_hand", "s_raise"),
        ("move_base", "s_base"),
        ("wiper_to", "s_above_box"),
        ("wiper_to", "s_mouth"),
        ("recalibrate_grasp", None),
    ]
    assert not any(c[1] == "s_to_mouth" for c in calls)


def test_a_square_head_just_short_of_the_front_face_is_clear():
    # The footprint, not the centre, decides: 1 mm short of the face is clear ...
    skills, _ = _skills(head_after_base=[FRONT_X - HEAD_HALF[0] - 0.001, Y, CARRY_Z])
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
