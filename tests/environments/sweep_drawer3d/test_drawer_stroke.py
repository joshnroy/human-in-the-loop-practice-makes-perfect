"""When a drawer stroke counts as done: a close has to actually shut the drawer."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)


def _stroke():
    from hitl_pmp.environments.sweep_drawer3d.primitives import DrawerStroke

    return DrawerStroke


def test_a_close_that_stalls_a_centimetre_short_has_not_closed_the_drawer() -> None:
    """Seed 1 of the 16-seed measurement: the close stopped at 1.13 cm, the primitive
    called that a success, and the reset ended at 1.01 cm."""
    assert not _stroke().reached(end=0.0113, target=0.0)
    assert not _stroke().reached(end=0.0101, target=0.0)


def test_a_drawer_at_its_joint_limit_is_closed() -> None:
    assert _stroke().reached(end=0.0001, target=0.0)


def test_the_close_tolerance_is_inside_the_resets_own_closed_criterion() -> None:
    from hitl_pmp.environments.sweep_drawer3d.types import ResetOutcome

    fields = {
        "locations": {},
        "in_pile": {},
        "drawer_pos": _stroke().CLOSED_TOLERANCE,
        "wiper_xy_error": 0.0,
        "wiper_yaw_error_deg": 0.0,
        "wiper_on_counter": True,
        "robot_actions": 0,
        "ticks": 0,
        "wall_s": 0.0,
    }
    assert ResetOutcome.model_validate(fields).drawer_closed


def test_an_opening_stroke_may_stop_within_three_centimetres() -> None:
    assert _stroke().reached(end=0.246, target=0.25)
    assert not _stroke().reached(end=0.20, target=0.25)


def test_a_drawer_that_stayed_behind_has_slipped_out_of_the_hand() -> None:
    """Measured on a laid-out scene: re-opened at 0.03 m a tick after a slam, the drawer
    came back to 0.140 of 0.252, and on the next stroke to 0.008."""
    from hitl_pmp.environments.sweep_drawer3d.repositioning import WiggleGrip

    assert WiggleGrip.slipped(opened=0.252, now=0.140)
    assert WiggleGrip.slipped(opened=0.252, now=0.008)


def test_a_drawer_that_came_back_a_little_far_is_still_in_the_hand() -> None:
    """It overshoots: 0.270 to 0.276 for a drawer gripped at 0.252."""
    from hitl_pmp.environments.sweep_drawer3d.repositioning import WiggleGrip

    assert not WiggleGrip.slipped(opened=0.252, now=0.276)
    assert not WiggleGrip.slipped(opened=0.252, now=0.240)


def test_a_closing_stroke_drives_the_base_past_shut() -> None:
    """A stroke exactly as long as the remaining slide loses part of it to slack in the
    grip: on seed 1 a 1.5 cm stroke moved the drawer 4 mm."""
    assert _stroke().base_travel(start=0.25, target=0.0) == pytest.approx(-0.27)


def test_an_opening_stroke_drives_the_base_exactly_the_slide() -> None:
    assert _stroke().base_travel(start=0.0, target=0.25) == pytest.approx(0.25)
