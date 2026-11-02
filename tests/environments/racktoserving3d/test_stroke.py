import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.stroke import plan_stroke


def test_rotated_start_is_behind_every_food_corner():
    corners = np.array([[1.56, 0.08, 0.94], [1.52, 0.06, 0.94], [1.45, -0.06, 0.94]])
    plan = plan_stroke(
        food_corners=corners,
        rack_back=1.6275,
        rack_front=1.3725,
        rack_center_y=0,
        rack_top=0.925,
        angle_deg=-40,
        blade_width=0.25,
        tray_back=1.3625,
        clearance=0.035,
    )
    assert np.min((corners - plan.start) @ plan.direction) >= 0.035 - 1e-9
    assert plan.end[0] < 1.3625
    assert plan.start[2] == pytest.approx(0.926)


def test_food_outside_blade_is_not_claimed_to_be_sweepable():
    with pytest.raises(ValueError, match="width"):
        plan_stroke(
            food_corners=np.array([[1.5, 0.3, 0.94]]),
            rack_back=1.6275,
            rack_front=1.3725,
            rack_center_y=0,
            rack_top=0.925,
            angle_deg=0,
            blade_width=0.25,
            tray_back=1.3625,
        )


@pytest.mark.parametrize("angle", [-20, 0, 20])
def test_entire_blade_follows_through_by_food_diameter_plus_tracking_margin(*, angle):
    plan = plan_stroke(
        food_corners=np.array([[1.50, 0, 0.94]]),
        rack_back=1.6275,
        rack_front=1.3725,
        rack_center_y=0,
        rack_top=0.925,
        angle_deg=angle,
        blade_width=0.25,
        tray_back=1.3625,
        release_clearance=0.055,
    )
    across = np.array([plan.direction[1], -plan.direction[0], 0])
    edges = np.stack([plan.end + 0.125 * across, plan.end - 0.125 * across])
    assert edges[:, 0].max() == pytest.approx(1.3625 - 0.055)


@pytest.mark.parametrize("angle", [float("nan"), -91, 91])
def test_invalid_angle_rejected(*, angle):
    with pytest.raises(ValueError):
        plan_stroke(
            food_corners=np.array([[1.5, 0, 0.94]]),
            rack_back=1.6275,
            rack_front=1.3725,
            rack_center_y=0,
            rack_top=0.925,
            angle_deg=angle,
            blade_width=0.25,
            tray_back=1.3625,
        )
