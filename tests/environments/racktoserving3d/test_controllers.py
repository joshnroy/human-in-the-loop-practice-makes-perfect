from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


def test_surface_approach_raises_arm_before_advancing_base():
    controller = RackControllers.__new__(RackControllers)
    events = []
    staging = np.eye(4)
    staging[:3, 3] = [0.65, 0, 1.1]
    controller.shelf = SimpleNamespace(
        ik=Mock(return_value=(np.zeros(7), 0)),
        site_at=Mock(return_value=staging),
        data=SimpleNamespace(qpos=np.zeros(10)),
        ba=[0, 1, 2],
    )
    controller.tool = SimpleNamespace(
        check_configuration=Mock(side_effect=lambda **kwargs: events.append("check")),
        transit=Mock(side_effect=lambda **kwargs: events.append("arm")),
        move_base=Mock(side_effect=lambda **kwargs: events.append("base")),
        line=Mock(side_effect=lambda **kwargs: events.append("hover")),
    )
    controller.approach_surface(
        position=np.array([1.44, 0, 1.1]),
        rotation=np.diag([1.0, -1.0, -1.0]),
        base=np.array([0.79, 0, 0]),
        holding=False,
        label="test",
    )
    assert events == ["check", "arm", "base", "hover"]
    np.testing.assert_allclose(controller.tool.transit.call_args.kwargs["position"], staging[:3, 3])


def test_blocked_destination_does_not_start_surface_approach():
    controller = RackControllers.__new__(RackControllers)
    controller.shelf = SimpleNamespace(ik=Mock(return_value=(np.zeros(7), 0)))
    controller.tool = SimpleNamespace(
        check_configuration=Mock(side_effect=SkillFailed("blocked")),
        transit=Mock(),
        move_base=Mock(),
    )
    with pytest.raises(SkillFailed, match="blocked"):
        controller.approach_surface(
            position=np.ones(3), rotation=np.eye(3), base=np.zeros(3), holding=True, label="test"
        )
    controller.tool.transit.assert_not_called()
    controller.tool.move_base.assert_not_called()


def test_push_keeps_45_degree_handle_behind_the_blade():
    rotation = RackControllers.push_rotation(phi=0)
    handle_direction = rotation[:, 2]
    assert handle_direction[0] > 0
    assert handle_direction[2] > 0
    assert handle_direction[0] == pytest.approx(handle_direction[2])


def test_upright_push_has_horizontal_face_normal():
    rotation = RackControllers.push_rotation(phi=0, tilt=0)
    np.testing.assert_allclose(rotation[:, 1], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)


def test_tool_target_uses_live_grasp_after_transport():
    controller = RackControllers.__new__(RackControllers)
    hand, body, target = np.eye(4), np.eye(4), np.eye(4)
    hand[:3, 3] = [1.0, 0, 1.0]
    body[:3, 3] = [1.1, 0.02, 0.9]
    target[:3, 3] = [1.6, 0, 0.93]
    controller.shelf = SimpleNamespace(site_pose=Mock(return_value=hand), in_hand=np.eye(4))
    controller.tool = SimpleNamespace(require_held=Mock(), wiper_pose=Mock(return_value=body))
    commanded = controller.hand_for_tool(body=target)
    np.testing.assert_allclose(commanded @ controller.shelf.in_hand, target)
    assert not np.allclose(commanded, target)
    controller.tool.require_held.assert_called_once()


def test_counter_circulation_goes_outside_before_crossing_counter_x():
    controller = RackControllers.__new__(RackControllers)
    controller.backend = SimpleNamespace(config=RackServingConfig())
    controller.shelf = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3)), ba=[0, 1, 2])
    controller.tool = SimpleNamespace(base=np.array([0.6, -0.34, 0]), move_base=Mock())
    controller.around_counter(to_rear=True)
    route = [call.kwargs["target"] for call in controller.tool.move_base.call_args_list]
    assert route[0][0] < controller.backend.config.counter_x_min
    assert route[0][1] < controller.backend.config.counter_y_min
    assert route[1][0] > controller.backend.config.counter_x_max
    assert route[1][1] == route[0][1]
    assert route[-1][2] == pytest.approx(np.pi)


def test_release_lifts_without_dragging_food_back_then_waits():
    controller = RackControllers.__new__(RackControllers)
    pose = np.eye(4)
    pose[:3, 3] = [1.33, 0.01, 0.926]
    events = []
    controller.tool = SimpleNamespace(wiper_pose=Mock(return_value=pose))
    controller.move_tool = Mock(side_effect=lambda **kwargs: events.append("lift"))
    controller.shelf = SimpleNamespace(
        hold=Mock(side_effect=lambda **kwargs: events.append("settle"))
    )
    controller.release_swept_food()
    target = controller.move_tool.call_args.kwargs["body"]
    np.testing.assert_allclose(target[:3, 3], [1.33, 0.01, 0.986])
    np.testing.assert_allclose(pose[:3, 3], [1.33, 0.01, 0.926])
    assert events == ["lift", "settle"]
    controller.shelf.hold.assert_called_once_with(count=40, label="food_settle")


GEOMETRY = {
    "food_corners": np.array([[[1.5, 0.0, 0.94]]]),
    "rack_back": 1.6275,
    "rack_front": 1.3725,
    "rack_y_min": -0.1875,
    "rack_y_max": 0.1875,
    "rack_top": 0.925,
    "plate_center_x": 1.2323,
    "plate_center_y": 0.0,
    "plate_radius": 0.1302,
    "blade_width": 0.25,
}


@pytest.mark.parametrize("phi", [-15.0, 0.0, 15.0])
def test_plow_rotation_yaws_the_blade_line_like_the_planned_sweep(*, phi):
    from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner

    rotation = RackControllers.stroke_rotation(
        stroke=GatherAndSweepPlanner.sweep(d=0.1, phi_deg=phi, **GEOMETRY)
    )
    plan = GatherAndSweepPlanner.sweep(d=0.1, phi_deg=phi, **GEOMETRY)
    np.testing.assert_allclose(rotation[:, 0], plan.axis, atol=1e-12)
    np.testing.assert_allclose(rotation, RackControllers.push_rotation(phi=phi), atol=1e-12)
    lean = rotation[:, 2]
    assert lean[:2] @ plan.axis[:2] == pytest.approx(0, abs=1e-12)
    assert lean[0] > 0 and lean[2] > 0


def test_gather_rotations_span_x_and_lean_against_each_push_like_the_sweep():
    from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner

    sweep = GatherAndSweepPlanner.sweep(d=0.1, phi_deg=0.0, **GEOMETRY)
    sweep_lean = RackControllers.stroke_rotation(stroke=sweep)[:, 2]
    for stroke in GatherAndSweepPlanner.gather(delta=0.0, **GEOMETRY):
        rotation = RackControllers.stroke_rotation(stroke=stroke)
        np.testing.assert_allclose(np.abs(rotation[:, 0]), [1.0, 0.0, 0.0], atol=1e-12)
        lean = rotation[:, 2]
        # Same lean as the sweep, measured relative to each stroke's own travel.
        assert lean @ stroke.direction == pytest.approx(sweep_lean @ sweep.direction)
        assert lean[2] == pytest.approx(sweep_lean[2])


@pytest.mark.parametrize(
    ("delta", "d", "phi"),
    [
        (0.031, 0.1, 0.0),
        (-0.021, 0.1, 0.0),
        (0.0, 0.049, 0.0),
        (0.0, 0.251, 0.0),
        (0.0, 0.1, 15.1),
        (0.0, 0.1, -15.1),
        (float("nan"), 0.1, 0.0),
    ],
)
def test_gather_and_sweep_rejects_parameters_outside_bounds_before_touching_the_scene(
    *, delta, d, phi
):
    controller = RackControllers.__new__(RackControllers)
    with pytest.raises(SkillFailed, match="GatherAndSweep requires"):
        controller.gather_and_sweep(delta=delta, d=d, phi=phi)


@pytest.mark.parametrize("side", [1, -1])
def test_gather_base_placement_is_one_mirrored_side_stance_outside_the_counter(*, side):
    from hitl_pmp.environments.racktoserving3d.controllers import GatherBasePlacement

    config = RackServingConfig()
    placement = GatherBasePlacement()
    base = placement.base(side=side, config=config)
    route = placement.route(side=side, config=config, start=np.array([0.5, 0.0, 0.0]))
    # The robot stands beside the counter on the side the stroke starts from.
    assert side * base[1] > config.counter_width / 2
    assert config.counter_x_min <= base[0] <= config.counter_x_max
    np.testing.assert_allclose(route[-1], base)
    lane = side * (config.counter_width / 2 + placement.lane_clearance)
    assert all(abs(waypoint[1] - lane) < 1e-9 for waypoint in route[1:-1])
    mirrored = placement.base(side=-side, config=config)
    np.testing.assert_allclose(mirrored[:2], [base[0], -base[1]])
