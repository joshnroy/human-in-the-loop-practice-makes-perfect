from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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
FOOD = np.array([[[1.5, 0.0, 0.94]]])


def test_squeeze_and_sweep_hold_the_blade_like_the_old_straight_sweep():
    from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner

    sweep = GatherAndSweepPlanner.sweep(d=0.1, plate_clear_z=0.921, **GEOMETRY)
    for stroke in (*GatherAndSweepPlanner.squeeze(s=0.07, **GEOMETRY), sweep):
        rotation = RackControllers.stroke_rotation(stroke=stroke)
        np.testing.assert_allclose(rotation, RackControllers.push_rotation(phi=0), atol=1e-12)
        lean = rotation[:, 2]
        assert lean[0] > 0 and lean[2] > 0


def test_gather_rotations_span_x_and_lean_against_each_push_like_the_sweep():
    from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner

    sweep = GatherAndSweepPlanner.sweep(d=0.1, plate_clear_z=0.921, **GEOMETRY)
    sweep_lean = RackControllers.stroke_rotation(stroke=sweep)[:, 2]
    for stroke in GatherAndSweepPlanner.gather(food_corners=FOOD, delta=0.045, **GEOMETRY):
        rotation = RackControllers.stroke_rotation(stroke=stroke)
        np.testing.assert_allclose(np.abs(rotation[:, 0]), [1.0, 0.0, 0.0], atol=1e-12)
        lean = rotation[:, 2]
        assert lean @ stroke.direction == pytest.approx(sweep_lean @ sweep.direction)
        assert lean[2] == pytest.approx(sweep_lean[2])


@pytest.mark.parametrize(
    ("s", "delta", "d"),
    [
        (0.029, 0.045, 0.1),
        (0.141, 0.045, 0.1),
        (0.07, 0.004, 0.1),
        (0.07, 0.076, 0.1),
        (0.07, 0.045, 0.039),
        (0.07, 0.045, 0.121),
        (float("nan"), 0.045, 0.1),
    ],
)
def test_gather_and_sweep_rejects_parameters_outside_bounds_before_touching_the_scene(
    *, s, delta, d
):
    controller = RackControllers.__new__(RackControllers)
    with pytest.raises(SkillFailed, match="GatherAndSweep requires"):
        controller.gather_and_sweep(s=s, delta=delta, d=d)


@pytest.mark.parametrize("side", [1, -1])
def test_side_stances_are_the_real_arm_validated_mirrored_poses(*, side):
    from hitl_pmp.environments.racktoserving3d.controllers import StrokeStances

    config = RackServingConfig()
    stances = StrokeStances()
    np.testing.assert_allclose(
        stances.side_base(side=side, config=config), [1.5, side * 0.85, -side * np.pi / 2]
    )
    lane = side * (config.counter_width / 2 + stances.lane_clearance)
    assert lane == pytest.approx(side * 1.0)


def test_rear_stance_is_the_validated_sweep_pose_shifted_with_the_stroke():
    from hitl_pmp.environments.racktoserving3d.controllers import StrokeStances

    config = RackServingConfig()
    stances = StrokeStances()
    np.testing.assert_allclose(stances.rear_base(y=0.0, config=config), [2.05, -0.15, np.pi - 0.30])
    np.testing.assert_allclose(
        stances.rear_base(y=0.0625, config=config), [2.05, -0.0875, np.pi - 0.30]
    )


def test_routes_between_stances_stay_outside_the_counter_and_end_on_the_stance():
    from hitl_pmp.environments.racktoserving3d.controllers import StrokeStances

    config = RackServingConfig()
    stances = StrokeStances()

    def outside(*, point):
        return (
            abs(point[1]) > config.counter_width / 2
            or point[0] > config.counter_x_max
            or point[0] < config.counter_x_min
        )

    rear = stances.rear_base(y=-0.0625, config=config)
    plus = stances.side_base(side=1, config=config)
    minus = stances.side_base(side=-1, config=config)
    for route, goal in (
        (stances.rear_to_side(side=1, start=rear, config=config), plus),
        (stances.side_to_side(side=-1, config=config), minus),
        (stances.side_to_rear(side=-1, config=config), None),
    ):
        assert all(outside(point=point) for point in route)
        if goal is not None:
            np.testing.assert_allclose(route[-1], goal)
        else:
            assert route[-1][0] > config.counter_x_max
            assert route[-1][2] == pytest.approx(np.pi)
    assert stances.side_exit == "rear_lane"


def test_release_levels_and_backs_off_a_rolled_blade_before_rising():
    controller = RackControllers.__new__(RackControllers)
    rolled = np.eye(4)
    rolled[:3, :3] = Rotation.from_euler("y", 4, degrees=True).as_matrix()
    rolled[:3, 3] = [1.50, -0.04, 0.926]
    edge = np.array([[1.375, -0.04, 0.9349], [1.625, -0.04, 0.9171]])
    level = RackControllers.push_rotation(phi=0)
    targets = []
    controller.tool = SimpleNamespace(
        wiper_pose=Mock(return_value=rolled),
        blade_edge=Mock(
            side_effect=lambda pose=None: (
                edge if pose is None else edge - rolled[:3, 3] + pose[:3, 3]
            )
        ),
    )
    controller.move_tool = Mock(side_effect=lambda **kwargs: targets.append(kwargs["body"]))
    controller.shelf = SimpleNamespace(hold=Mock())
    controller.release_swept_food(rotation=level, retreat=np.array([0.0, -0.03, 0.0]))
    first, last = targets[0], targets[-1]
    np.testing.assert_allclose(first[:3, :3], level)
    np.testing.assert_allclose(last[:3, :3], level)
    # Levelled and backed off 3 cm against the push, about the edge centre, then lifted.
    assert first[2, 3] == pytest.approx(rolled[2, 3] + 0.002, abs=0.01)
    assert first[1, 3] - rolled[1, 3] == pytest.approx(-0.03, abs=0.01)
    assert last[1, 3] == pytest.approx(first[1, 3])
    assert last[2, 3] == pytest.approx(first[2, 3] + 0.06)


def test_sweep_clears_the_plate_rim_by_more_than_an_ik_accepted_blade_roll():
    controller = RackControllers.__new__(RackControllers)
    config = RackServingConfig()
    controller.backend = SimpleNamespace(
        config=config, support=Mock(return_value=SimpleNamespace(top_z=0.907))
    )
    # 2 deg of accepted IK rotation error drops one end of the 25 cm blade 4.4 mm.
    roll_drop = 0.125 * np.sin(np.deg2rad(2.0))
    clearance = controller.plate_clear_z() - config.corelle_rim_top_z
    # The blade's lowest point was measured 5 mm under its edge centre mid-sweep.
    assert clearance > max(roll_drop, 0.005) + 0.0025


def _settling_controller(*, lag):
    """Fake arm whose blade edge sits a fixed ``lag`` short of every commanded edge."""
    from hitl_pmp.environments.racktoserving3d.stroke import GatherAndSweepPlanner

    controller = RackControllers.__new__(RackControllers)
    stroke = GatherAndSweepPlanner.gather(food_corners=FOOD, delta=0.045, **GEOMETRY)[0]
    edge = {"now": stroke.end - lag}
    steps = []

    def step(*, arm, label):
        steps.append(label)
        edge["now"] = np.asarray(arm) - lag

    controller.tool = SimpleNamespace(
        blade_edge=Mock(side_effect=lambda pose=None: np.array([edge["now"], edge["now"]])),
        ik=Mock(side_effect=lambda **kwargs: kwargs["position"].copy()),
        step=Mock(side_effect=step),
    )
    controller.shelf = SimpleNamespace(
        base_target=None, data=SimpleNamespace(qpos=np.zeros(7)), qa=np.arange(7)
    )
    controller.metrics = {}
    controller.tool.wiper_pose = Mock(
        side_effect=lambda: np.block([
            [np.eye(3), edge["now"].reshape(3, 1)],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ])
    )
    controller.hand_for_tool = lambda *, body: body
    body = np.eye(4)
    body[:3, 3] = stroke.start
    return controller, stroke, body, steps


def test_stroke_end_settle_removes_a_steady_lag_along_travel_and_in_height():
    # The +y gather travels -y with the blade along x: travel is y, the blade's length x.
    lag = np.array([0.0, 0.0075, 0.001])
    controller, stroke, body, steps = _settling_controller(lag=lag)
    before, after = controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    assert before == pytest.approx(np.linalg.norm(lag))
    assert after < 0.005
    assert 0 < len(steps) <= RackControllers.SETTLE_STEPS
    assert set(steps) == {"gather_settle"}


def test_stroke_end_settle_never_corrects_along_the_blade():
    # Dragging the blade along its own length rearranged the food and made the next
    # gather climb a dumpling (real arm, 2026-11-03): that error is left alone.
    lag = np.array([0.05, 0.0075, 0.0])
    controller, stroke, body, _ = _settling_controller(lag=lag)
    _, after = controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    commanded = [call.kwargs["position"] for call in controller.tool.ik.call_args_list]
    assert commanded
    assert all(position[0] == pytest.approx(stroke.end[0]) for position in commanded)
    assert after == pytest.approx(0.05, abs=0.005)


def test_stroke_end_settle_does_nothing_when_already_on_target():
    controller, stroke, body, steps = _settling_controller(lag=np.array([0.03, 0.001, 0.0]))
    before, after = controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    assert steps == []
    assert before == after == pytest.approx(np.hypot(0.03, 0.001))


def test_stroke_end_settle_never_presses_the_blade_down_into_the_rack():
    # The blade rode up 1.3 cm on a dumpling: the settle may not chase it 1.3 cm down.
    controller, stroke, body, _ = _settling_controller(lag=np.array([0.0, 0.0, -0.013]))
    controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    commanded = [call.kwargs["position"] for call in controller.tool.ik.call_args_list]
    assert min(position[2] for position in commanded) >= stroke.end[2] - 0.002 - 1e-12


def test_optional_settling_backtracks_a_guarded_correction_before_moving():
    controller, stroke, body, steps = _settling_controller(lag=np.array([0.0, 0.0075, 0.0]))
    calls = []

    def guarded_ik(**kwargs):
        calls.append(kwargs["position"].copy())
        if len(calls) == 1:
            raise SkillFailed("axis-clipped joint segment would collide")
        return kwargs["position"].copy()

    controller.tool.ik.side_effect = guarded_ik
    controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    assert steps
    assert abs(calls[1][1] - stroke.end[1]) < abs(calls[0][1] - stroke.end[1])
    assert controller.metrics["settling_corrections"][0]["fraction"] == 0.5


def test_optional_settling_leaves_blocked_corrections_unexecuted():
    controller, stroke, body, steps = _settling_controller(lag=np.array([0.0, 0.0075, 0.0]))
    controller.tool.ik.side_effect = SkillFailed("fixture blocks correction")
    before, after = controller.settle_stroke_end(
        stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
    )
    assert not steps
    assert before == after
    assert controller.metrics["settling_corrections"][0]["fraction"] == 0
    assert "fixture blocks" in controller.metrics["settling_corrections"][0]["reason"]


def test_optional_settling_does_not_swallow_execution_failure():
    controller, stroke, body, _ = _settling_controller(lag=np.array([0.0, 0.0075, 0.0]))
    controller.tool.step.side_effect = SkillFailed("lost held scraper during execution")
    with pytest.raises(SkillFailed, match="lost held scraper"):
        controller.settle_stroke_end(
            stroke=stroke, body=body, base=np.zeros(3), arm=np.zeros(7), label="gather"
        )
