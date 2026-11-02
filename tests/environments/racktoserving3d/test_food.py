from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.food import RackFoodController, _FoodMotion
from hitl_pmp.environments.racktoserving3d.tool import RackScraper
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import Frames, SkillFailed


@pytest.fixture
def food():
    pose = Frames.pose(rotation=np.eye(3), position=[1.2, 0, 1.1])
    backend = SimpleNamespace(
        config=RackServingConfig(),
        cube_names=("cube_0", "cube_1"),
        sync_clock=Mock(),
        ledger=SimpleNamespace(ready=True),
        held=Mock(return_value=False),
        location=Mock(return_value="OnTray"),
        food_available=Mock(return_value=True),
        core=Mock(
            return_value=SimpleNamespace(
                get_object=Mock(
                    return_value=SimpleNamespace(
                        get_pose=Mock(
                            return_value=(np.array([1.2, 0, 0.925]), np.array([1.0, 0, 0, 0]))
                        )
                    )
                )
            )
        ),
    )
    shelf = SimpleNamespace(
        grip=0.0,
        in_hand=None,
        held_cube=None,
        hold=Mock(),
        site_pose=Mock(return_value=pose),
    )
    controllers = SimpleNamespace(
        backend=backend,
        shelf=shelf,
        tool=SimpleNamespace(line=Mock()),
        held_food=None,
        feedback=Mock(side_effect=nullcontext),
    )
    helper = RackFoodController(controllers=controllers)
    helper.motion = SimpleNamespace(
        line=Mock(),
        transit=Mock(),
        move_base=Mock(),
        plan_base=Mock(),
        check_arm_clearance=Mock(),
    )
    helper._plan_pick = Mock(return_value=(pose.copy(), pose.copy(), np.array([0.7, 0, 0])))
    helper._approach = Mock()
    helper._lower_food = Mock()
    return helper


@pytest.mark.parametrize("name", ["unknown", "wiper_0", "rack_0"])
def test_rejects_nonfood_before_motion(*, food, name):
    with pytest.raises(SkillFailed, match="food"):
        food.pick(name=name)
    food.controllers.feedback.assert_not_called()


def test_not_ready_preserves_state(*, food):
    food.backend.ledger.ready = False
    with pytest.raises(SkillFailed, match="ready"):
        food.pick(name="cube_0")
    food.shelf.hold.assert_not_called()


def test_occupied_hand_is_not_opened(*, food):
    food.backend.held.return_value = True
    food.shelf.grip = 1.0
    with pytest.raises(SkillFailed, match="empty"):
        food.pick(name="cube_0")
    assert food.shelf.grip == 1.0


def test_no_reachable_grasp_does_not_move(*, food):
    food._plan_pick.side_effect = SkillFailed("no collision-free food grasp")
    with pytest.raises(SkillFailed, match="collision-free"):
        food.pick(name="cube_0")
    food.shelf.hold.assert_not_called()
    food._approach.assert_not_called()


def test_unsupported_food_is_not_moved_even_if_old_location_was_on_tray(*, food):
    food.backend.food_available.return_value = False
    with pytest.raises(SkillFailed, match="pickable support"):
        food.pick(name="cube_0")
    food.shelf.hold.assert_not_called()
    food._approach.assert_not_called()


def test_failed_contact_does_not_claim_or_lift_food(*, food):
    with pytest.raises(SkillFailed, match="bilateral"):
        food.pick(name="cube_0")
    assert food.controllers.held_food is None
    assert all("lift" not in call.kwargs["label"] for call in food.motion.line.call_args_list)


def test_food_closure_uses_configured_calibration_without_changing_grasp_semantics(*, food):
    food.backend.config = RackServingConfig(food_grip_command=0.85)
    with pytest.raises(SkillFailed, match="bilateral"):
        food.pick(name="cube_0")
    assert food.shelf.grip == 0.85
    assert food.controllers.held_food is None


def test_unknown_place_destination_retains_food(*, food):
    food.backend.held.return_value = True
    food.shelf.grip = 1.0
    with pytest.raises(SkillFailed, match="destination"):
        food.place(name="cube_0", destination="counter")
    assert food.shelf.grip == 1.0
    food.controllers.feedback.assert_not_called()


def test_place_requires_live_grasp_not_bookkeeping(*, food):
    food.controllers.held_food = "cube_0"
    with pytest.raises(SkillFailed, match="held"):
        food.place(name="cube_0", destination="rack_0")
    food.controllers.feedback.assert_not_called()


@pytest.mark.parametrize("depth,blocked", [(0.0, False), (0.006, True)])
def test_payload_uses_ellipsoid_support_not_bounding_box(*, depth, blocked):
    rotation = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    size = np.array([0.015, 0.0225, 0.0125])
    center = [0, 0, 0.9 + np.linalg.norm(rotation[2] * size) - depth]
    data = SimpleNamespace(
        geom_xpos=np.array([center, [0, 0, 0.45]]),
        geom_xmat=np.array([rotation.reshape(9), np.eye(3).reshape(9)]),
        contact=[],
        ncon=0,
    )
    shelf = SimpleNamespace(
        data=data,
        ik_data=data,
        cupboard_geoms={1},
        box_geoms=set(),
        wiper_geoms=set(),
        robot_geoms=set(),
        pad_geoms=set(),
        site_pose=Mock(return_value=np.eye(4)),
        site_at=Mock(return_value=np.eye(4)),
        model=SimpleNamespace(
            geom_size=np.array([size, [0.5, 0.5, 0.45]]),
            geom_type=np.array([4, 6]),
            geom=Mock(return_value=SimpleNamespace(name="counter")),
        ),
    )
    motion = _FoodMotion.__new__(_FoodMotion)
    motion.shelf = shelf
    motion.food = SimpleNamespace(
        carried="cube_0",
        target_geoms={0},
        backend=SimpleNamespace(cube_names=("cube_0",), body_geoms=Mock(return_value={0})),
    )
    with patch.object(RackScraper, "check_configuration"):
        if blocked:
            with pytest.raises(SkillFailed, match="Carried food would collide"):
                motion.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)
        else:
            motion.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)


def test_lost_food_clears_bookkeeping_without_opening_hand(*, food):
    food.shelf.grip = 1.0
    food.carried = food.controllers.held_food = food.shelf.held_cube = "cube_0"
    with pytest.raises(SkillFailed, match="bilateral"):
        food._require_held(name="cube_0")
    assert food.carried is food.controllers.held_food is food.shelf.held_cube is None
    assert food.shelf.grip == 1.0


def test_successful_pick_requires_contact_after_lift(*, food):
    food.backend.held.side_effect = [False, False, False, True, False]
    with pytest.raises(SkillFailed, match="bilateral"):
        food.pick(name="cube_0")
    assert food.motion.line.call_args.kwargs["label"] == "food_lift"
    assert food.controllers.held_food is None


def test_pick_does_not_repurpose_scraper_transform(*, food):
    food.backend.held.side_effect = [False, False, False, True, True]
    food.pick(name="cube_0")
    assert food.controllers.held_food == food.shelf.held_cube == "cube_0"
    assert food.shelf.in_hand is None
    assert food.shelf.grip == 0.8
    assert all(not call.kwargs["holding"] for call in food.motion.line.call_args_list)


def test_placement_avoids_occupied_footprints(*, food):
    food.backend.support = Mock(
        return_value=SimpleNamespace(
            x_min=1.37,
            x_max=1.63,
            y_min=-0.1875,
            y_max=0.1875,
            top_z=0.925,
        )
    )
    half_size = np.array([0.015, 0.0225, 0.0125])
    occupied = np.array([1.395, -0.155, 0.94])
    food.backend.corners = Mock(
        side_effect=lambda *, name: (
            np.array([
                np.array([1.2, 0, 0.925]) - half_size,
                np.array([1.2, 0, 0.925]) + half_size,
            ])
            if name == "cube_0"
            else np.array([occupied - half_size, occupied + half_size])
        )
    )
    candidates = food._placement_candidates(name="cube_0", destination="rack_0")
    assert len(candidates) == 14
    for target in candidates:
        assert target[0] - half_size[0] >= 1.37 + 0.02 - 1e-9
        assert target[0] + half_size[0] <= 1.63 - 0.02 + 1e-9
        assert np.any(np.abs(target - occupied) > 2 * half_size + 0.005)


def test_no_free_placement_retains_grasp(*, food):
    food.backend.held.return_value = True
    food.shelf.grip = 1.0
    food._placement_candidates = Mock(side_effect=SkillFailed("No unoccupied food placement"))
    with pytest.raises(SkillFailed, match="unoccupied"):
        food.place(name="cube_0", destination="rack_0")
    food.shelf.hold.assert_not_called()
    assert food.shelf.grip == 1.0


@pytest.mark.parametrize(
    "approach,expected", [([-1.0, 0, -1], [2.0, 0, np.pi]), ([0, 1.0, -1], [1.5, -0.5, np.pi / 2])]
)
def test_stances_face_approach_instead_of_always_front(*, food, approach, expected):
    bases = food._base_candidates(centre=np.array([1.5, 0, 0.94]), approach=np.array(approach))
    np.testing.assert_allclose(bases[0], expected)


def test_grasps_prefer_nearest_counter_edge(*, food):
    food.backend.support = Mock(
        return_value=SimpleNamespace(
            x_min=0.98,
            x_max=1.7,
            y_min=-0.5,
            y_max=0.5,
        )
    )
    food._pose = Mock(return_value=Frames.pose(rotation=np.eye(3), position=[1.55, 0, 0.94]))

    def cube_grasp(*, cube, centre, lean_deg, face, sign):
        closing = np.eye(3)[:, face]
        toward = sign * np.array([-closing[1], closing[0], 0])
        return Frames.pose(
            rotation=Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean_deg),
            position=centre,
        )

    food.shelf.cube_grasp = cube_grasp
    grasps = food._grasp_candidates(name="cube_0")
    assert grasps[0][0, 2] < 0
    assert grasps[0][2, 2] < 0
    np.testing.assert_allclose(grasps[0][:3, 2], grasps[1][:3, 2])
    np.testing.assert_allclose(grasps[0][:3, 0], -grasps[1][:3, 0])


def test_ik_rejects_joint_limit_solution_before_motion(*, food):
    food.shelf.joints = np.arange(7)
    food.shelf.model = SimpleNamespace(
        jnt_limited=np.ones(7, dtype=bool),
        jnt_range=np.tile([-2.0, 2.0], (7, 1)),
    )
    food.shelf.ik = Mock(return_value=(np.full(7, -2.0), 0.0))
    food.shelf.site_at = Mock(return_value=np.eye(4))
    food.motion.rotation_error = Mock(return_value=0.0)
    food.motion.check_configuration = Mock()
    with pytest.raises(SkillFailed, match="joint limit"):
        food._checked_arm(pose=np.eye(4), base=np.zeros(3))
    food.motion.check_configuration.assert_not_called()


@pytest.mark.parametrize("destination,location", [("rack_0", "OnRack"), ("box_0", "OnTray")])
def test_place_verifies_release_and_observed_destination(*, food, destination, location):
    food.backend.held.side_effect = [True, True, False]
    food.backend.location.return_value = location
    food.backend.body_geoms = Mock(return_value={0})
    food._placement_candidates = Mock(return_value=[np.array([1.4, 0, 0.94])])
    food._checked_arm = Mock(return_value=np.zeros(7))
    food._base_candidates = Mock(return_value=[np.array([0.8, 0, 0])])
    food.motion.check_arm_segment = Mock()
    food.shelf.cube_pre = lambda *, grasp: grasp.copy()
    food.place(name="cube_0", destination=destination)
    assert food.shelf.grip == 0.0
    assert food.carried is food.controllers.held_food is food.shelf.held_cube is None
    assert [call.kwargs["label"] for call in food.motion.line.call_args_list] == [
        "food_retreat",
    ]
    food._lower_food.assert_called_once()
    food.backend.location.assert_called_once_with(name="cube_0")


def test_place_does_not_release_after_slip(*, food):
    food.shelf.grip = 0.8
    food.backend.held.side_effect = [True, False]
    food._placement_candidates = Mock(return_value=[np.array([1.4, 0, 0.94])])
    food._checked_arm = Mock(return_value=np.zeros(7))
    food._base_candidates = Mock(return_value=[np.array([0.8, 0, 0])])
    food.motion.check_arm_segment = Mock()
    food.shelf.cube_pre = lambda *, grasp: grasp.copy()
    with pytest.raises(SkillFailed, match="bilateral"):
        food.place(name="cube_0", destination="rack_0")
    assert food.shelf.grip == 0.8
    food.shelf.hold.assert_not_called()


def test_robot_skills_never_write_object_poses_or_scene_geometry():
    import ast
    import inspect

    from hitl_pmp.environments.racktoserving3d import food as module

    tree = ast.parse(inspect.getsource(module))
    forbidden = {"set_pose", "set_velocity", "_commit", "reset_to_seed", "initialize_rest"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and node.attr in {"qpos", "qvel", "geom_size", "geom_contype", "geom_conaffinity"}
        for node in ast.walk(tree)
    )


def test_food_base_detour_is_fully_checked_before_execution():
    motion = _FoodMotion.__new__(_FoodMotion)
    motion.shelf = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(3)),
        ba=np.arange(3),
        arm_reference=np.zeros(7),
    )
    motion.food = SimpleNamespace(
        backend=SimpleNamespace(
            support=Mock(
                return_value=SimpleNamespace(y_min=-0.5, y_max=0.5),
            )
        )
    )
    events = []
    calls = []

    def drive(*, target, label):
        calls.append(target.copy())
        if len(calls) == 1:
            raise SkillFailed(f"{label}: no safe base transit: blocked")
        events.append("drive")

    motion.check_configuration = Mock(side_effect=lambda **kwargs: events.append("check"))
    with patch.object(RackScraper, "move_base", side_effect=drive):
        motion.move_base(target=np.array([2.0, 0, np.pi]), label="test")
    assert len(calls) == 5
    assert events[-4:] == ["drive"] * 4
    assert all(event == "check" for event in events[:-4])
    assert abs(calls[1][1]) > 0.5


def test_food_base_does_not_retry_execution_failures():
    motion = _FoodMotion.__new__(_FoodMotion)
    with (
        patch.object(RackScraper, "move_base", side_effect=SkillFailed("lost food")) as drive,
        pytest.raises(SkillFailed, match="lost food"),
    ):
        motion.move_base(target=np.zeros(3), label="test")
    drive.assert_called_once()


def test_fresh_place_helper_rejects_blocked_base_route_before_moving(*, food):
    food.shelf.grip = 0.8
    food.backend.held.side_effect = [True, True, False]
    food.backend.location.return_value = "OnRack"
    food.backend.body_geoms = Mock(return_value={0})
    food._placement_candidates = Mock(return_value=[np.array([1.4, 0, 0.94])])
    food._checked_arm = Mock(return_value=np.zeros(7))
    blocked, reachable = np.array([0.9, -0.15, 0]), np.array([0.8, -0.15, 0])
    food._base_candidates = Mock(return_value=[blocked, reachable])
    food.motion.check_arm_segment = Mock()
    food.motion.plan_base = Mock(side_effect=[SkillFailed("counter leg blocks route"), [reachable]])
    food.shelf.cube_pre = lambda *, grasp: grasp.copy()
    assert food.carried is None
    food.place(name="cube_0", destination="rack_0")
    assert food.motion.plan_base.call_count == 2
    food._approach.assert_called_once()
    np.testing.assert_allclose(food._approach.call_args.kwargs["base"], reachable)


def test_all_blocked_placement_routes_retain_food_without_robot_motion(*, food):
    food.shelf.grip = 0.8
    food.backend.held.return_value = True
    food._placement_candidates = Mock(return_value=[np.array([1.4, 0, 0.94])])
    food._checked_arm = Mock(return_value=np.zeros(7))
    food._base_candidates = Mock(return_value=[np.array([0.9, -0.15, 0])])
    food.motion.check_arm_segment = Mock()
    food.motion.plan_base = Mock(side_effect=SkillFailed("counter leg blocks route"))
    food.shelf.cube_pre = lambda *, grasp: grasp.copy()
    with pytest.raises(SkillFailed, match="counter leg"):
        food.place(name="cube_0", destination="rack_0")
    assert food.shelf.grip == 0.8
    food._approach.assert_not_called()
    food.motion.line.assert_not_called()
    food.shelf.hold.assert_not_called()


def test_surface_staging_preserves_checked_joint_branch(*, food):
    food.carried = "cube_0"
    checked_arm = np.arange(7, dtype=float) * 0.1
    food._checked_arm = Mock(return_value=checked_arm)
    food.motion.transit_arm = Mock()
    food.shelf.data = SimpleNamespace(qpos=np.zeros(10))
    food.shelf.ba = np.arange(3)
    food.shelf.site_at = Mock(return_value=np.eye(4))
    RackFoodController._approach(
        food,
        pose=np.eye(4),
        base=np.array([0.8, -0.15, 0]),
        label="food_place",
    )
    food.motion.transit_arm.assert_called_once()
    np.testing.assert_array_equal(food.motion.transit_arm.call_args.kwargs["arm"], checked_arm)
    food.motion.transit.assert_not_called()


def test_lowering_reobserves_food_extent_instead_of_using_old_grasp_offset(*, food):
    centre = np.array([1.4, 0.0, 0.98])
    half_height = [0.0125]
    food.backend.held.return_value = True
    food._pose = lambda **kwargs: Frames.pose(rotation=np.eye(3), position=centre.copy())
    food.backend.corners = lambda **kwargs: np.array([
        centre - [0.015, 0.02, half_height[0]],
        centre + [0.015, 0.02, half_height[0]],
    ])
    food.backend.support = Mock(
        return_value=SimpleNamespace(x_min=1.2, x_max=1.6, y_min=-0.2, y_max=0.2, top_z=0.925)
    )
    food.shelf.site_pose = lambda: Frames.pose(
        rotation=np.eye(3), position=centre + [0.0, 0.0, 0.03]
    )

    def moved(**kwargs):
        centre[:] += kwargs["position"] - food.shelf.site_pose()[:3, 3]
        half_height[0] = 0.02

    food.motion.line.side_effect = moved
    RackFoodController._lower_food(
        food, name="cube_0", destination="rack_0", target_xy=np.array([1.4, 0]), rotation=np.eye(3)
    )
    assert centre[2] - half_height[0] == pytest.approx(0.928)
    assert food.motion.line.call_count >= 2


def test_lowering_failure_does_not_open_gripper(*, food):
    food.shelf.grip = 0.8
    food.backend.held.return_value = False
    with pytest.raises(SkillFailed, match="bilateral"):
        RackFoodController._lower_food(
            food, name="cube_0", destination="rack_0", target_xy=np.zeros(2), rotation=np.eye(3)
        )
    assert food.shelf.grip == 0.8
    food.motion.line.assert_not_called()


def test_food_ik_rejects_small_forearm_clearance(*, food):
    food.carried = "cube_0"
    food.shelf.joints = np.arange(7)
    food.shelf.model = SimpleNamespace(
        jnt_limited=np.ones(7, dtype=bool),
        jnt_range=np.tile([-2.0, 2.0], (7, 1)),
    )
    food.shelf.ik = Mock(return_value=(np.zeros(7), 0.0))
    food.shelf.site_at = Mock(return_value=np.eye(4))
    food.motion.rotation_error = Mock(return_value=0.0)
    food.motion.check_configuration = Mock()
    food.motion.check_arm_clearance = Mock(side_effect=SkillFailed("tracking clearance"))
    with pytest.raises(SkillFailed, match="tracking clearance"):
        food._checked_arm(pose=np.eye(4), base=np.zeros(3))


@pytest.mark.parametrize(
    "separation,blocked", [(0.001, True), (0.005, False), (float("nan"), True)]
)
def test_tracking_margin_applies_to_arm_not_gripper(*, separation, blocked):
    motion = _FoodMotion.__new__(_FoodMotion)
    distance = Mock(return_value=separation)
    motion.shelf = SimpleNamespace(
        model=SimpleNamespace(
            geom_contype=np.ones(3),
            geom_conaffinity=np.ones(3),
            geom_rbound=np.ones(3),
        ),
        ik_data=SimpleNamespace(geom_xpos=np.zeros((3, 3))),
        site_at=Mock(),
        robot_geoms={0, 1},
        gripper_geoms={1},
        cupboard_geoms={2},
        box_geoms=set(),
        mujoco=SimpleNamespace(mj_geomDistance=distance),
    )
    if blocked:
        with pytest.raises(SkillFailed, match="tracking clearance"):
            motion.check_arm_clearance(base=np.zeros(3), arm=np.zeros(7))
    else:
        motion.check_arm_clearance(base=np.zeros(3), arm=np.zeros(7))
    distance.assert_called_once()
    assert distance.call_args.args[2:4] == (0, 2)


def test_joint_staging_executes_planned_configuration_without_ik():
    motion = _FoodMotion.__new__(_FoodMotion)
    motion.shelf = SimpleNamespace(
        base_target=np.zeros(3),
        qa=np.arange(7),
        data=SimpleNamespace(qpos=np.zeros(7)),
        site_at=Mock(return_value=np.eye(4)),
    )
    target = np.full(7, 0.2)
    planner = SimpleNamespace(
        plan=Mock(return_value=[target]),
        delta=lambda *, start, target: target - start,
    )

    def robot_step(*, arm, label):
        motion.shelf.data.qpos[:] = arm

    motion.step = Mock(side_effect=robot_step)
    motion.check_arm_segment = Mock()
    motion.settle_hand = Mock()
    with patch("hitl_pmp.environments.racktoserving3d.food.ArmTransit", return_value=planner):
        motion.transit_arm(arm=target, label="test")
    np.testing.assert_array_equal(planner.plan.call_args.kwargs["target"], target)
    np.testing.assert_array_equal(motion.step.call_args.kwargs["arm"], target)
    np.testing.assert_array_equal(motion.settle_hand.call_args.kwargs["arm"], target)
