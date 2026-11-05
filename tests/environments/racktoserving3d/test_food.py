from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.food import (
    FINGER_NEWTONS_PER_ACTUATOR_UNIT,
    RackFoodController,
    _FoodMotion,
)
from hitl_pmp.environments.racktoserving3d.semantics import CircularSupport, SupportRegion
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
        support=Mock(return_value=SimpleNamespace(x_min=0.98, x_max=1.7, y_min=-0.5, y_max=0.5)),
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
        model=SimpleNamespace(
            nu=2,
            actuator=lambda index: SimpleNamespace(
                name=("robot_joint_7", "robot_fingers_actuator")[index]
            ),
            actuator_forcerange=np.array([[0.0, 0.0], [-5.0, 5.0]]),
        ),
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
        departing_from_contacts=Mock(side_effect=nullcontext),
        departing={},
        departing_food={},
    )
    # The planner (PyBullet) and the planned-path followers stand in: these tests are
    # about the scripted contact motions and the bookkeeping around them.
    helper._plan_pick = Mock(
        return_value=(SimpleNamespace(), pose.copy(), pose.copy(), np.array([0.7, 0, 0]))
    )
    helper._plan_place = Mock(
        return_value=(SimpleNamespace(), pose.copy(), pose.copy(), np.array([1.2, 0, 0.925]))
    )
    helper._follow = Mock()
    helper._lower_food = Mock()
    return helper


@pytest.mark.parametrize("name", ["unknown", "wiper_0", "rack_0"])
def test_rejects_nonfood_before_motion(*, food, name):
    with pytest.raises(SkillFailed, match="food"):
        food.pick(name=name)
    food.controllers.feedback.assert_not_called()


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
    food._follow.assert_not_called()


def test_unsupported_food_is_not_moved_even_if_old_location_was_on_tray(*, food):
    food.backend.food_available.return_value = False
    with pytest.raises(SkillFailed, match="pickable support"):
        food.pick(name="cube_0")
    food.shelf.hold.assert_not_called()
    food._follow.assert_not_called()


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


def _closing_limit(*, food) -> float:
    return float(food.shelf.model.actuator_forcerange[1, 1])


def test_pick_closes_force_limited_and_stays_limited_through_the_lift(*, food):
    food.backend.config = RackServingConfig(food_grip_force=20.0)
    food.backend.held.side_effect = [False, False, False, True, True]
    seen = []
    food.motion.line = Mock(
        side_effect=lambda **kwargs: seen.append((kwargs["label"], _closing_limit(food=food)))
    )
    food.pick(name="cube_0")
    limited = 20.0 / FINGER_NEWTONS_PER_ACTUATOR_UNIT
    assert seen == [("food_approach", 5.0), ("food_lift", pytest.approx(limited))]
    assert food.shelf.grip == food.backend.config.food_grip_command
    assert _closing_limit(food=food) == pytest.approx(limited)
    assert food.shelf.model.actuator_forcerange[1, 0] == -5.0  # opening untouched


def test_losing_the_food_restores_the_grippers_own_force_limit(*, food):
    food.backend.held.side_effect = [False, False, False, True, False]
    with pytest.raises(SkillFailed, match="bilateral"):
        food.pick(name="cube_0")
    assert _closing_limit(food=food) == 5.0


def test_release_restores_the_grippers_own_force_limit(*, food):
    food.backend.config = RackServingConfig(food_grip_force=20.0)
    food._limit_grip(newtons=20.0)
    food.shelf.grip = 1.0
    food.backend.held.side_effect = [True, True, False]
    food.backend.location.return_value = "OnTray"
    food.backend.body_geoms = Mock(return_value={0})
    food._placement_candidates = Mock(return_value=[np.array([1.2, 0, 0.94])])
    food._checked_arm = Mock(return_value=np.zeros(7))
    food.motion.check_arm_segment = Mock()
    food.shelf.cube_pre = lambda *, grasp: grasp.copy()
    released = []
    food.shelf.hold = Mock(
        side_effect=lambda *, count, label: released.append((label, _closing_limit(food=food)))
    )
    food.place(name="cube_0", destination="box_0")
    assert released[0][0] == "food_release"
    assert _closing_limit(food=food) == 5.0


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
    import mujoco

    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        f'<geom type="ellipsoid" size="{size[0]} {size[1]} {size[2]}" '
        f'pos="{center[0]} {center[1]} {center[2]}" '
        f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>'
        '<geom name="counter" type="box" size="0.5 0.5 0.45" pos="0 0 0.45"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
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
        model=model,
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


def test_pick_lifts_departing_from_the_contacts_the_grasp_closed_onto(*, food):
    food.backend.held.side_effect = [False, False, False, True, True]
    inside = []
    food.motion.departing_from_contacts = Mock(
        side_effect=lambda: _Recording(log=inside, label="departing")
    )
    food.motion.line = Mock(side_effect=lambda **kwargs: inside.append(kwargs["label"]))
    food.pick(name="cube_0")
    # The first scope covers planning and the planned reach (a pad may start on a
    # fixture); the lift has its own, measured from the closed grasp.
    assert inside == [
        "enter departing",
        "exit departing",
        "food_approach",
        "enter departing",
        "food_lift",
        "exit departing",
    ]


class _Recording:
    def __init__(self, *, log: list, label: str) -> None:
        self.log, self.label = log, label

    def __enter__(self) -> None:
        self.log.append(f"enter {self.label}")

    def __exit__(self, *exc: object) -> None:
        self.log.append(f"exit {self.label}")


class _PadOnWire:
    """Pad geom 1 resting on rack wire 10 when a lift starts; 2 is a non-pad link."""

    PAD, LINK, WIRE, OTHER_WIRE = 1, 2, 10, 11

    @staticmethod
    def motion(*, contacts: list[tuple[int, int, float]]) -> _FoodMotion:
        motion = _FoodMotion.__new__(_FoodMotion)
        scratch = SimpleNamespace(
            contact=[SimpleNamespace(geom1=a, geom2=b, dist=d) for a, b, d in contacts],
            ncon=len(contacts),
        )
        motion.shelf = SimpleNamespace(
            ik_data=scratch,
            robot_geoms={_PadOnWire.PAD, _PadOnWire.LINK},
            pad_geoms={_PadOnWire.PAD},
            cupboard_geoms=set(),
            box_geoms={_PadOnWire.WIRE, _PadOnWire.OTHER_WIRE},
            wiper_geoms=[],
        )
        motion.food = SimpleNamespace(
            backend=SimpleNamespace(cube_names=(), body_geoms=Mock(return_value=set())),
            carried=None,
            target_geoms=set(),
        )
        motion.departing = {}
        return motion

    @staticmethod
    def check(*, motion: _FoodMotion) -> None:
        with patch.object(
            RackScraper,
            "check_configuration",
            side_effect=SkillFailed("Scraper motion would collide"),
        ):
            motion.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)


def test_pad_fixture_contacts_record_only_pads_against_fixtures():
    motion = _PadOnWire.motion(
        contacts=[(10, 1, -1e-5), (1, 10, -2e-5), (2, 10, -1e-5), (1, 2, -1e-3)]
    )
    assert motion.pad_fixture_contacts() == {(1, 10): -2e-5}


@pytest.mark.parametrize(
    "now,passes",
    [
        ([(1, 10, -1e-5)], True),  # the start contact, receding
        ([(1, 10, 0.002)], True),  # within margin, further away
        ([(1, 10, -5e-4)], False),  # the start contact, deepening
        ([(1, 11, -1e-5)], False),  # a new pad-fixture contact
        ([(1, 10, -1e-5), (2, 11, -1e-5)], False),  # plus a new arm contact
    ],
)
def test_lift_tolerates_only_start_contacts_that_do_not_deepen(*, now, passes):
    motion = _PadOnWire.motion(contacts=now)
    motion.departing = {(1, 10): -2e-5}
    if passes:
        _PadOnWire.check(motion=motion)
    else:
        with pytest.raises(SkillFailed, match="would collide"):
            _PadOnWire.check(motion=motion)


def test_without_a_departure_every_fixture_contact_still_fails():
    motion = _PadOnWire.motion(contacts=[(1, 10, -1e-5)])
    with pytest.raises(SkillFailed, match="would collide"):
        _PadOnWire.check(motion=motion)


def test_pick_regrasps_food_that_moved_before_the_gripper_closes(*, food):
    food.backend.held.side_effect = [False, False, False, True, True]
    events = []
    moved, replanned_pre = np.eye(4), np.eye(4)
    moved[:3, 3], replanned_pre[:3, 3] = [1.2, 0.01, 0.93], [1.2, 0.01, 1.0]
    food._replan_grasp = Mock(side_effect=[(moved, replanned_pre), None])
    food.motion.line = Mock(side_effect=lambda **kwargs: events.append(kwargs["label"]))
    food.shelf.hold = Mock(side_effect=lambda *, count, label: events.append(label))
    food.pick(name="cube_0")
    assert events == [
        "food_open",
        "food_approach",
        "food_regrasp_retreat",
        "food_regrasp_pre",
        "food_regrasp",
        "food_close",
        "food_lift",
    ]
    second = food._replan_grasp.call_args_list[1].kwargs
    np.testing.assert_allclose(second["grasp"], moved)


class _Regrasp:
    """Food planned at yaw 0, observed at the grasp pose at some later yaw."""

    @staticmethod
    def pose(*, yaw_deg: float, x: float = 1.2) -> np.ndarray:
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        return Frames.pose(rotation=rotation, position=[x, 0, 0.93])

    @staticmethod
    def setup(*, food, observed: np.ndarray) -> list[np.ndarray]:
        food._pose = Mock(return_value=observed)
        food.motion.rotation_error = RackScraper.rotation_error
        far, near = _Regrasp.pose(yaw_deg=90), _Regrasp.pose(yaw_deg=10)
        food._grasp_candidates = Mock(return_value=[far, near])
        food.shelf.cube_pre = lambda *, grasp: grasp.copy()
        food._check_descent = Mock()
        return [far, near]


def test_regrasp_keeps_the_grasp_when_the_food_has_not_moved(*, food):
    _Regrasp.setup(food=food, observed=_Regrasp.pose(yaw_deg=1.0, x=1.201))
    replanned = food._replan_grasp(
        name="cube_0",
        grasp=_Regrasp.pose(yaw_deg=0),
        base=np.zeros(3),
        planned=_Regrasp.pose(yaw_deg=0),
    )
    assert replanned is None
    food._grasp_candidates.assert_not_called()


def test_regrasp_follows_a_rotated_food_with_the_nearest_reachable_yaw(*, food):
    far, near = _Regrasp.setup(food=food, observed=_Regrasp.pose(yaw_deg=13.0))
    grasp, pre = food._replan_grasp(
        name="cube_0",
        grasp=_Regrasp.pose(yaw_deg=0),
        base=np.zeros(3),
        planned=_Regrasp.pose(yaw_deg=0),
    )
    np.testing.assert_allclose(grasp, near)
    assert food._check_descent.call_args.kwargs["base"].tolist() == [0, 0, 0]


def test_regrasp_skips_a_yaw_unreachable_from_the_current_stance(*, food):
    far, near = _Regrasp.setup(food=food, observed=_Regrasp.pose(yaw_deg=13.0))
    food._check_descent = Mock(side_effect=[SkillFailed("arm hits counter"), None])
    grasp, _ = food._replan_grasp(
        name="cube_0",
        grasp=_Regrasp.pose(yaw_deg=0),
        base=np.zeros(3),
        planned=_Regrasp.pose(yaw_deg=0),
    )
    np.testing.assert_allclose(grasp, far)


def test_pick_does_not_repurpose_scraper_transform(*, food):
    food.backend.held.side_effect = [False, False, False, True, True]
    food.pick(name="cube_0")
    assert food.controllers.held_food == food.shelf.held_cube == "cube_0"
    assert food.shelf.in_hand is None
    assert food.shelf.grip == food.backend.config.food_grip_command
    assert all(not call.kwargs["holding"] for call in food.motion.line.call_args_list)


def test_placement_avoids_occupied_footprints(*, food):
    food.backend.support = Mock(
        return_value=SupportRegion(
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
    candidates = food._placement_candidates(name="cube_0", destination="box_0")
    assert len(candidates) == 14
    for target in candidates:
        assert target[0] - half_size[0] >= 1.37 + 0.02 - 1e-9
        assert target[0] + half_size[0] <= 1.63 - 0.02 + 1e-9
        assert np.any(np.abs(target - occupied) > 2 * half_size + 0.005)


def test_rack_put_back_tries_the_shared_layout_draws_in_order(*, food):
    rack = SupportRegion(x_min=1.37, x_max=1.63, y_min=-0.1875, y_max=0.1875, top_z=0.925)
    food.backend.support = Mock(return_value=rack)
    half_size = np.array([0.015, 0.0225, 0.0125])
    centre = np.array([1.2, 0, 0.925])  # the fixture's live pose
    food.backend.corners = Mock(return_value=np.array([centre - half_size, centre + half_size]))
    draws = [(1.45, -0.1), (1.52, 0.08), (1.41, 0.13)]
    food.backend.rack_put_back_draws = Mock(return_value=draws)
    candidates = food._placement_candidates(name="cube_0", destination="rack_0")
    food.backend.rack_put_back_draws.assert_called_once_with(
        name="cube_0", count=food.RACK_PUT_BACK_DRAWS
    )
    assert len(candidates) == len(draws)
    for target, (x, y) in zip(candidates, draws, strict=True):
        np.testing.assert_allclose(target, [x, y, rack.top_z + half_size[2] + 0.003])


def test_no_free_placement_retains_grasp(*, food):
    food.backend.held.return_value = True
    food.shelf.grip = 1.0
    food._placement_candidates = Mock(side_effect=SkillFailed("No unoccupied food placement"))
    del food._plan_place  # the real one, which asks for the placements first
    with pytest.raises(SkillFailed, match="unoccupied"):
        food.place(name="cube_0", destination="rack_0")
    food.shelf.hold.assert_not_called()
    assert food.shelf.grip == 1.0


@pytest.mark.parametrize("gap,blocked", [(0.004, False), (-0.002, True)])
def test_carried_food_collides_by_ellipsoid_shape_not_bounding_box(*, gap, blocked):
    import mujoco

    size = np.array([0.015, 0.0225, 0.0125])
    # Offset along the diagonal: the bounding boxes overlap there even with a real gap.
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    reach = 1.0 / np.sqrt((direction[0] / size[0]) ** 2 + (direction[1] / size[1]) ** 2)
    offset = (2 * reach + gap) * direction
    assert np.all(np.abs(offset[:2]) < 2 * size[:2])
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        f'<geom type="ellipsoid" size="{size[0]} {size[1]} {size[2]}" pos="0 0 1"/>'
        f'<geom type="ellipsoid" size="{size[0]} {size[1]} {size[2]}" '
        f'pos="{offset[0]} {offset[1]} 1"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shelf = SimpleNamespace(
        data=data,
        ik_data=data,
        model=model,
        cupboard_geoms=set(),
        box_geoms=set(),
        wiper_geoms=set(),
        robot_geoms=set(),
        pad_geoms=set(),
        site_pose=Mock(return_value=np.eye(4)),
        site_at=Mock(return_value=np.eye(4)),
    )
    motion = _FoodMotion.__new__(_FoodMotion)
    motion.shelf = shelf
    motion.food = SimpleNamespace(
        carried="cube_0",
        target_geoms={0},
        backend=SimpleNamespace(
            cube_names=("cube_0", "cube_1"),
            body_geoms=lambda *, name: {0} if name == "cube_0" else {1},
        ),
    )
    with patch.object(RackScraper, "check_configuration"):
        if blocked:
            with pytest.raises(SkillFailed, match="Carried food would collide"):
                motion.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)
        else:
            motion.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)


def test_food_grasps_close_across_the_short_sides_not_the_tips(*, food):
    food._pose = Mock(return_value=Frames.pose(rotation=np.eye(3), position=[1.55, 0, 0.94]))
    faces = []

    def cube_grasp(*, cube, centre, lean_deg, face, sign):
        faces.append(face)
        closing = np.eye(3)[:, face]
        toward = sign * np.array([-closing[1], closing[0], 0])
        return Frames.pose(
            rotation=Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean_deg),
            position=centre,
        )

    food.shelf.cube_grasp = cube_grasp
    assert food._grasp_candidates(name="cube_0")
    assert set(faces) == {0}


def test_grasps_prefer_nearest_counter_edge(*, food):
    food.backend.support = Mock(
        return_value=SimpleNamespace(
            x_min=0.98,
            x_max=1.7,
            y_min=-0.5,
            y_max=0.5,
        )
    )
    # Fat-side grasps close along the food's x and lean in y: put it near the +y edge.
    food._pose = Mock(return_value=Frames.pose(rotation=np.eye(3), position=[1.3, 0.4, 0.94]))

    def cube_grasp(*, cube, centre, lean_deg, face, sign):
        closing = np.eye(3)[:, face]
        toward = sign * np.array([-closing[1], closing[0], 0])
        return Frames.pose(
            rotation=Frames.leaned(closing=closing, lean_toward=toward, lean_deg=lean_deg),
            position=centre,
        )

    food.shelf.cube_grasp = cube_grasp
    grasps = food._grasp_candidates(name="cube_0")
    assert grasps[0][1, 2] < 0
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
        return_value=SupportRegion(x_min=1.2, x_max=1.6, y_min=-0.2, y_max=0.2, top_z=0.925)
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


def _round_plate_food(*, food):
    plate = CircularSupport(center_x=1.2325, center_y=0.0, radius=0.13, top_z=0.912)
    food.backend.support = Mock(return_value=plate)
    half_size = np.array([0.015, 0.0225, 0.0125])
    food.backend.corners = Mock(
        side_effect=lambda *, name: (
            np.array([
                np.array([1.2, 0, 0.925]) - half_size,
                np.array([1.2, 0, 0.925]) + half_size,
            ])
            if name == "cube_0"
            else np.array([[5, 5, 5], [5.01, 5.01, 5.01]])
        )
    )
    return plate, half_size


def test_round_plate_placements_keep_the_whole_footprint_inside_the_circle(*, food):
    plate, half_size = _round_plate_food(food=food)
    candidates = food._placement_candidates(name="cube_0", destination="box_0")
    assert len(candidates) >= 5
    for target in candidates:
        corners = [
            (target[0] + sx * half_size[0], target[1] + sy * half_size[1])
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
        assert plate.contains_xy(points=corners, margin=0.02)
        assert target[2] - half_size[2] == pytest.approx(0.912 + 0.003)


def test_round_plate_lowering_rejects_a_footprint_leaving_the_circle(*, food):
    plate, _ = _round_plate_food(food=food)
    food.backend.held.return_value = True
    centre = np.array([1.2325 + 0.09, 0.09, 0.95])
    food._pose = lambda **kwargs: Frames.pose(rotation=np.eye(3), position=centre.copy())
    food.backend.corners = lambda **kwargs: np.array([
        centre - [0.015, 0.0225, 0.0125],
        centre + [0.015, 0.0225, 0.0125],
    ])
    assert plate.x_min < centre[0] < plate.x_max and plate.y_min < centre[1] < plate.y_max
    with pytest.raises(SkillFailed, match="no longer fits"):
        RackFoodController._lower_food(
            food,
            name="cube_0",
            destination="box_0",
            target_xy=centre[:2],
            rotation=np.eye(3),
        )
    food.motion.line.assert_not_called()
