"""Conservative geometry proofs, state ownership, and controller parity."""

import importlib.util
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.types import PlanarCollisionBox, TossFeasibilityGeometry


def _geometry(*, barrier_length: float = 10.0) -> TossFeasibilityGeometry:
    return TossFeasibilityGeometry(
        robot_pose=(0.0, 0.0, 0.0),
        robot_size=(0.55, 0.55),
        bin_pose=(3.0, 0.0, 0.0),
        obstacles=(
            PlanarCollisionBox(
                name="barrier",
                center=(1.3, 0.0),
                width=0.06,
                height=barrier_length,
                yaw=0.0,
            ),
        ),
        sampling_x_bounds=(-2.5, 2.5),
        sampling_y_bounds=(-2.5, 2.5),
    )


def _params(*, distance: float, rotation: float = 0.0) -> np.ndarray:
    return np.array([distance, rotation, 300.0, 700.0])


@pytest.mark.parametrize(
    "distance,reason",
    [
        (2.5, None),
        (1.7, "target_collision:barrier"),
        (1.0, "separating_obstacle:barrier"),
    ],
)
def test_near_side_overlap_and_disconnected_clear_target(*, distance, reason) -> None:
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=_geometry(), params=_params(distance=distance)
        )
        == reason
    )


def test_finite_barrier_allows_a_route_around_it() -> None:
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=_geometry(barrier_length=2.0), params=_params(distance=1.0)
        )
        is None
    )


def test_sampling_bounds_are_not_hard_endpoint_bounds() -> None:
    geometry = _geometry().model_copy(update={"obstacles": ()})
    # Target x=-3 is outside the RRT sample range but a clear straight path exists.
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=6.0)
        )
        is None
    )


def test_outside_sample_range_endpoint_prevents_false_separator_proof() -> None:
    geometry = _geometry(barrier_length=6.0).model_copy(
        update={
            "bin_pose": (3.0, 4.0, 0.0),
            "robot_pose": (0.0, 4.0, 0.0),
        }
    )
    # Both endpoints are above the barrier. Ignoring them in the hull would reject.
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=1.0)
        )
        is None
    )


def test_robot_yaw_changes_footprint_clearance() -> None:
    geometry = _geometry().model_copy(update={"bin_pose": (0.96, 0.0, 0.0)})
    # Same target centre: a square fits when aligned, intersects when rotated.
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=0.0)
        )
        is None
    )
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=0.0, rotation=math.pi / 4)
        )
        == "target_collision:barrier"
    )


def test_rotated_barrier_separator_and_same_side_target() -> None:
    geometry = _geometry().model_copy(
        update={
            "bin_pose": (0.0, 3.0, math.pi / 2),
            "obstacles": (
                PlanarCollisionBox(
                    name="horizontal",
                    center=(0.0, 1.3),
                    width=10.0,
                    height=0.06,
                    yaw=0.0,
                ),
            ),
        }
    )
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=1.0)
        )
        == "separating_obstacle:horizontal"
    )
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=2.5)
        )
        is None
    )


def test_speed_release_and_input_are_untouched() -> None:
    geometry = _geometry()
    for speed, release in [(30.0, 450.0), (420.0, 840.0), (300.0, 700.0)]:
        params = np.array([2.5, 0.0, speed, release])
        original = params.copy()
        assert (
            TossParameterFeasibility.rejection_reason_from_geometry(
                geometry=geometry, params=params
            )
            is None
        )
        assert np.array_equal(params, original)


def test_float32_parameter_semantics_match_controller() -> None:
    params = _params(distance=2.00500004)
    rounded = params.astype(np.float32).astype(float)
    assert TossParameterFeasibility.rejection_reason_from_geometry(
        geometry=_geometry(), params=params
    ) == TossParameterFeasibility.rejection_reason_from_geometry(
        geometry=_geometry(), params=rounded
    )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.filterwarnings("error:overflow encountered in cast:RuntimeWarning")
def test_float32_overflow_is_unknown_geometry(*, index: int) -> None:
    params = _params(distance=2.5)
    params[index] = 1e300
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(geometry=_geometry(), params=params)
        is None
    )


def test_missing_snapshot_and_other_skills_are_accepted() -> None:
    env = Tossing3DEnvironment()
    toss = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
    )
    pick = GroundSkill(
        skill=Tossing3DSkills.PICK_CUBE,
        objects=(env.robot, env.cube, env.barrier, Tossing3DSides.robot),
    )
    assert (
        TossParameterFeasibility.rejection_reason(
            state=State(data={}), ground_skill=toss, params=_params(distance=1.0)
        )
        is None
    )
    assert (
        TossParameterFeasibility.rejection_reason(
            state=State(data={}), ground_skill=pick, params=np.zeros(0)
        )
        is None
    )
    assert KinderBackend.toss_feasibility_geometry(snapshot=object()) is None


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None,
    reason="KINDER geometry adapter dependency",
)
def test_unsupported_robot_snapshot_dtype_disables_gate() -> None:
    snapshot = MagicMock()
    snapshot.get_objects.return_value = (object(),)
    snapshot.__getitem__.return_value = np.zeros(22, dtype=np.int64)
    assert KinderBackend.toss_feasibility_geometry(snapshot=snapshot) is None
    snapshot.get_object_from_name.assert_not_called()


def test_float32_state_write_preserves_a_clear_near_contact_target() -> None:
    geometry = TossFeasibilityGeometry(
        robot_pose=(1.485786461830139, -1.4142135381698608, 0.7853981852531433),
        robot_size=(0.55, 0.55),
        robot_state_dtype="float32",
        bin_pose=(3.0, 0.0, 0.0),
        obstacles=(
            PlanarCollisionBox(
                name="near_contact",
                center=(1.979695200920105, -1.4142135381698608),
                width=0.010000011883676052,
                height=0.1,
                yaw=0.0,
            ),
        ),
        sampling_x_bounds=(-2.5, 2.5),
        sampling_y_bounds=(-2.5, 2.5),
    )
    params = _params(distance=2.0, rotation=math.pi / 4)
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(geometry=geometry, params=params)
        is None
    )
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry.model_copy(update={"robot_state_dtype": "float64"}), params=params
        )
        == "target_collision:near_contact"
    )


def test_separator_declines_an_obstacle_thinner_than_state_precision() -> None:
    geometry = _geometry().model_copy(
        update={
            "robot_state_dtype": "float32",
            "obstacles": (
                PlanarCollisionBox(
                    name="unresolved_thickness",
                    center=(1.3, 0.0),
                    width=1e-8,
                    height=10,
                    yaw=0,
                ),
            ),
        }
    )
    assert (
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=_params(distance=1.0)
        )
        is None
    )


@pytest.mark.skipif(
    importlib.util.find_spec("tomsgeoms2d") is None,
    reason="controller's optional geometry dependency",
)
def test_rectangle_rejections_agree_with_independent_pinned_collision_library() -> None:
    from tomsgeoms2d.structs import Rectangle
    from tomsgeoms2d.utils import geom2ds_intersect

    rng = np.random.default_rng(721)
    for _ in range(200):
        boxes = [
            PlanarCollisionBox(
                name=str(i),
                center=tuple(rng.uniform(-1, 1, 2)),
                width=float(rng.uniform(0.1, 2)),
                height=float(rng.uniform(0.1, 2)),
                yaw=float(rng.uniform(-math.pi, math.pi)),
            )
            for i in range(2)
        ]
        geoms = [
            Rectangle.from_center(*b.center, b.width, b.height, rotation_about_center=b.yaw)
            for b in boxes
        ]
        assert TossParameterFeasibility._strictly_overlap(  # noqa: SLF001
            first=boxes[0], second=boxes[1]
        ) == geom2ds_intersect(*geoms)


@pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_snapshot_adapter_reads_supplied_state_after_live_environment_changes() -> None:
    env = Tossing3DEnvironment()
    try:
        old_state = env.reset_to_seed(seed=125)
        toss = GroundSkill(
            skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
        )
        before = KinderBackend.toss_feasibility_geometry(snapshot=old_state.object_centric)
        assert before is not None and before.robot_size == (0.55, 0.55)
        robot = old_state.object_centric.get_object_from_name("robot")
        assert before.robot_state_dtype == str(old_state.object_centric[robot].dtype)
        assert any(o.name.startswith("collider:cuboid_barrier:") for o in before.obstacles)
        assert not any(o.name == "cube_0" for o in before.obstacles)
        safe = np.array([2.5, 0.0, 360.0, 500.0])
        assert (
            TossParameterFeasibility.rejection_reason(
                state=old_state, ground_skill=toss, params=safe
            )
            is None
        )
        blocked = np.array([1.35, 0.0, 140.0, 792.0])
        reason = TossParameterFeasibility.rejection_reason(
            state=old_state, ground_skill=toss, params=blocked
        )
        assert reason is not None
        env.reset_to_seed(seed=17)
        assert KinderBackend.toss_feasibility_geometry(snapshot=old_state.object_centric) == before
        assert (
            TossParameterFeasibility.rejection_reason(
                state=old_state, ground_skill=toss, params=blocked
            )
            == reason
        )
        assert (
            TossParameterFeasibility.rejection_reason(
                state=old_state, ground_skill=toss, params=safe
            )
            is None
        )
    finally:
        env.close()
