"""A robot-side bin a practice pick dragged and rotated: reject, never raise.

Offline: the geometry is `test_toss_direction.py`'s copy of the live colliders and the
base planner is a stub that always finds a path, so every rejection here comes from
the geometry gate. The two moved-bin poses are the ones two 2026-09-25 practice runs
crashed on with `NoFeasibleTossDirectionError` (Model B cycle 6, Model A cycle 7); the
first one's four per-direction reasons are reproduced verbatim by this geometry.
"""

import math

import numpy as np
import pytest

from hitl_pmp.core.method.method import NoFeasibleParametersError
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend
from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import (
    NO_FEASIBLE_TOSS_DIRECTION_REJECTION,
    Tossing3DToss,
)
from hitl_pmp.environments.tossing3d.toss_direction import (
    TOSS_DIRECTIONS_DEG,
    NoFeasibleTossDirectionError,
    TossDirectionSelector,
    TossStandoffBand,
)
from hitl_pmp.environments.tossing3d.types import PlanarCollisionBox, TossFeasibilityGeometry
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import WIDE_TOSS_STANDOFF_BOUNDS
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

from .observations import state
from .test_toss_direction import _geometry

# (bin pose, robot pose, the standoff the run crashed on)
MODEL_B_CYCLE_6 = ((-0.008, -1.276, 2.741), (0.647, -1.63, -3.094), 2.5358096296804824)
MODEL_A_CYCLE_7 = ((0.129, -1.893, 2.819), (0.251, -1.201, -1.674), 2.5487)
CRASHES = pytest.mark.parametrize(
    "crash", [MODEL_B_CYCLE_6, MODEL_A_CYCLE_7], ids=["model-b-cycle-6", "model-a-cycle-7"]
)
# Model B's bin before the pick moved it: at the reset yaw, inside the practice region.
MODEL_B_RESET_BIN = ((0.098, -1.377, math.pi), (0.647, -1.63, -3.094))


def _moved_geometry(*, bin_pose, robot_pose) -> TossFeasibilityGeometry:
    return _geometry(bin_xy=bin_pose[:2], bin_yaw=bin_pose[2]).model_copy(
        update={"robot_pose": robot_pose}
    )


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.robot),
    )


@pytest.fixture
def live_scene(*, monkeypatch: pytest.MonkeyPatch):
    """A state whose snapshot resolves to `geometry`, with a planner that never fails."""

    def install(*, env: Tossing3DEnvironment, geometry: TossFeasibilityGeometry):
        snapshot = object()
        scene = state(
            env=env,
            bin_x=geometry.bin_pose[0],
            base_x=geometry.robot_pose[0],
            base_y=geometry.robot_pose[1],
            base_rot=geometry.robot_pose[2],
        ).model_copy(update={"object_centric": snapshot})

        def geometry_for(*, snapshot: object) -> TossFeasibilityGeometry:
            assert snapshot is scene.object_centric
            return geometry

        monkeypatch.setattr(KinderBackend, "toss_feasibility_geometry", geometry_for)
        monkeypatch.setattr(TossDirectionSelector, "base_plan_failure", lambda **kwargs: None)
        return scene

    TossDirectionSelector.clear_plan_cache()
    TossStandoffBand.clear_cache()
    yield install
    TossDirectionSelector.clear_plan_cache()
    TossStandoffBand.clear_cache()


def _some_direction_passes_the_gate(*, geometry: TossFeasibilityGeometry, standoff: float) -> bool:
    return any(
        TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=np.array([standoff, math.radians(direction)])
        )
        is None
        for direction in TOSS_DIRECTIONS_DEG
    )


def test_the_logged_reasons_are_reproduced_at_the_model_b_crash_pose(*, live_scene) -> None:
    bin_pose, robot_pose, standoff = MODEL_B_CYCLE_6
    geometry = _moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose)
    live_scene(env=Tossing3DEnvironment(), geometry=geometry)
    with pytest.raises(NoFeasibleTossDirectionError) as caught:
        TossDirectionSelector.select(geometry=geometry, snapshot=None, standoff=standoff)
    assert caught.value.reasons == {
        0: "separating_obstacle:collider:cuboid_barrier:9",
        90: "target_collision:collider:cuboid_barrier:9",
        180: "target_collision:collider:tossing_room:7",
        270: "outside_room",
    }


@CRASHES
def test_the_rejection_path_rejects_the_crash_standoff_instead_of_raising(
    *, crash, live_scene
) -> None:
    bin_pose, robot_pose, standoff = crash
    env = Tossing3DEnvironment()
    scene = live_scene(env=env, geometry=_moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose))
    reason = Tossing3DSkillProvider(env=env).parameter_rejection_reason(
        ground_skill=_toss(env=env), params=np.array([standoff, 200.0, 600.0]), state=scene
    )
    assert reason == NO_FEASIBLE_TOSS_DIRECTION_REJECTION


@CRASHES
def test_the_band_is_computed_from_the_moved_bins_live_pose(*, crash, live_scene) -> None:
    """The band a moved bin draws from ends below the crash standoff, and every
    standoff inside it has at least one direction the geometry gate admits."""
    bin_pose, robot_pose, standoff = crash
    env = Tossing3DEnvironment()
    geometry = _moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose)
    scene = live_scene(env=env, geometry=geometry)
    low, high = Tossing3DToss.standoff_bounds(ground_skill=_toss(env=env), state=scene)
    assert low == WIDE_TOSS_STANDOFF_BOUNDS[0]
    assert high < standoff
    assert high < WIDE_TOSS_STANDOFF_BOUNDS[1]
    for inside in np.linspace(low, high, 60):
        assert _some_direction_passes_the_gate(geometry=geometry, standoff=float(inside))
    assert not _some_direction_passes_the_gate(geometry=geometry, standoff=high + 0.01)


def test_a_bin_at_its_reset_pose_keeps_exactly_the_full_band(*, live_scene) -> None:
    """Unmoved bins draw from the same band, with the same rng stream, as before."""
    bin_pose, robot_pose = MODEL_B_RESET_BIN
    env = Tossing3DEnvironment()
    scene = live_scene(env=env, geometry=_moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose))
    assert (
        Tossing3DToss.standoff_bounds(ground_skill=_toss(env=env), state=scene)
        == WIDE_TOSS_STANDOFF_BOUNDS
    )


@CRASHES
def test_ees_samples_a_moved_bins_toss_without_raising(*, crash, live_scene) -> None:
    bin_pose, robot_pose, _ = crash
    env = Tossing3DEnvironment()
    geometry = _moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose)
    scene = live_scene(env=env, geometry=geometry)
    method = EesMethod(
        env=env, skill_provider=Tossing3DSkillProvider(env=env), seed=0, num_candidates=50
    )
    candidates = method.sample_parameter_candidates(
        ground_skill=_toss(env=env), state=scene, explore=True
    )
    assert len(candidates) == 50
    for candidate in candidates:
        choice = Tossing3DToss.choose_direction(state=scene, params=candidate)
        assert choice.direction_deg in TOSS_DIRECTIONS_DEG


def _boxed_in(*, geometry: TossFeasibilityGeometry) -> TossFeasibilityGeometry:
    """Pillars on every stand ray: no direction is reachable at any standoff."""
    x, y, _ = geometry.bin_pose
    slabs = tuple(
        PlanarCollisionBox(name=f"slab:{index}", center=center, width=w, height=h, yaw=0.0)
        for index, (center, w, h) in enumerate([
            ((x - 1.9, y), 1.6, 6.0),
            ((x + 1.9, y), 1.6, 6.0),
            ((x, y - 1.9), 6.0, 1.6),
            ((x, y + 1.9), 6.0, 1.6),
        ])
    )
    return geometry.model_copy(update={"obstacles": (*geometry.obstacles, *slabs)})


def test_a_bin_pose_with_no_feasible_toss_is_an_empty_pool_the_planner_replans_around(
    *, live_scene
) -> None:
    bin_pose, robot_pose = MODEL_B_RESET_BIN
    env = Tossing3DEnvironment()
    geometry = _boxed_in(geometry=_moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose))
    scene = live_scene(env=env, geometry=geometry)
    for standoff in np.linspace(*WIDE_TOSS_STANDOFF_BOUNDS, 30):
        assert not _some_direction_passes_the_gate(geometry=geometry, standoff=float(standoff))
    method = EesMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        num_candidates=4,
        max_proposals_per_candidate=5,
    )
    with pytest.raises(NoFeasibleParametersError) as caught:
        method.sample_parameter_candidates(ground_skill=_toss(env=env), state=scene, explore=True)
    assert caught.value.diagnostics.rejection_reasons == {NO_FEASIBLE_TOSS_DIRECTION_REJECTION: 20}


def test_executing_a_standoff_with_no_direction_still_raises(*, live_scene) -> None:
    """Only an accepted candidate reaches `compute_action`, so arriving there with no
    direction is an impossible state, and it stays loud."""
    bin_pose, robot_pose, standoff = MODEL_B_CYCLE_6
    env = Tossing3DEnvironment()
    scene = live_scene(env=env, geometry=_moved_geometry(bin_pose=bin_pose, robot_pose=robot_pose))
    with pytest.raises(NoFeasibleTossDirectionError):
        Tossing3DToss.compute_action(params=np.array([standoff, 200.0, 600.0]), state=scene)
