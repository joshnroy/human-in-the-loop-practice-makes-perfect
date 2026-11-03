"""Skills only propose robot poses the kitchen drive rule allows.

Measured on Della v3 practice (16648321): far tosses with standoff below ~1.45 m failed
345/871 times because their base pose lies past the mat edge, and picks of cubes past
x ~1.27 failed 149/161 times although the `reachable` flag promised them.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.drive_zone import KitchenDriveZone
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecyclingEnvironment
from hitl_pmp.environments.kitchen_recycling3d.skill_provider import (
    KitchenRecyclingSkillProvider,
)

EXPERIMENT = Path(__file__).parents[3] / "scripts" / "kitchen_recycling" / "experiment-config.json"
CONFIG = KitchenConfig.model_validate(json.loads(EXPERIMENT.read_text()))


def _robot_xy(*, bin_xy, standoff, azimuth):
    return np.array(bin_xy[:2]) - standoff * np.array((np.cos(azimuth), np.sin(azimuth)))


def test_the_drive_zone_keeps_the_footprint_and_a_tracking_margin_inside_the_kitchen():
    left, bottom, right, top = CONFIG.kitchen_bounds
    inset = CONFIG.robot_radius + KitchenDriveZone.TRACKING_MARGIN
    assert KitchenDriveZone.allows(config=CONFIG, position=(right - inset, 0.0))
    assert not KitchenDriveZone.allows(config=CONFIG, position=(right - inset + 0.01, 0.0))
    assert not KitchenDriveZone.allows(config=CONFIG, position=(0.0, top - inset + 0.01))


@pytest.mark.parametrize("context", ["far", "near"])
def test_every_sampled_toss_standoff_has_an_allowed_base_pose(*, context):
    env = KitchenRecyclingEnvironment(config=CONFIG)
    env.hard_reset()
    try:
        if context == "near":
            assert env.reset_movables(destination="relocate_bin_near")
        state = env.get_current_state()
        provider = KitchenRecyclingSkillProvider(env=env)
        toss = next(
            skill for skill in provider.skills() if skill.name == f"Toss_cube_control_{context}"
        )
        ground = GroundSkill(skill=toss, objects=(env.robot, env.items[0], env.bin))
        calibration = CONFIG.calibration(name="cube_control", context=context)
        bin_ = state.data[env.bin]
        yaw = Rotation.from_quat(bin_[3:7]).as_euler("xyz")[2]
        rng = np.random.default_rng(0)
        standoffs = [
            provider.sample_params_at_state(ground_skill=ground, rng=rng, state=state)[0]
            for _ in range(400)
        ]
        for standoff in standoffs:
            pose = _robot_xy(
                bin_xy=bin_[:2], standoff=standoff, azimuth=yaw + calibration.toss_azimuth
            )
            assert KitchenDriveZone.allows(config=CONFIG, position=pose), standoff
        # The range is clipped, not collapsed: it still spans most of the feasible band.
        assert np.ptp(standoffs) > 0.4
    finally:
        env.close()


def test_far_toss_range_starts_at_the_first_allowed_standoff():
    low, high = KitchenDriveZone.standoff_range(
        config=CONFIG,
        bin_xy=CONFIG.bin_position,
        azimuth=0.4,
        lower=1.0,
        upper=2.0,
    )
    assert 1.44 < low < 1.48
    assert high == pytest.approx(2.0)


def test_mat_side_cubes_beyond_pick_reach_are_not_reachable():
    # Pick drives its base to TARGET_DISTANCE from the cube; with the base held inside
    # the drive zone that ends near x = 1.26, matching the measured 0/117 beyond 1.35.
    assert KitchenDriveZone.pick_reachable(config=CONFIG, position=(1.20, 0.0))
    assert not KitchenDriveZone.pick_reachable(config=CONFIG, position=(1.40, 0.0))
    assert not KitchenDriveZone.pick_reachable(config=CONFIG, position=(1.55, -0.2))


def test_drive_zone_constants_match_the_pinned_controllers():
    from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController
    from kinder_models.dynamic3d.utils import WAYPOINT_TOLERANCE

    assert KitchenDriveZone.PICK_REACH == PickCubeController.TARGET_DISTANCE
    assert KitchenDriveZone.TRACKING_MARGIN == WAYPOINT_TOLERANCE


def test_live_reachable_flag_is_false_for_a_cube_past_pick_reach():
    env = KitchenRecyclingEnvironment(config=KitchenConfig(scatter_objects=False))
    env.hard_reset()
    try:
        # A missed toss leaves the cube here; no human placement may, so set it directly.
        simulator = env.simulator()
        raw = simulator.backend._require_state().copy()
        cube_body = raw.get_object_from_name(simulator.names["cube_control"])
        for feature, value in zip("xyz", (1.45, -0.25, 0.026), strict=True):
            raw.set(cube_body, feature, value)
        simulator.scene.set_state(raw)
        simulator.backend._state = raw
        simulator.idle(seconds=0.5)
        state = env.get_current_state()
        cube = next(item for item in env.items if item.name == "cube_control")
        assert state.get(obj=cube, feature_name="reachable") == 0
    finally:
        env.close()


def test_live_toss_from_a_standoff_past_the_mat_edge_is_refused_before_driving():
    env = KitchenRecyclingEnvironment(config=KitchenConfig(scatter_objects=False))
    env.hard_reset()
    try:
        simulator = env.simulator()
        picked = simulator.pick(name="cube_control")
        assert picked["success"], picked
        calibration = CONFIG.calibration(name="cube_control", context="far").model_copy(
            update={"toss_standoff": 1.40}
        )
        result = simulator.toss(name="cube_control", calibration=calibration)
        assert result["error"] is not None
        assert "PermissionError" not in result["error"]
        assert result["steps"] == 0
    finally:
        env.close()
