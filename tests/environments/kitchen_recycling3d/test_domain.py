"""Recycling semantics must not inherit cube-only scoring or free human recovery."""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.kitchen_recycling3d.domain import (
    Calibration,
    KitchenConfig,
    ObjectObservation,
    RecyclingEpisode,
    classify_zone,
    contained,
    grasp_yaws,
    world_extent,
)
from hitl_pmp.environments.kitchen_recycling3d.scene import scene_config


def observations(*, config, **updates):
    result = []
    for spec in config.objects:
        values = {"name": spec.name, "position": (0.5, 0, spec.dimensions[2] / 2)}
        values.update(updates.get(spec.name, {}))
        result.append(ObjectObservation(**values))
    return result


def test_config_is_new_variant_with_hardware_unknowns():
    config = KitchenConfig()
    assert config.variant != "tossing3d"
    assert config.required_objects == ("gatorade_20oz", "food_can")
    assert config.bin_outside == (0.381, 0.32385, 0.1651)
    assert config.hardware_blockers()
    assert config.measured_bin_mass is None
    assert config.mat_product.startswith("existing Sprout")
    assert not config.crossing_difficulty_demonstrated


def test_bin_within_direct_placement_reach_is_rejected():
    with pytest.raises(ValidationError, match="beyond direct placement"):
        KitchenConfig(bin_position=(1.3, 0, 0))


def test_sideways_bottle_has_object_pose_and_not_cube_yaw_assumption():
    config = KitchenConfig()
    observation = ObjectObservation(
        name="gatorade_20oz",
        position=(0.5, 0, 0.037),
        quaternion=tuple(Rotation.from_euler("y", np.pi / 2).as_quat()),
    )
    np.testing.assert_allclose(
        world_extent(spec=config.object(name=observation.name), observation=observation),
        (0.195, 0.073, 0.073),
    )
    assert len(grasp_yaws(spec=config.object(name=observation.name), observation=observation)) == 2
    assert classify_zone(config=config, observation=observation) == "kitchen"
    boundary = observation.model_copy(update={"position": (1.1, 0, 0.037)})
    assert classify_zone(config=config, observation=boundary) == "boundary_or_outside"


def test_drive_rule_checks_whole_robot_and_endpoints():
    episode = RecyclingEpisode(config=KitchenConfig())
    episode.check_drive(start=(0, 0), end=(0.7, 0))
    for endpoint in ((1, 0), (2, 0), (0, 2.4), (float("nan"), 0)):
        with pytest.raises(PermissionError):
            episode.check_drive(start=(0, 0), end=endpoint)


def test_full_object_containment_not_center_only():
    config = KitchenConfig()
    pose = (config.bin_position, (0, 0, 0, 1))
    sideways = tuple(Rotation.from_euler("y", np.pi / 2).as_quat())
    centered = ObjectObservation(
        name="gatorade_20oz", position=(2.05, 0, 0.043), quaternion=sideways
    )
    assert contained(config=config, observation=centered, bin_pose=pose)
    assert not contained(
        config=config,
        observation=centered.model_copy(update={"position": (2.18, 0, 0.043)}),
        bin_pose=pose,
    )
    assert not contained(
        config=config,
        observation=centered.model_copy(update={"position": (2.05, 0, 0.18)}),
        bin_pose=pose,
    )
    assert not contained(
        config=config, observation=centered, bin_pose=(config.bin_position, (1, 0, 0, 0))
    )


def test_tall_bottle_can_protrude_above_open_rim_but_cannot_balance_on_it():
    config = KitchenConfig()
    pose = (config.bin_position, (0, 0, 0, 1))
    bottle = ObjectObservation(name="gatorade_20oz", position=(2.05, 0, 0.1035))
    assert contained(config=config, observation=bottle, bin_pose=pose)
    rim_balanced = bottle.model_copy(update={"position": (2.05, 0, 0.2626)})
    assert not contained(config=config, observation=rim_balanced, bin_pose=pose)


def test_rotated_can_uses_cylinder_support_not_fictitious_box_corners():
    config = KitchenConfig()
    can = ObjectObservation(
        name="food_can",
        position=(2.05, 0, 0.041),
        quaternion=tuple(Rotation.from_euler("xyz", [0, np.pi / 2, 0.4]).as_quat()),
    )
    assert contained(config=config, observation=can, bin_pose=(config.bin_position, (0, 0, 0, 1)))


def test_settled_goal_requires_every_recycling_object_and_quiet_time():
    config = KitchenConfig()
    episode = RecyclingEpisode(config=config)
    pose = (config.bin_position, (0, 0, 0, 1))
    values = observations(
        config=config,
        gatorade_20oz={
            "position": (2.05, -0.06, 0.043),
            "quaternion": tuple(Rotation.from_euler("y", np.pi / 2).as_quat()),
        },
        food_can={"position": (2.05, 0.06, 0.056)},
    )
    episode.update(observations=values, bin_pose=pose, time=0)
    assert not episode.solved
    episode.update(observations=values, bin_pose=pose, time=0.7)
    assert episode.solved
    assert "cube_control" not in episode.settled
    values[-1] = values[-1].model_copy(update={"linear_speed": 0.1})
    episode.update(observations=values, bin_pose=pose, time=0.8)
    assert not episode.solved


def test_holding_requires_lift_bilateral_contacts_and_duration():
    config = KitchenConfig()
    episode = RecyclingEpisode(config=config)
    pose = episode.bin_pose
    episode.update(observations=observations(config=config), bin_pose=pose, time=0)
    lifted = observations(config=config, food_can={"position": (0.5, 0, 0.5), "finger_contacts": 0})
    episode.update(observations=lifted, bin_pose=pose, time=0.2)
    assert not episode.holding
    lifted[-1] = lifted[-1].model_copy(update={"finger_contacts": 2})
    episode.update(observations=lifted, bin_pose=pose, time=0.3)
    assert not episode.holding
    episode.update(observations=lifted, bin_pose=pose, time=0.5)
    assert episode.holding == {"food_can"}


@pytest.mark.parametrize("action", ["recover_object", "relocate_bin", "restore_bin", "dump_bin"])
def test_deployment_rejects_humans_without_changing_state_or_cost(*, action):
    episode = RecyclingEpisode(config=KitchenConfig(), phase="deployment")
    before = episode.model_dump()
    with pytest.raises(PermissionError):
        episode.request_human(action=action, object_name="food_can", destination=(0.5, 0, 0.05))
    assert episode.model_dump() == before


def test_training_interruption_relocation_and_restore_are_costed():
    episode = RecyclingEpisode(config=KitchenConfig())
    episode.request_human(
        action="recover_object", object_name="food_can", destination=(0.5, 0, 0.05)
    )
    episode.request_human(action="relocate_bin", destination=(0.7, 0.8, 0))
    episode.request_human(action="restore_bin")
    assert episode.human_cost == 36
    assert len(episode.events) == 3
    with pytest.raises(ValueError, match="kitchen"):
        episode.request_human(
            action="recover_object", object_name="food_can", destination=(2, 0, 0.05)
        )


def test_unrestored_bin_blocks_deployment():
    episode = RecyclingEpisode(config=KitchenConfig())
    episode.bin_pose = ((0.7, 0.8, 0), (0, 0, 0, 1))
    with pytest.raises(ValueError, match="restore"):
        episode.begin_deployment()
    assert episode.phase == "training"


def test_contexts_do_not_share_statistics_or_calibration():
    config = KitchenConfig(calibrations={"food_can/near": Calibration(release_ms=800)})
    episode = RecyclingEpisode(config=config)
    assert episode.context() == "far"
    episode.record_practice(name="food_can", skill="pick", success=True, context="near")
    episode.record_practice(name="food_can", skill="pick", success=False, context="far")
    assert episode.events[0]["statistics_key"] != episode.events[1]["statistics_key"]
    assert config.calibration(name="food_can", context="near").release_ms == 800
    assert config.calibration(name="food_can", context="far").release_ms == 720
    assert config.calibration(name="gatorade_20oz", context="near").release_ms == 720


def test_observation_set_must_be_complete_and_time_monotonic():
    episode = RecyclingEpisode(config=KitchenConfig())
    with pytest.raises(ValueError, match="complete"):
        episode.update(observations=[], bin_pose=episode.bin_pose, time=0)
    episode.update(
        observations=observations(config=episode.config), bin_pose=episode.bin_pose, time=1
    )
    with pytest.raises(ValueError, match="increasing"):
        episode.update(
            observations=observations(config=episode.config), bin_pose=episode.bin_pose, time=1
        )


def test_separate_scene_keeps_multiobject_goal_and_red_movable_bin():
    config = KitchenConfig()
    scene = scene_config(config=config)
    assert len(scene["goal_state"]) == 2
    assert len(scene["objects"]["kitchen_recycling_proxy"]) == 3
    assert not scene["objects"]["bin"]["bin_0"]["fixed"]
    assert scene["objects"]["bin"]["bin_0"]["rgba"][0] == 0.8
    assert "kitchenmatvisual" in scene["fixtures"]
