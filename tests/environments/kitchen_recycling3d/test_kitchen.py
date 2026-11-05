"""The kitchen geometry Tossing3D's skills run in: scene, drive zone, human placements."""

import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS
from hitl_pmp.environments.kitchen_recycling3d.backend import KitchenKinderBackend
from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.placement import (
    NEAR_BIN_POSITION,
    KitchenHumanPlacement,
)
from hitl_pmp.environments.kitchen_recycling3d.scene import scene_config
from hitl_pmp.environments.kitchen_recycling3d.step_protocol import KitchenRecyclingStepProtocol

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = REPO / "scripts/kitchen_recycling/experiment-config.json"
COSTS = REPO / "scripts/kitchen_recycling/costs.json"


def config() -> KitchenConfig:
    return KitchenConfig.model_validate_json(EXPERIMENT_CONFIG.read_text())


def test_the_goal_region_carries_the_name_kinders_in_bin_classifier_reads():
    scene = scene_config(config=config(), seed=0)
    assert "blocks_goal_region" in scene["regions"]
    assert scene["regions"]["blocks_goal_region"]["target"] == "bin_0"
    assert scene["goal_state"] == [["on", "cube_0", "blocks_goal_region"]]


def test_the_mats_are_a_visual_not_a_scene_object():
    scene = scene_config(config=config(), seed=0)
    assert "sprout_mats" not in json.dumps(scene["fixtures"])
    assert not any("sprout_mats" in atom for atom in scene["initial_state"])
    assert scene["kitchen_mats"] == list(config().mat_bounds)


def test_the_barrier_is_the_mat_edge():
    scene = scene_config(config=config(), seed=0)
    barrier = scene["regions"]["boundary_start"]["ranges"][0]
    assert barrier[0] == config().mat_bounds[0] == 1.15


def test_multi_object_configs_are_refused():
    with pytest.raises(ValueError, match="single-object"):
        KitchenRecycling3DEnvironment(kitchen_config=KitchenConfig()).build_backend()


def test_resets_map_to_the_two_kitchen_placements():
    env = KitchenRecycling3DEnvironment(kitchen_config=config())
    assert env.bin_reset_region(destination="robot_side") == {"kitchen_side": "robot_side"}
    assert env.bin_reset_region(destination="opposite_side") == {"kitchen_side": "opposite_side"}


def test_pick_reach_is_the_drive_zones_not_the_barrier_plane():
    backend = KitchenKinderBackend(kitchen_config=config())
    at = {"robot": {"pos_base_x": 0.0}, "cuboid_barrier": {"x": 1.15}}
    # Just onto the mats: across the barrier plane, still inside the pick's reach.
    assert not backend.pick_known_unplannable(features={**at, "cube_0": {"x": 1.2, "y": 0.0}})
    assert backend.pick_known_unplannable(features={**at, "cube_0": {"x": 1.4, "y": 0.0}})


def test_base_stands_are_limited_to_the_drive_zone():
    backend = KitchenKinderBackend(kitchen_config=config())
    assert backend.base_stand_allowed(xy=(0.7, 0.0))
    assert not backend.base_stand_allowed(xy=(0.8, 0.0))


def test_near_bin_spot_is_unchanged_when_the_robot_is_clear():
    spot = KitchenHumanPlacement.near_bin_destination(
        config=config(), robot=(0.0, 0.0), items=[], furniture=[]
    )
    assert spot == NEAR_BIN_POSITION


def test_near_bin_moves_off_a_robot_standing_on_its_spot():
    spot = KitchenHumanPlacement.near_bin_destination(
        config=config(), robot=NEAR_BIN_POSITION[:2], items=[], furniture=[]
    )
    assert np.linalg.norm(np.array(spot[:2]) - NEAR_BIN_POSITION[:2]) > 0.3


def test_registered_for_both_runners():
    assert ENVIRONMENTS["kitchen_recycling3d"] is KitchenRecyclingCli
    assert STEP_PROTOCOL_ENVIRONMENTS["kitchen_recycling3d"] is KitchenRecyclingStepProtocol


def test_cost_file_prices_both_resets_at_five():
    human = json.loads(COSTS.read_text())["human_skills"]
    names = set(KitchenRecyclingStepProtocol.human_skill_names().values())
    assert names == set(human)
    assert {human[name]["cost"]["value"] for name in names} == {5}


def test_practice_charges_resets_under_the_cost_file_names():
    import argparse

    from hitl_pmp.core.problem.problem import Problem

    args = argparse.Namespace(
        kitchen_config=EXPERIMENT_CONFIG,
        scene_bg=False,
        seed=0,
        test_env_seed_offset=10000,
        human_reset=True,
        oracle_throw_standoff=1.55,
    )
    problem = KitchenRecyclingCli.build_problem(args=args)
    assert isinstance(problem, Problem)
    provider = KitchenRecyclingCli.practice_context(args=args, problem=problem).skill_provider
    charged = {provider.human_skill_name(ground_skill=g) for g in provider.movables_reset_skills()}
    human = json.loads(COSTS.read_text())["human_skills"]
    assert charged == set(KitchenRecyclingStepProtocol.human_skill_names().values()) == set(human)
