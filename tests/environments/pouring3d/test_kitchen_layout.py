"""The kitchen's station layout is a `KitchenGeometry` choice: the default reproduces the
stations asset, and the near-stove prototype puts the jug's table beside the stove."""

import json
from pathlib import Path

import pytest

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry

ASSETS = Path(__file__).parents[3] / "src/hitl_pmp/environments/pouring3d/assets"
NEAR_STOVE = ASSETS / "kitchen-geometry-table-near-stove.json"


def footprint(*, scene: dict, fixture: str, region: str) -> tuple[float, float, float, float]:
    kind = "table" if fixture == "prep_table" else "pourstovetop"
    table = scene["fixtures"][kind][fixture]
    x, y = scene["regions"][region]["ranges"][0][:2]
    half_x, half_y = table["length"] / 2, table["width"] / 2
    return x - half_x, x + half_x, y - half_y, y + half_y


def test_the_default_layout_reproduces_the_stations_asset():
    source = json.loads((ASSETS / "potsticker-pour3-stations.json").read_text())
    scene = KitchenGeometry().scene_config()
    for region in ("robot_home", "pitcher_home", "prep_table_home", "stovetop_home", "tray_home"):
        assert (
            scene["regions"][region]["ranges"][0][:2] == source["regions"][region]["ranges"][0][:2]
        )
    assert scene["cameras"] == source["cameras"]


def test_the_near_stove_layout_sets_the_jug_table_beside_the_stove():
    geometry = KitchenGeometry.model_validate_json(NEAR_STOVE.read_text())
    scene = geometry.scene_config()
    prep = footprint(scene=scene, fixture="prep_table", region="prep_table_home")
    stove = footprint(scene=scene, fixture="stovetop", region="stovetop_home")
    # Side by side along y, not overlapping, with no aisle-sized gap between them.
    assert prep[3] <= stove[2] + 1e-9
    assert stove[2] - prep[3] < 0.05
    # The jug stands on its table and the robot starts outside both tables.
    jug = scene["regions"]["pitcher_home"]["ranges"][0]
    assert prep[0] < jug[0] < prep[1] and prep[2] < jug[1] < prep[3]
    robot = scene["regions"]["robot_home"]["ranges"][0]
    assert robot[0] < min(prep[0], stove[0]) - 0.3
    # The jug's home is within half a metre of the pan's.
    pan = scene["regions"]["tray_home"]["ranges"][0]
    assert ((jug[0] - pan[0]) ** 2 + (jug[1] - pan[1]) ** 2) ** 0.5 < 0.5
    # Same height class: both worktops at the shared desk height.
    assert geometry.counter_height_m == geometry.hob_height_m


def test_a_jug_home_off_its_table_is_refused():
    with pytest.raises(ValueError, match="jug's home"):
        KitchenGeometry(pitcher_home_xy=(0.7, 0.9))


def test_the_clear_workspace_backoff_defaults_to_the_old_35_cm_and_shrinks_near_the_stove():
    assert KitchenGeometry().clear_workspace_backoff_m == 0.35
    near = KitchenGeometry.model_validate_json(NEAR_STOVE.read_text())
    assert near.clear_workspace_backoff_m < 0.35
    with pytest.raises(ValueError):
        KitchenGeometry(clear_workspace_backoff_m=-0.1)
