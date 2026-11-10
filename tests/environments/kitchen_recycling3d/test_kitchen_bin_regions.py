"""The kitchen's bin regions: randomized per side, and every corner probe-certified."""

import json
from pathlib import Path

import numpy as np

from hitl_pmp.environments.kitchen_recycling3d.bin_regions import (
    KITCHEN_BIN_REGION_BY_SIDE,
    PROBE_GRID_STEP_M,
    KitchenBinRegions,
)
from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig
from hitl_pmp.environments.kitchen_recycling3d.environment import KitchenRecycling3DEnvironment
from hitl_pmp.environments.kitchen_recycling3d.scene import initial_positions, scene_config
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide

REPO = Path(__file__).resolve().parents[3]
CONFIG = KitchenConfig.model_validate_json(
    (REPO / "scripts/kitchen_recycling/experiment-config.json").read_text()
)
PROBE_TABLE = REPO / "scripts/kitchen_recycling/bin_region_probe_results.jsonl"
FAR = KITCHEN_BIN_REGION_BY_SIDE[Tossing3DSide.OPPOSITE]
NEAR = KITCHEN_BIN_REGION_BY_SIDE[Tossing3DSide.ROBOT]


def rectangle(*, side: Tossing3DSide) -> tuple[float, float, float, float]:
    (extent,) = KITCHEN_BIN_REGION_BY_SIDE[side].ranges
    return extent


def inside(*, xy, side: Tossing3DSide) -> bool:
    x0, y0, x1, y1 = rectangle(side=side)
    return x0 - 1e-9 <= xy[0] <= x1 + 1e-9 and y0 - 1e-9 <= xy[1] <= y1 + 1e-9


def test_the_far_region_is_on_the_mats_and_the_near_region_on_the_kitchen_floor():
    half = np.array(CONFIG.bin_outside[:2]) / 2
    x0, y0, x1, y1 = rectangle(side=Tossing3DSide.OPPOSITE)
    assert x0 - half[0] >= CONFIG.mat_bounds[0]
    assert x1 + half[0] <= CONFIG.mat_bounds[2]
    x0, y0, x1, y1 = rectangle(side=Tossing3DSide.ROBOT)
    assert x1 + half[0] <= CONFIG.kitchen_bounds[2]
    assert y1 + half[1] <= CONFIG.kitchen_bounds[3]


def test_both_regions_span_at_least_0_3_m_each_way():
    for side in Tossing3DSide:
        x0, y0, x1, y1 = rectangle(side=side)
        assert x1 - x0 >= 0.3 and y1 - y0 >= 0.3, side


def test_the_scene_start_bin_is_drawn_from_the_far_region_by_scene_seed():
    starts = []
    for seed in range(30):
        (bin_range,) = scene_config(config=CONFIG, seed=seed)["regions"]["bin_start"]["ranges"]
        assert bin_range[:3] == bin_range[3:]
        assert inside(xy=bin_range[:2], side=Tossing3DSide.OPPOSITE)
        starts.append(tuple(bin_range[:2]))
    assert len(set(starts)) == 30
    again = scene_config(config=CONFIG, seed=7)["regions"]["bin_start"]["ranges"][0]
    assert tuple(again[:2]) == starts[7]


def test_the_bin_draw_leaves_the_cube_start_unchanged():
    for seed in (0, 3, 11):
        scene = scene_config(config=CONFIG, seed=seed)
        (cube_start,) = initial_positions(config=CONFIG, seed=seed)
        assert scene["regions"]["start_0"]["ranges"][0][:2] == list(cube_start)


def test_resets_name_their_side_and_carry_its_region():
    env = KitchenRecycling3DEnvironment(kitchen_config=CONFIG)
    for side in Tossing3DSide:
        region = env.bin_reset_region(destination=side.value)
        assert region is not None
        assert region["kitchen_side"] == side.value
        assert [tuple(r) for r in region["ranges"]] == list(KITCHEN_BIN_REGION_BY_SIDE[side].ranges)
    assert env.bin_reset_region(destination=None)["kitchen_side"] == "opposite_side"


def test_human_reset_destinations_lie_in_their_sides_region_and_vary():
    for side in Tossing3DSide:
        spots = [
            KitchenBinRegions.human_destination(
                config=CONFIG, side=side, robot=(0.0, 0.0), furniture=(), seed=5, index=index
            )
            for index in range(20)
        ]
        assert all(inside(xy=spot, side=side) for spot in spots)
        assert len({spot[:2] for spot in spots}) == 20
        assert spots[3] == KitchenBinRegions.human_destination(
            config=CONFIG, side=side, robot=(0.0, 0.0), furniture=(), seed=5, index=3
        )


def test_a_robot_side_destination_keeps_clear_of_the_robot():
    x0, y0, x1, y1 = rectangle(side=Tossing3DSide.ROBOT)
    robot = ((x0 + x1) / 2, (y0 + y1) / 2)
    half = np.array(CONFIG.bin_outside[:2]) / 2
    for index in range(20):
        spot = KitchenBinRegions.human_destination(
            config=CONFIG, side=Tossing3DSide.ROBOT, robot=robot, furniture=(), seed=1, index=index
        )
        outside = np.maximum(np.abs(np.array(robot) - np.array(spot[:2])) - half, 0.0)
        assert np.linalg.norm(outside) >= CONFIG.robot_radius + 0.05


def probe_cells(*, side: Tossing3DSide) -> dict[tuple[float, float], bool]:
    rows = [json.loads(line) for line in PROBE_TABLE.read_text().splitlines() if line]
    return {(row["x"], row["y"]): row["certified"] for row in rows if row["side"] == side.value}


def test_every_probed_cell_inside_a_region_and_every_corners_nearest_cell_is_certified():
    """The probe's rule (`bin_region_probe.py`): a region holds only certified cells,
    and each of its corners is within half a grid step of a certified cell."""
    for side in Tossing3DSide:
        cells = probe_cells(side=side)
        assert cells, side
        x0, y0, x1, y1 = rectangle(side=side)
        for (x, y), certified in cells.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                assert certified, (side, x, y)
        for corner in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            nearest = min(
                cells, key=lambda cell: np.hypot(cell[0] - corner[0], cell[1] - corner[1])
            )
            assert cells[nearest], (side, corner, nearest)
            assert np.hypot(nearest[0] - corner[0], nearest[1] - corner[1]) <= PROBE_GRID_STEP_M


def corners(*, side: Tossing3DSide) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = rectangle(side=side)
    return [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]


def test_every_far_region_bin_passes_the_configs_own_deployment_rules():
    """`KitchenConfig.valid_layout` refuses a bin within direct placement reach of the
    robot or off the mats; the region must not bypass it."""
    for x, y in corners(side=Tossing3DSide.OPPOSITE):
        KitchenConfig.model_validate({**CONFIG.model_dump(), "bin_position": (x, y, 0.0)})


def test_the_cube_spawn_keeps_clear_of_every_region_bin():
    from hitl_pmp.environments.kitchen_recycling3d.backend import KitchenKinderBackend
    from hitl_pmp.environments.kitchen_recycling3d.placement import KitchenHumanPlacement

    radius = float(np.linalg.norm(CONFIG.objects[0].dimensions)) / 2
    # The scene start's cube is on the kitchen floor, the far bin wholly on the mats.
    half = CONFIG.bin_outside[0] / 2
    for seed in range(20):
        (cube,) = initial_positions(config=CONFIG, seed=seed)
        bin_x = KitchenBinRegions.scene_start(seed=seed)[0]
        assert cube[0] + radius < CONFIG.mat_bounds[0] <= bin_x - half
    # A robot-side reset places the cube after the bin, clear of it.
    backend = KitchenKinderBackend(kitchen_config=CONFIG)
    for x, y in corners(side=Tossing3DSide.ROBOT):
        spots = backend._recovery_positions(robot_xy=np.array([0.0, 0.0]), bin_xyz=(x, y, 0.0))  # noqa: SLF001
        for _, spot in zip(range(10), spots, strict=False):
            assert KitchenHumanPlacement.recovery_clear_of_bin(
                config=CONFIG,
                position=spot,
                item_radius=radius,
                bin_pose=((x, y, 0.0), (0.0, 0.0, 0.0, 1.0)),
            )
