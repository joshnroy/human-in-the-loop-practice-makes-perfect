"""With the jug's table beside the stove, a pour cycle needs no long base drive, and the
dumpling pick after the placement still plans from the shortened back-off pose."""

from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_layout_near_stove")

NEAR_STOVE = (
    Path(__file__).parents[3]
    / "src/hitl_pmp/environments/pouring3d/assets/kitchen-geometry-table-near-stove.json"
)
PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
POUR1 = np.array([1, 0, 1, 1.0, 0, 0], dtype=float)
PLACE = np.array([2, 0, 1, 0, 0, 0], dtype=float)
PICK_FROM_PAN = 3
# The default layout drives 1.9 m to pour and 2.3 m back to place (2026-11-05).
LONGEST_DRIVE_M = 1.0


def test_pick_pour_place_needs_no_long_base_drive():
    # Seed 0's poured dumpling lands where a pre-existing pick failure ("Motor tracking
    # failed in descend to shrimp body") also strikes in the default layout; seed 1's does not.
    env = KitchenEnvironment(
        canonical_seed=1,
        requested_count=1,
        geometry=KitchenGeometry.model_validate_json(NEAR_STOVE.read_text()),
    )
    c = env.controller()
    path = {"m": 0.0, "last": c.data.qpos[c.base_q][:2].copy()}

    def after() -> None:
        here = c.data.qpos[c.base_q][:2].copy()
        path["m"] += float(np.linalg.norm(here - path["last"]))
        path["last"] = here

    env.set_control_step_observers(before=None, after=after)
    # The dumpling pick after the placement plans from the pose the shortened
    # clear-workspace backoff leaves the base in.
    for action in (PICK_PITCHER, POUR1, PLACE, None):
        if action is None:
            index = c.scene.cube_locations().index("tray")
            action = np.array([PICK_FROM_PAN, index, 1, 0, 0, 0], dtype=float)
        path["m"] = 0.0
        env.take_action(action=action)
        skill = env.last_info["skill"]
        assert env.last_info["error"] is None, env.last_info
        assert path["m"] < LONGEST_DRIVE_M, (skill, path["m"])
        if skill == "PlacePitcherAtHome":
            assert c.vessel_home(name="pitcher")
            backoff = c.base_home[0] - c.data.qpos[c.base_q][0]
            assert backoff == pytest.approx(env.geometry.clear_workspace_backoff_m, abs=0.01)
    assert c.held == c.scene.cube_names[index]
