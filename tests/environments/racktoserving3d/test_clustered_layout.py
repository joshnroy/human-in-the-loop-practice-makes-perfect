"""The opt-in clustered start sampler: one random pile whose spread sigma is log-uniform.

Pile centre uniform over the rack inset, sigma log-uniform in [0.015, 0.10] m, each
dumpling's (x, y) the centre plus N(0, sigma) per axis clipped to the inset, yaw
uniform +-45 deg, redrawn until no two real footprints overlap.
"""

import importlib.util
import itertools
import math

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.layout import RackLayout
from hitl_pmp.environments.racktoserving3d.potsticker import RackPotsticker
from hitl_pmp.environments.racktoserving3d.semantics import SupportRegion

RACK = SupportRegion(x_min=1.3725, x_max=1.6275, y_min=-0.1875, y_max=0.1875, top_z=0.925)


def _footprint_points(*, x, y, yaw, n=400):
    """Dense sample of the real footprint, independent of the sampler's own check."""
    points = RackPotsticker.support_vertices()[:, :2]
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(len(points)), size=n)
    inside = np.vstack([points, weights @ points])
    c, s = math.cos(yaw), math.sin(yaw)
    return inside @ np.array([[c, s], [-s, c]]) + [x, y]


def _contains(*, x, y, yaw, points):
    """Points strictly inside the convex footprint placed at (x, y, yaw)."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(RackPotsticker.support_vertices()[:, :2])
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = (points - [x, y]) @ np.array([[c, s], [-s, c]])
    return np.all(local @ hull.equations[:, :2].T + hull.equations[:, 2] < -1e-4, axis=1)


@pytest.mark.parametrize("seed", range(40))
def test_clustered_layouts_keep_the_inset_yaw_sigma_and_real_footprints_apart(*, seed):
    layout, sigma = RackLayout.clustered(rng=np.random.default_rng(seed), rack=RACK, count=5)
    assert 0.015 <= sigma <= 0.10
    assert len(layout) == 5
    for x, y, yaw in layout:
        assert RACK.x_min + RackLayout.MARGIN <= x <= RACK.x_max - RackLayout.MARGIN
        assert RACK.y_min + RackLayout.MARGIN <= y <= RACK.y_max - RackLayout.MARGIN
        assert -math.pi / 4 <= yaw <= math.pi / 4
    for a, b in itertools.combinations(layout, 2):
        points = _footprint_points(x=a[0], y=a[1], yaw=a[2])
        assert not _contains(x=b[0], y=b[1], yaw=b[2], points=points).any()


def test_sigma_is_log_uniform_over_its_range():
    sigmas = np.array([
        RackLayout.clustered(rng=np.random.default_rng(seed), rack=RACK, count=5)[1]
        for seed in range(300)
    ])
    logs = (np.log(sigmas) - np.log(0.015)) / (np.log(0.10) - np.log(0.015))
    assert abs(np.mean(logs) - 0.5) < 0.06
    assert np.mean(logs < 1 / 3) == pytest.approx(1 / 3, abs=0.08)


def test_small_sigma_piles_are_tight_and_large_ones_spread():
    def spread(*, low, high):
        values = []
        for seed in range(400):
            layout, sigma = RackLayout.clustered(
                rng=np.random.default_rng(seed), rack=RACK, count=5
            )
            if low <= sigma < high:
                values.append(np.ptp([y for _, y, _ in layout]))
        return np.median(values)

    assert spread(low=0.015, high=0.025) < 0.08 < spread(low=0.06, high=0.10)


def test_footprints_that_touch_are_rejected_but_neighbours_are_not():
    assert RackLayout.footprints_overlap(a=(0.0, 0.0, 0.0), b=(0.0, 0.05, 0.0))
    assert not RackLayout.footprints_overlap(a=(0.0, 0.0, 0.0), b=(0.0, 0.057, 0.0))
    # Side by side along x, 22 mm deep each: 25 mm apart is clear, under the 61 mm rule.
    assert not RackLayout.footprints_overlap(a=(0.0, 0.0, 0.0), b=(0.025, 0.0, 0.0))
    assert RackLayout.footprints_overlap(a=(0.0, 0.0, 0.0), b=(0.02, 0.0, 0.0))


def test_clustered_is_opt_in():
    assert RackServingConfig().start_layout == "scattered"
    assert RackServingConfig(start_layout="clustered").start_layout == "clustered"


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")
def test_a_clustered_start_settles_five_pieces_on_the_rack_and_reports_sigma():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

    env = RackServingEnvironment(config=RackServingConfig(start_layout="clustered"))
    try:
        env.reset_to_seed(seed=2)
        backend = env.backend()
        assert 0.015 <= backend.start_sigma() <= 0.10
        atoms = env.get_current_state().abstract_atoms
        for name in backend.cube_names:
            assert ("OnRack", (name, "rack_0")) in atoms
    finally:
        env.close()
