"""Gather succeeds only when it leaves the food clustered tightly enough for sweep5.

The blade-band Gathered classifier passed almost every gather at delta <= 0.08, while
sweep5 after it succeeded 9/10 down to 6/17 as delta grew (2026-11-23), so gather's own
label said nothing about what sweep5 needs. The opt-in test is on the measured state
at the end of gather: every dumpling not yet on the plate lies within a band about the
plate's centre line and, optionally, within a depth behind the rack's front edge.
"""

import argparse
import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.predicates import RackServing3DAtoms


def _box(*, x: float, y: float) -> np.ndarray:
    return np.array([[x - 0.015, y - 0.03, 0.93], [x + 0.015, y + 0.03, 0.93]])


@pytest.mark.parametrize(
    ("ys", "half_width", "clustered"),
    [
        ([0.0, 0.02, -0.02], 0.06, True),
        ([0.0, 0.04], 0.06, False),
        ([0.0, 0.04], 0.08, True),
        ([], 0.06, True),
    ],
)
def test_clustered_bounds_every_unserved_footprint_about_the_centre_line(
    *, ys, half_width, clustered
):
    corners = [_box(x=1.45, y=y) for y in ys]
    assert (
        RackServing3DAtoms.clustered_for_sweep(
            unserved_corners=corners,
            center_y=0.0,
            rack_front=1.3725,
            half_width=half_width,
            depth=None,
        )
        is clustered
    )


@pytest.mark.parametrize(("x", "clustered"), [(1.44, True), (1.47, False)])
def test_clustered_bounds_how_far_behind_the_rack_front_the_food_lies(*, x, clustered):
    assert (
        RackServing3DAtoms.clustered_for_sweep(
            unserved_corners=[_box(x=x, y=0.0)],
            center_y=0.0,
            rack_front=1.3725,
            half_width=0.1,
            depth=0.09,
        )
        is clustered
    )


def test_the_cli_carries_the_gather_success_thresholds_to_the_toy_environment():
    from hitl_pmp.environments.racktoserving3d.cli import RackServingCli

    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    env = RackServingCli.build_problem(args=args, mode="training").env
    assert env.gather_success_half_width is None
    assert env.gather_success_depth is None
    args = parser.parse_args(
        ["--gather-success-half-width", "0.09", "--gather-success-depth", "0.12"]
    )
    args.seed = 0
    env = RackServingCli.build_problem(args=args, mode="training").env
    assert env.gather_success_half_width == pytest.approx(0.09)
    assert env.gather_success_depth == pytest.approx(0.12)


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="KINDER simulator")
def test_a_loose_gather_fails_only_under_the_tight_test():
    from hitl_pmp.environments.racktoserving3d.cli import RackServingCli

    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    problem = RackServingCli.build_problem(args=args, mode="training")
    problem.hard_reset()
    env = problem.env
    try:
        backend = env.backend()
        height = backend.support(name="rack_0").top_z + backend.config.dumpling_height / 2
        x = backend.support(name="rack_0").x_min + 0.06
        placements = {
            name: (np.array([x, y, height + 0.002]), np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]))
            for name, y in zip(backend.cube_names, (-0.07, -0.035, 0.0, 0.035, 0.07), strict=True)
        }
        backend._commit(placements=placements)
        env.rack_controllers().shelf.hold(count=40, label="food_in_a_loose_column")
        assert env.measure().gathered
        env.gather_success_half_width = 0.06
        assert not env.measure().gathered
        env.gather_success_half_width = 0.11
        assert env.measure().gathered
    finally:
        env.close()
