"""Simulator-backed checks of the reset's primitives and the stock-skill planning fixes."""

import importlib.util

import numpy as np
import pytest

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)


@needs_kinder
def test_the_robot_pulls_the_drawer_past_the_countertop_and_closes_it_again() -> None:
    """The stock OpenDrawer stops at ~0.11 m, under the countertop's edge. Gripping the
    handle and driving the base back opens it to 0.25 m -- no task change -- and the
    same grip pushes it shut."""
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession

    session = SweepDrawerSession(seed=1)
    try:
        prims = SweepDrawerSelfReset(session=session).primitives
        prims.move_drawer(target=0.25)
        assert session.drawer_pos() == pytest.approx(0.25, abs=0.03)
        prims.move_drawer(target=0.0)
        assert session.drawer_pos() < 0.01
        # the cubes on the counter were not disturbed by any of it
        assert all(session.in_pile(cube=c) for c in session.locations())
    finally:
        session.close()


def test_stock_planning_functions_are_unmodified_upstream():
    from kinder_models.dynamic3d import utils
    from kinder_models.dynamic3d.sweep3D import parameterized_skills as sweep

    assert sweep.PyBulletSim is utils.PyBulletSim
    assert sweep.run_base_motion_planning is utils.run_base_motion_planning


def test_jitter_does_not_make_a_refused_pick_worth_planning_again() -> None:
    """Seed 22: the planner refused two cubes, then searched 30 s for each again, because
    a position rounded to the millimetre had crossed a rounding boundary."""
    from hitl_pmp.environments.sweep_drawer3d.self_reset import Unmoved

    before = np.array([0.88549, 0.04975, 0.45389, 0.0003])
    now = np.array([0.88551, 0.04975, 0.45389, 0.0])
    assert round(float(before[0]), 3) != round(float(now[0]), 3)
    assert Unmoved.since(before=before, now=now)


def test_a_cube_that_was_shifted_is_worth_planning_again() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import Unmoved

    before = np.array([0.8855, 0.0498, 0.4539, 0.0])
    assert not Unmoved.since(before=before, now=before + np.array([0.08, 0.0, 0.0, 0.08]))


def test_a_neighbour_taken_away_makes_a_pick_worth_planning_again() -> None:
    """Seed 21: a floor cube had no pick until the cube beside it was put back."""
    from hitl_pmp.environments.sweep_drawer3d.self_reset import Unmoved

    alone = np.array([0.95, 0.15, 0.01, 0.0])
    beside = np.array([0.95, 0.15, 0.01, 0.96, 0.13, 0.01, 0.0])
    assert not Unmoved.since(before=beside, now=alone)


def test_a_pick_never_refused_is_not_stuck() -> None:
    from hitl_pmp.environments.sweep_drawer3d.self_reset import Unmoved

    assert not Unmoved.since(before=None, now=np.zeros(4))
