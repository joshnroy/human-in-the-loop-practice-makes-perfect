"""The kitchen sim throws from one fixed standoff, as the real robot does."""

import argparse

import numpy as np

from hitl_pmp.environments.kitchen_recycling3d.cli import KitchenRecyclingCli
from hitl_pmp.environments.kitchen_recycling3d.environment import (
    KITCHEN_TOSS_PARAMETERS,
    KITCHEN_TOSS_STANDOFF_M,
    KitchenRecycling3DEnvironment,
)
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    SIM_TOSS_PARAMETERS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)


def test_the_kitchen_standoff_is_pinned_to_one_centimetre_around_1_45_m():
    assert KITCHEN_TOSS_STANDOFF_M == 1.45
    assert KITCHEN_TOSS_PARAMETERS.standoff_bounds == (1.445, 1.455)


def test_only_speed_and_release_keep_the_simulators_ranges():
    assert KITCHEN_TOSS_PARAMETERS.speed_bounds == SIM_TOSS_PARAMETERS.speed_bounds
    assert KITCHEN_TOSS_PARAMETERS.release_ms_bounds == SIM_TOSS_PARAMETERS.release_ms_bounds


def test_the_kitchen_environment_carries_the_fixed_standoff_set():
    env = KitchenRecycling3DEnvironment()
    assert Tossing3DToss.parameters_of(env=env) == KITCHEN_TOSS_PARAMETERS


def test_every_kitchen_draw_lies_in_the_fixed_band():
    rng = np.random.default_rng(0)
    for _ in range(200):
        standoff, _, _ = WideLongRangeTossProposal.sample(
            rng=rng, parameters=KITCHEN_TOSS_PARAMETERS
        )
        assert 1.445 <= standoff <= 1.455


def test_original_tossing3d_keeps_its_wide_standoff_band():
    assert SIM_TOSS_PARAMETERS.standoff_bounds == WIDE_TOSS_STANDOFF_BOUNDS
    assert WIDE_TOSS_STANDOFF_BOUNDS[1] - WIDE_TOSS_STANDOFF_BOUNDS[0] > 1.0


def test_the_kitchen_oracle_throws_from_the_fixed_standoff():
    parser = argparse.ArgumentParser()
    KitchenRecyclingCli.add_arguments(parser=parser)
    assert parser.parse_args([]).oracle_throw_standoff == KITCHEN_TOSS_STANDOFF_M


def test_the_kitchen_classifier_learns_speed_and_release_only():
    # The 1 cm band around 1.45 m is a placeholder interval; min/max normalisation
    # would stretch it to the span of speed and release (Josh, 2026-11-08).
    assert KITCHEN_TOSS_PARAMETERS.classifier_inputs == ("speed", "release_ms")


def test_a_kitchen_classifier_row_ignores_the_standoff():
    env = KitchenRecycling3DEnvironment()
    rows = [
        Tossing3DToss.feature_row(
            params=np.array([standoff, 150.0, 720.0]),
            parameters=Tossing3DToss.parameters_of(env=env),
        )
        for standoff in (1.445, 1.450, 1.455)
    ]
    assert rows == [[1.0, 150.0, 720.0]] * 3
