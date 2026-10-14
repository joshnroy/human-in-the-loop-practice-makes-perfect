"""`--env sweepoffshelf3d`'s flags: the human's true reset_cubes competence."""

import argparse

import pytest

from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli


def _parse(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    SweepOffShelf3DCli.add_arguments(parser=parser)
    return parser.parse_args(list(argv))


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_reset_cubes_success_probability_reaches_the_backend(*, p):
    args = _parse("--reset-cubes-success-probability", str(p))
    env = SweepOffShelf3DCli.build_problem(args=args, human_seed=0).env
    assert env.reset_cubes_success_probability == p
    assert env.backend().reset_cubes_success_probability == p


def test_the_human_succeeds_by_default():
    assert _parse().reset_cubes_success_probability == 1.0


@pytest.mark.parametrize("p", ["-0.1", "1.5"])
def test_a_probability_outside_the_unit_interval_is_rejected(*, p):
    with pytest.raises(SystemExit):
        _parse("--reset-cubes-success-probability", p)


def test_the_partial_left_behind_reset_is_gone():
    with pytest.raises(SystemExit):
        _parse("--left-behind", "2")
