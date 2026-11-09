"""`--human-reset-success-probability p`: the human pour-back's true competence.

Mirrors SweepOffShelf3D's `--reset-cubes-success-probability` (#433). Each attempt fully
succeeds with probability p and otherwise changes nothing; it is charged either way.
"""

import argparse
import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import RESET_NAME
from tests.practice_cost_fixtures import CostFixtures

needs_kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER")


def _spilled(**fields: float) -> KitchenEnvironment:
    import mujoco

    env = KitchenEnvironment(requested_count=3, **fields)
    c = env.controller()
    c.scene.get_object("potsticker_0").set_pose([0.35, 0.2, 0.002], [1, 0, 0, 0])
    mujoco.mj_forward(c.model, c.data)
    assert c.scene.cube_locations()[0] == "floor"
    return env


def test_the_default_is_a_human_who_always_succeeds():
    env = KitchenEnvironment()
    assert env.human_reset_success_probability == 1.0
    assert all(env.human_reset_succeeds() for _ in range(200))


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_a_probability_outside_the_unit_interval_is_refused(*, p):
    with pytest.raises(ValueError):
        KitchenEnvironment(human_reset_success_probability=p)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.8])
def test_the_empirical_success_rate_over_seeded_attempts_is_close_to_p(*, p):
    attempts = 4000
    env = KitchenEnvironment(human_reset_success_probability=p, human_seed=3)
    successes = sum(env.human_reset_succeeds() for _ in range(attempts))
    # Five standard errors: a seeded, deterministic check that cannot flake.
    assert abs(successes / attempts - p) < 5 * np.sqrt(p * (1 - p) / attempts)


def test_the_draws_are_determined_by_the_seed_and_independent_of_a_methods_rng():
    def draws(*, seed: int) -> list[bool]:
        env = KitchenEnvironment(human_reset_success_probability=0.5, human_seed=seed)
        return [env.human_reset_succeeds() for _ in range(64)]

    assert draws(seed=4) == draws(seed=4)
    assert draws(seed=4) != draws(seed=5)
    # Not the stream a method seeded with the same run seed would draw from.
    method = np.random.default_rng(4)
    assert draws(seed=4) != [bool(method.random() < 0.5) for _ in range(64)]


def test_the_cli_flag_reaches_the_practice_environment_seeded_by_the_run_seed():
    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    assert parser.parse_args([]).human_reset_success_probability == 1.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--human-reset-success-probability", "1.5"])
    args = parser.parse_args(["--human-reset-success-probability", "0.25"])
    args.seed = 7
    args.practice_cost_config = CostFixtures.BY_ENV["pouring3d_kitchen"]
    env = KitchenCli.build_problem(args=args).env
    assert env.human_reset_success_probability == 0.25
    assert env.human_seed == 7


@needs_kinder
@pytest.mark.xdist_group("kitchen_human_reset_success_probability")
def test_p_one_is_exactly_todays_reset():
    new = _spilled(human_reset_success_probability=1.0, human_seed=11)
    old = _spilled()
    try:
        assert new.reset_movables(destination=HUMAN_RESET)
        assert new.last_info["success"]
        assert FrozenPourEnvironment.reset_movables(old, destination=RESET_NAME)
        np.testing.assert_array_equal(new.controller().data.qpos, old.controller().data.qpos)
        np.testing.assert_array_equal(new.controller().data.qvel, old.controller().data.qvel)
    finally:
        new.close()
        old.close()


@needs_kinder
@pytest.mark.xdist_group("kitchen_human_reset_success_probability")
def test_p_zero_never_changes_the_state_but_is_charged_and_scored_a_failure():
    env = _spilled(human_reset_success_probability=0.0)
    try:
        c = env.controller()
        human = KitchenSkillProvider(env=env).human_cube_bin_reset_skill()
        assert human is not None
        before = env.observe().abstract_atoms
        for attempt in range(1, 4):
            start = float(c.data.time)
            # Attempted, so the human oracle does not raise: the outcome is the failure.
            assert env.reset_movables(destination=HUMAN_RESET)
            assert not env.last_info["success"]
            assert env.human_interventions == attempt
            assert env.interruption_cost_total == 5 * attempt
            # The attempt takes the human its duration; the world does not pause for it.
            assert float(c.data.time) == pytest.approx(start + 1 / c.frequency)
            assert c.scene.cube_locations()[0] == "floor"
            after = env.observe().abstract_atoms

            # The human moved nothing; only whether a piece has come to rest may change.
            def placed(*, atoms):
                return {a for a in atoms if a[0] != "Settled"}

            assert placed(atoms=after) == placed(atoms=before)
            # How the methods score a human skill (`observe_help_granted`): a success
            # exactly when its promised effects hold afterwards.
            promised = {
                (a.predicate.name, tuple(o.name for o in a.objects)) for a in human.add_effects
            }
            assert not promised <= after
    finally:
        env.close()


def test_the_cli_reads_the_human_attempts_duration_from_the_costs_file():
    parser = argparse.ArgumentParser()
    KitchenCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    args.practice_cost_config = CostFixtures.BY_ENV["pouring3d_kitchen"]
    env = KitchenCli.build_problem(args=args).env
    assert env.human_reset_ticks == 1


@needs_kinder
@pytest.mark.xdist_group("kitchen_human_reset_success_probability")
def test_a_failed_attempt_lets_the_scene_run_for_the_attempts_duration():
    """Curtis seed 1, p = 0.1 (pouring-hrsweep-evalcost-20261108): 21 failed attempts in
    a row left the sim clock at one instant, so a dumpling caught mid-rock stayed
    caught. A failed attempt now steps the scene for the human's duration."""
    env = _spilled(human_reset_success_probability=0.0, human_reset_ticks=5)
    try:
        c = env.controller()
        start = float(c.data.time)
        assert env.reset_movables(destination=HUMAN_RESET)
        assert float(c.data.time) == pytest.approx(start + 5 / c.frequency)
    finally:
        env.close()
