"""The SweepOffShelf3D practice domain: deployment value and belief routing."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.methods.belief_space.practice_domain import (
    SHELF_ROBOT_SKILLS,
    SWEEP_OFF_SHELF_DOMAIN,
    SweepOffShelfDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_observation_model import skill_belief_model
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource


def _value(*, horizon=4, **competence):
    competences = {name: np.array([competence.get(name, 0.0)]) for name in SHELF_ROBOT_SKILLS}
    return float(SweepOffShelfDeployment.values(competences=competences, horizon=horizon)[0])


def test_a_perfect_sweep_solves_in_two_skills():
    assert _value(horizon=2, PickWiper=1.0, SweepShelf=1.0) == 1.0
    assert _value(horizon=1, PickWiper=1.0, SweepShelf=1.0) == 0.0


def test_perfect_pick_and_place_needs_ten_skills():
    assert _value(horizon=10, PickCubeFromShelf=1.0, PlaceCubeInBox=1.0) == 1.0
    assert _value(horizon=9, PickCubeFromShelf=1.0, PlaceCubeInBox=1.0) == 0.0


def test_the_policy_takes_the_better_route():
    sweep_only = _value(horizon=12, PickWiper=0.9, SweepShelf=0.5)
    carry_only = _value(horizon=12, PickCubeFromShelf=0.95, PlaceCubeInBox=0.95)
    both = _value(
        horizon=12, PickWiper=0.9, SweepShelf=0.5, PickCubeFromShelf=0.95, PlaceCubeInBox=0.95
    )
    assert both >= max(sweep_only, carry_only) - 1e-12


def test_retries_raise_the_value_of_an_unreliable_sweep():
    assert _value(horizon=4, PickWiper=1.0, SweepShelf=0.5) == pytest.approx(1 - 0.5**3)


def test_every_shelf_skill_updates_its_own_belief():
    sweep = GroundSkill(skill=S.SWEEP_SHELF, objects=(O.robot, O.wiper, O.box, *O.cubes))
    model = skill_belief_model(ground_skill=sweep, domain=SWEEP_OFF_SHELF_DOMAIN)
    assert model.belief_name == "SweepShelf"
    assert model.example_source == PracticeExampleSource.SAMPLER
    reset = GroundSkill(skill=S.RESET_CUBES, objects=(O.robot, O.box, *O.cubes))
    model = skill_belief_model(ground_skill=reset, domain=SWEEP_OFF_SHELF_DOMAIN)
    assert model.belief_name == "reset_cubes"
    assert model.example_source == PracticeExampleSource.OUTCOME


def test_human_skills_are_inferred_not_assumed():
    """The robot is not told k: reset_cubes's competence is learned, never fixed at 1."""
    assert not SWEEP_OFF_SHELF_DOMAIN.deterministic_skills
    assert {"reset_cubes", "reset_wiper"} <= SWEEP_OFF_SHELF_DOMAIN.failure_effect_skills


def test_the_deployment_horizon_is_the_evaluation_episode_horizon():
    from hitl_pmp.environments.sweepoffshelf3d.problem import EPISODE_HORIZON

    assert SWEEP_OFF_SHELF_DOMAIN.deployment_horizon == EPISODE_HORIZON
