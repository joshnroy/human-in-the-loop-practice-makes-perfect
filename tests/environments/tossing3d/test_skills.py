import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import AT_THROW_POSE, HOLDING, REACHABLE
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills

from .conftest import build_state

_ENV = Tossing3DEnvironment
_PICK_OBJECTS = (_ENV.robot, _ENV.cube, _ENV.barrier, _ENV.bin_object)
_MOVE_OBJECTS = (_ENV.robot, _ENV.bin_object)
_TOSS_OBJECTS = (_ENV.robot, _ENV.cube, _ENV.bin_object, _ENV.goal_region, _ENV.barrier)


def _ground(*, skill, objects) -> GroundSkill:
    return GroundSkill(skill=skill, objects=objects)


def test_every_skill_grounds_with_the_objects_the_domain_exposes() -> None:
    """A type mismatch between a skill's parameters and the objects a caller binds is
    a GroundSkill validation error, so this is the whole binding contract in one test."""
    for skill, objects in (
        (Tossing3DSkills.PICK, _PICK_OBJECTS),
        (Tossing3DSkills.MOVE_TO_THROW_POSE, _MOVE_OBJECTS),
        (Tossing3DSkills.TOSS, _TOSS_OBJECTS),
    ):
        assert _ground(skill=skill, objects=objects).skill is skill


def test_pick_requires_the_cube_to_still_be_reachable() -> None:
    """Without this precondition the planner emits a cheap retrieve-and-retry plan the
    base can never execute, and prefers it over admitting the task is lost."""
    preconditions = {atom.predicate for atom in Tossing3DSkills.PICK.preconditions}
    assert REACHABLE in preconditions


def test_pick_deletes_at_throw_pose() -> None:
    """pick_shelf drives the base to the cube, so picking genuinely un-does having
    moved to the throw pose. Omitting this lets the planner emit Move, Pick, Toss."""
    deleted = {atom.predicate for atom in Tossing3DSkills.PICK.delete_effects}
    assert AT_THROW_POSE in deleted


def test_toss_deletes_reachable_as_well_as_holding() -> None:
    """A toss puts the cube past the barrier whether or not it lands in the region, so
    the model of a FAILED toss has to be as pessimistic as the dynamics are."""
    deleted = {atom.predicate for atom in Tossing3DSkills.TOSS.delete_effects}
    assert REACHABLE in deleted
    assert HOLDING in deleted


def test_the_three_skills_chain_into_a_solve() -> None:
    """Pick's add effects satisfy Toss's Holding precondition, and MoveToThrowPose's
    satisfy its AtThrowPose one -- i.e. Pick, Move, Toss is a real plan skeleton."""
    toss_preconditions = {atom.predicate for atom in Tossing3DSkills.TOSS.preconditions}
    added = {atom.predicate for atom in Tossing3DSkills.PICK.add_effects} | {
        atom.predicate for atom in Tossing3DSkills.MOVE_TO_THROW_POSE.add_effects
    }
    assert toss_preconditions <= added


def test_param_dims_match_what_compute_action_reads() -> None:
    assert Tossing3DSkills.PICK.param_dim == 2
    assert Tossing3DSkills.MOVE_TO_THROW_POSE.param_dim == 0
    assert Tossing3DSkills.TOSS.param_dim == 1


def test_sample_params_respects_the_configured_bounds() -> None:
    rng = np.random.default_rng(0)
    env = Tossing3DEnvironment(swing_low=0.3, swing_high=0.9)
    pick = _ground(skill=Tossing3DSkills.PICK, objects=_PICK_OBJECTS)
    toss = _ground(skill=Tossing3DSkills.TOSS, objects=_TOSS_OBJECTS)
    for _ in range(200):
        distance, rot = Tossing3DSkills.sample_params(ground_skill=pick, rng=rng, env=env)
        assert env.pick_distance_low <= distance < env.pick_distance_high
        assert env.pick_rot_low <= rot < env.pick_rot_high
        (swing,) = Tossing3DSkills.sample_params(ground_skill=toss, rng=rng, env=env)
        assert 0.3 <= swing < 0.9


def test_sample_params_returns_nothing_for_the_parameterless_skill() -> None:
    rng = np.random.default_rng(0)
    move = _ground(skill=Tossing3DSkills.MOVE_TO_THROW_POSE, objects=_MOVE_OBJECTS)
    params = Tossing3DSkills.sample_params(ground_skill=move, rng=rng, env=Tossing3DEnvironment())
    assert params.shape == (0,)


def test_compute_action_encodes_each_skill_id_and_its_parameters() -> None:
    state = build_state()
    pick = Tossing3DSkills.compute_action(
        ground_skill=_ground(skill=Tossing3DSkills.PICK, objects=_PICK_OBJECTS),
        params=np.array([0.57, -0.3]),
        state=state,
    )
    assert pick[0] == _ENV.SKILL_PICK
    assert pick[1] == pytest.approx(0.57)
    assert pick[2] == pytest.approx(-0.3)

    move = Tossing3DSkills.compute_action(
        ground_skill=_ground(skill=Tossing3DSkills.MOVE_TO_THROW_POSE, objects=_MOVE_OBJECTS),
        params=np.empty(0),
        state=state,
    )
    assert move[0] == _ENV.SKILL_MOVE_TO_THROW_POSE

    toss = Tossing3DSkills.compute_action(
        ground_skill=_ground(skill=Tossing3DSkills.TOSS, objects=_TOSS_OBJECTS),
        params=np.array([0.75]),
        state=state,
    )
    assert toss[0] == _ENV.SKILL_TOSS
    assert toss[1] == pytest.approx(0.75)


def test_compute_action_dispatches_on_skill_value_not_identity() -> None:
    """A Method that rebuilds an equal-content Skill (e.g. after serialization) must
    still work -- the same guarantee Tossing Room's compute_action makes."""
    rebuilt = Tossing3DSkills.TOSS.model_copy(deep=True)
    action = Tossing3DSkills.compute_action(
        ground_skill=_ground(skill=rebuilt, objects=_TOSS_OBJECTS),
        params=np.array([0.5]),
        state=build_state(),
    )
    assert action[0] == _ENV.SKILL_TOSS


def test_compute_action_rejects_an_unknown_skill() -> None:
    foreign = Tossing3DSkills.TOSS.model_copy(update={"name": "NotASkill"})
    with pytest.raises(ValueError, match="Unknown skill"):
        Tossing3DSkills.compute_action(
            ground_skill=_ground(skill=foreign, objects=_TOSS_OBJECTS),
            params=np.array([0.5]),
            state=build_state(),
        )
