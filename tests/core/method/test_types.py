import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    Rollout,
    SetupCommand,
    SetupCommandTarget,
    Skill,
    SkillPracticeTally,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)
_OBJ2 = Object(name="block2", type=_BLOCK)
_OBJ3 = Object(name="block3", type=_BLOCK)

_AT = Predicate(name="At", types=(_BLOCK, _BLOCK), holds=lambda state, objects: True)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


def _move_skill() -> tuple[Skill, Variable, Variable, Variable]:
    """A generic "Move" skill: At(robot, current) -> At(robot, target), matching
    the shape LightSwitchSkills.MOVE_ROBOT will use, but domain-agnostic here."""
    robot = Variable(name="robot", type=_BLOCK)
    current = Variable(name="current", type=_BLOCK)
    target = Variable(name="target", type=_BLOCK)
    skill = Skill(
        name="Move",
        parameters=(robot, current, target),
        preconditions=frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))}),
        add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, target))}),
        delete_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))}),
        param_dim=0,
    )
    return skill, robot, current, target


def test_rollout_accepts_one_fewer_action_than_states() -> None:
    rollout = Rollout(states=[_state(x=0.0), _state(x=1.0)], actions=[np.array([1.0])])
    assert len(rollout.actions) == len(rollout.states) - 1


def test_rollout_accepts_single_state_and_no_actions() -> None:
    rollout = Rollout(states=[_state(x=0.0)], actions=[])
    assert rollout.actions == []


def test_rollout_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        Rollout(states=[_state(x=0.0)], actions=[np.array([1.0])])


def test_setup_command_for_robot_target() -> None:
    command = SetupCommand(target=SetupCommandTarget.ROBOT, goal=Goal(atoms=frozenset()))
    assert command.target is SetupCommandTarget.ROBOT


def test_setup_command_for_human_target() -> None:
    command = SetupCommand(target=SetupCommandTarget.HUMAN, goal=Goal(atoms=frozenset()))
    assert command.target is SetupCommandTarget.HUMAN


def test_variable_declares_a_name_and_type() -> None:
    var = Variable(name="robot", type=_BLOCK)
    assert var.name == "robot"
    assert var.type == _BLOCK


def test_variables_with_equal_content_are_equal_and_hashable() -> None:
    a = Variable(name="robot", type=_BLOCK)
    b = Variable(name="robot", type=_BLOCK)
    assert a == b
    assert hash(a) == hash(b)


def test_lifted_atom_grounds_into_a_ground_atom_via_substitution() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    lifted = LiftedAtom(predicate=_AT, variables=(robot, robot))
    ground = lifted.ground(substitution={robot: _OBJ})
    assert ground == GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ))


def test_lifted_atoms_with_equal_content_are_equal_and_hashable() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    a = LiftedAtom(predicate=_AT, variables=(robot,))
    b = LiftedAtom(predicate=_AT, variables=(robot,))
    assert a == b
    assert hash(a) == hash(b)


def test_skill_declares_parameters_preconditions_effects_and_param_dim() -> None:
    skill, robot, current, target = _move_skill()
    assert skill.name == "Move"
    assert skill.parameters == (robot, current, target)
    assert skill.preconditions == frozenset({LiftedAtom(predicate=_AT, variables=(robot, current))})
    assert skill.add_effects == frozenset({LiftedAtom(predicate=_AT, variables=(robot, target))})
    assert skill.delete_effects == frozenset({
        LiftedAtom(predicate=_AT, variables=(robot, current))
    })
    assert skill.param_dim == 0


def test_skill_ignore_effects_default_to_empty() -> None:
    """Most operators are monotone, so `ignore_effects` is opt-in -- every existing
    Skill construction keeps working unchanged."""
    skill, *_ = _move_skill()
    assert skill.ignore_effects == frozenset()


def test_skill_accepts_ignore_effects_as_bare_predicates() -> None:
    """Unlike the other three effect fields, ignore effects are Predicates, not
    LiftedAtoms: they name a whole predicate, with no variables to bind."""
    robot = Variable(name="robot", type=_BLOCK)
    skill = Skill(
        name="Teleport",
        parameters=(robot,),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        ignore_effects=frozenset({_AT}),
        param_dim=0,
    )
    assert skill.ignore_effects == frozenset({_AT})


def test_skill_rejects_a_precondition_variable_not_in_parameters() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    stray = Variable(name="stray", type=_BLOCK)
    with pytest.raises(ValidationError):
        Skill(
            name="Move",
            parameters=(robot,),
            preconditions=frozenset({LiftedAtom(predicate=_AT, variables=(robot, stray))}),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            param_dim=0,
        )


def test_skill_rejects_an_effect_variable_not_in_parameters() -> None:
    robot = Variable(name="robot", type=_BLOCK)
    stray = Variable(name="stray", type=_BLOCK)
    with pytest.raises(ValidationError):
        Skill(
            name="Move",
            parameters=(robot,),
            preconditions=frozenset(),
            add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(robot, stray))}),
            delete_effects=frozenset(),
            param_dim=0,
        )


def test_skills_with_equal_content_are_equal_and_hashable() -> None:
    a, *_ = _move_skill()
    b, *_ = _move_skill()
    assert a == b
    assert hash(a) == hash(b)


def test_ground_skill_binds_a_skill_to_concrete_objects() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.skill == skill
    assert ground_skill.objects == (_OBJ, _OBJ2, _OBJ3)


def test_ground_skill_rejects_wrong_number_of_objects() -> None:
    skill, *_ = _move_skill()
    with pytest.raises(ValidationError):
        GroundSkill(skill=skill, objects=(_OBJ, _OBJ2))


def test_ground_skill_rejects_an_object_whose_type_does_not_match_its_parameter() -> None:
    skill, *_ = _move_skill()
    wrong_type = Type(name="not_a_block", feature_names=())
    wrong_obj = Object(name="thing", type=wrong_type)
    with pytest.raises(ValidationError):
        GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, wrong_obj))


def test_ground_skill_grounds_preconditions_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.preconditions == frozenset({
        GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ2))
    })


def test_ground_skill_grounds_add_effects_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.add_effects == frozenset({GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ3))})


def test_ground_skill_grounds_delete_effects_by_substituting_objects_for_parameters() -> None:
    skill, *_ = _move_skill()
    ground_skill = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert ground_skill.delete_effects == frozenset({
        GroundAtom(predicate=_AT, objects=(_OBJ, _OBJ2))
    })


def test_ground_skill_forwards_ignore_effects_ungrounded() -> None:
    """There is nothing to substitute into an ignore effect -- it names a Predicate,
    not an atom -- so grounding passes it straight through, like predicators'
    `_GroundNSRT.ignore_effects`."""
    robot = Variable(name="robot", type=_BLOCK)
    skill = Skill(
        name="Teleport",
        parameters=(robot,),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        ignore_effects=frozenset({_AT}),
        param_dim=0,
    )
    assert GroundSkill(skill=skill, objects=(_OBJ,)).ignore_effects == frozenset({_AT})


def test_ground_skills_with_equal_content_are_equal_and_hashable() -> None:
    skill, *_ = _move_skill()
    a = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    b = GroundSkill(skill=skill, objects=(_OBJ, _OBJ2, _OBJ3))
    assert a == b
    assert hash(a) == hash(b)


def test_a_fresh_practice_tally_is_all_zeros() -> None:
    tally = SkillPracticeTally()
    assert tally.num_attempts == 0
    assert tally.num_successes == 0
    assert tally.num_random_attempts == 0
    assert tally.num_random_successes == 0
    assert tally.num_informed_attempts == 0
    assert tally.num_informed_successes == 0


def test_with_attempt_files_one_execution_into_exactly_one_pool() -> None:
    """Every attempt lands in exactly one of random / informed / fallback, which is
    what makes the three recoverable from six numbers: fallback is the remainder."""
    tally = SkillPracticeTally()
    tally = tally.with_attempt(success=True, was_random=True, was_informed=False)
    tally = tally.with_attempt(success=False, was_random=False, was_informed=True)
    tally = tally.with_attempt(success=True, was_random=False, was_informed=False)
    assert (tally.num_successes, tally.num_attempts) == (2, 3)
    assert (tally.num_random_successes, tally.num_random_attempts) == (1, 1)
    assert (tally.num_informed_successes, tally.num_informed_attempts) == (0, 1)
    assert tally.num_fallback_attempts() == 1
    assert tally.num_fallback_successes() == 1


def test_an_attempt_cannot_be_both_random_and_informed() -> None:
    """SamplerChoice already forbids the combination; recording it would silently
    double-count one execution into two pools."""
    with pytest.raises(ValueError, match="never informed"):
        SkillPracticeTally().with_attempt(success=True, was_random=True, was_informed=True)


def test_minus_differences_two_cumulative_readings() -> None:
    earlier = SkillPracticeTally(num_attempts=4, num_successes=1, num_informed_attempts=2)
    later = SkillPracticeTally(num_attempts=10, num_successes=6, num_informed_attempts=5)
    assert later.minus(previous=earlier) == SkillPracticeTally(
        num_attempts=6, num_successes=5, num_informed_attempts=3
    )


def test_a_counter_that_went_backwards_is_rejected_rather_than_clamped() -> None:
    """Callers difference two cumulative readings, so a negative delta means a
    counter went backwards -- a bug worth surfacing, not averaging away."""
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=1).minus(previous=SkillPracticeTally(num_attempts=3))


def test_plus_sums_two_windows() -> None:
    first = SkillPracticeTally(num_attempts=3, num_successes=1, num_random_attempts=2)
    second = SkillPracticeTally(num_attempts=5, num_successes=4, num_random_attempts=1)
    assert first.plus(other=second) == SkillPracticeTally(
        num_attempts=8, num_successes=5, num_random_attempts=3
    )


def test_more_successes_than_attempts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=2, num_successes=3)


def test_more_pooled_attempts_than_attempts_is_rejected() -> None:
    """random + informed can at most account for every attempt; the remainder is the
    uniform-fallback pool, so an overflow means one execution was filed twice."""
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=2, num_random_attempts=2, num_informed_attempts=1)


def test_a_pool_cannot_succeed_more_often_than_it_was_attempted() -> None:
    with pytest.raises(ValidationError):
        SkillPracticeTally(
            num_attempts=5, num_successes=5, num_informed_attempts=1, num_informed_successes=2
        )


def test_a_negative_counter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=-1)
