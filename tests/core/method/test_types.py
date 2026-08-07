import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    PracticeTargetTally,
    Rollout,
    SamplerConsultation,
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
    assert tally.num_unparameterized_attempts == 0
    assert tally.num_unparameterized_successes == 0


def test_with_attempt_files_one_execution_into_exactly_one_pool() -> None:
    """Every attempt lands in exactly one of the four pools, which is what makes all
    four recoverable from eight numbers: uninformative is the remainder."""
    tally = SkillPracticeTally()
    tally = tally.with_attempt(success=True, consultation=SamplerConsultation.EPSILON_RANDOM)
    tally = tally.with_attempt(success=False, consultation=SamplerConsultation.INFORMED)
    tally = tally.with_attempt(success=True, consultation=SamplerConsultation.UNINFORMATIVE)
    tally = tally.with_attempt(success=False, consultation=SamplerConsultation.NO_SAMPLER)
    assert (tally.num_successes, tally.num_attempts) == (2, 4)
    assert (tally.num_random_successes, tally.num_random_attempts) == (1, 1)
    assert (tally.num_informed_successes, tally.num_informed_attempts) == (0, 1)
    assert (tally.num_unparameterized_successes, tally.num_unparameterized_attempts) == (0, 1)
    assert (tally.num_uninformative_successes(), tally.num_uninformative_attempts()) == (1, 1)
    # The #111 pool, unchanged in meaning: the union of the last two.
    assert tally.num_fallback_attempts() == 2
    assert tally.num_fallback_successes() == 1


def test_the_four_pools_always_reconcile_against_the_total() -> None:
    """The invariant that makes the instrument trustworthy: whatever sequence of
    consultations arrives, the three stored pools plus the derived remainder account
    for every attempt and every success exactly once. Checked over all four values so
    no future pool can be added without either closing or failing this."""
    tally = SkillPracticeTally()
    for index, consultation in enumerate(SamplerConsultation):
        for _ in range(index + 1):
            tally = tally.with_attempt(success=index % 2 == 0, consultation=consultation)
    assert (
        tally.num_random_attempts
        + tally.num_informed_attempts
        + tally.num_unparameterized_attempts
        + tally.num_uninformative_attempts()
        == tally.num_attempts
        == 1 + 2 + 3 + 4
    )
    assert (
        tally.num_random_successes
        + tally.num_informed_successes
        + tally.num_unparameterized_successes
        + tally.num_uninformative_successes()
        == tally.num_successes
    )


def test_minus_differences_two_cumulative_readings() -> None:
    earlier = SkillPracticeTally(
        num_attempts=4, num_successes=1, num_informed_attempts=2, num_informed_successes=1
    )
    later = SkillPracticeTally(
        num_attempts=10, num_successes=6, num_informed_attempts=5, num_informed_successes=4
    )
    assert later.minus(previous=earlier) == SkillPracticeTally(
        num_attempts=6, num_successes=5, num_informed_attempts=3, num_informed_successes=3
    )


def test_minus_and_plus_carry_the_unparameterized_pool() -> None:
    """A field the windowing arithmetic forgot would read as zero in every per-window
    record while the run total was right -- the quietest possible way to lose it."""
    earlier = SkillPracticeTally(num_attempts=2, num_unparameterized_attempts=2)
    later = SkillPracticeTally(num_attempts=7, num_unparameterized_attempts=5)
    window = later.minus(previous=earlier)
    assert window.num_unparameterized_attempts == 3
    assert window.plus(other=earlier).num_unparameterized_attempts == 5


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
    """The stored pools can at most account for every attempt; the remainder is the
    uninformative pool, so an overflow means one execution was filed twice."""
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=2, num_random_attempts=2, num_informed_attempts=1)
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=2, num_random_attempts=1, num_unparameterized_attempts=2)


def test_the_derived_pool_cannot_succeed_more_often_than_it_was_attempted() -> None:
    """The hole #111 left open. Every one of its checks passes on this tally -- one
    attempt, one success, both filed as epsilon-random -- while the derived remainder
    comes out at 1 success from 0 attempts, a rate above 1.0 that would have serialized
    into `stats.json` without complaint. Deriving a pool only closes the arithmetic if
    the derived values are validated too."""
    with pytest.raises(ValidationError, match="uninformative successes"):
        SkillPracticeTally(
            num_attempts=1, num_successes=1, num_random_attempts=1, num_random_successes=0
        )


def test_a_pool_cannot_succeed_more_often_than_it_was_attempted() -> None:
    with pytest.raises(ValidationError):
        SkillPracticeTally(
            num_attempts=5, num_successes=5, num_informed_attempts=1, num_informed_successes=2
        )


def test_a_negative_counter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SkillPracticeTally(num_attempts=-1)


# --------------------------------------------------- PracticeTargetTally (selection)


def test_practice_target_tally_defaults_to_all_zero() -> None:
    tally = PracticeTargetTally()
    assert tally.num_scored == 0
    assert tally.num_declined_perfect == 0
    assert tally.num_selected == 0
    assert tally.num_unreachable == 0


def test_practice_target_tally_records_a_declined_perfect_grounding() -> None:
    """The Tossing3D failure state: a grounding whose measured rate hit 1.0, so
    score_ground_skill returned -inf and choose_practice_target dropped it."""
    tally = PracticeTargetTally().with_declined_perfect()
    assert tally.num_declined_perfect == 1
    assert tally.num_scored == 0


def test_practice_target_tally_records_scored_selected_and_unreachable() -> None:
    tally = PracticeTargetTally().with_scored().with_scored().with_selected().with_unreachable()
    assert tally.num_scored == 2
    assert tally.num_selected == 1
    assert tally.num_unreachable == 1


def test_practice_target_tally_plus_sums_every_counter() -> None:
    left = PracticeTargetTally(num_scored=3, num_declined_perfect=1, num_selected=2)
    right = PracticeTargetTally(num_scored=6, num_unreachable=5, num_selected=1)
    total = left.plus(other=right)
    assert total.num_scored == 9
    assert total.num_declined_perfect == 1
    assert total.num_selected == 3
    assert total.num_unreachable == 5


def test_practice_target_tally_minus_differences_a_cumulative_reading() -> None:
    later = PracticeTargetTally(num_scored=7, num_declined_perfect=2, num_selected=3)
    earlier = PracticeTargetTally(num_scored=4, num_declined_perfect=2, num_selected=1)
    delta = later.minus(previous=earlier)
    assert delta.num_scored == 3
    assert delta.num_declined_perfect == 0
    assert delta.num_selected == 2


def test_practice_target_tally_rejects_a_counter_that_went_backwards() -> None:
    with pytest.raises(ValidationError):
        PracticeTargetTally(num_scored=2).minus(previous=PracticeTargetTally(num_scored=5))


def test_practice_target_tally_rejects_more_selections_than_scorings() -> None:
    """A grounding can only be selected out of the ranked list it was scored into,
    so selections above scorings means one of the two counters is being fed wrong."""
    with pytest.raises(ValidationError):
        PracticeTargetTally(num_scored=1, num_selected=2)


def test_practice_target_tally_rejects_selected_plus_unreachable_above_scored() -> None:
    with pytest.raises(ValidationError):
        PracticeTargetTally(num_scored=3, num_selected=2, num_unreachable=2)
