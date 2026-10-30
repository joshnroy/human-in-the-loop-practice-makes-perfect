import pytest

from hitl_pmp.core.method.types import (
    ConditionalAddEffect,
    GroundSkill,
    LiftedAtom,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder
from hitl_pmp.planning.pddl import PddlWriter

_ITEM_TYPE = Type(name="item", feature_names=())
_SIDE_TYPE = Type(name="side", feature_names=())
_ITEM = Variable(name="item", type=_ITEM_TYPE)
_SIDE = Variable(name="local_side", type=_SIDE_TYPE)
_OBJECT = Object(name="item1", type=_ITEM_TYPE)
_LEFT = Object(name="robot_side", type=_SIDE_TYPE)
_RIGHT = Object(name="opposite_side", type=_SIDE_TYPE)
_READY = Predicate(name="Ready", types=(_SIDE_TYPE,), holds=lambda state, objects: False)
_AT = Predicate(name="At", types=(_ITEM_TYPE, _SIDE_TYPE), holds=lambda state, objects: False)
_TYPES = (_ITEM_TYPE, _SIDE_TYPE)
_PREDICATES = (_READY, _AT)


def _skill(*, constants=(_LEFT, _RIGHT)):
    return Skill(
        name="Place",
        parameters=(_ITEM,),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        conditional_add_effects=frozenset(
            ConditionalAddEffect(
                conditions=frozenset({LiftedAtom(predicate=_READY, variables=(_SIDE,))}),
                add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(_ITEM, _SIDE))}),
                constant_bindings=((_SIDE, constant),),
            )
            for constant in constants
        ),
        param_dim=0,
    )


def test_pddl_emits_deduplicated_typed_constants_and_effect_local_references():
    skill = _skill()
    other = skill.model_copy(update={"name": "Other"})
    domain = PddlWriter.domain_str(skills=(skill, other), predicates=_PREDICATES, types=_TYPES)
    assert domain.count("(:constants") == 1
    assert domain.count("robot_side - side") == 1
    assert domain.count("opposite_side - side") == 1
    assert "(when (and (Ready robot_side)) (and (At ?item robot_side)))" in domain
    assert "(when (and (Ready opposite_side)) (and (At ?item opposite_side)))" in domain
    assert "?local_side" not in domain
    assert domain.count(":parameters (?item - item)") == 2
    assert PddlWriter.collect_constants(skills=(skill, other)) == (_RIGHT, _LEFT)
    assert domain == PddlWriter.domain_str(
        skills=(other, skill), predicates=_PREDICATES[::-1], types=_TYPES[::-1]
    )


def test_problem_omits_constants_only_from_objects_and_preserves_init_and_goal():
    problem = PddlWriter.problem_str(
        objects=(_LEFT, _OBJECT, _RIGHT),
        constants=PddlWriter.collect_constants(skills=(_skill(),)),
        init_atoms=frozenset({GroundAtom(predicate=_READY, objects=(_LEFT,))}),
        goal=frozenset({GroundAtom(predicate=_AT, objects=(_OBJECT, _RIGHT))}),
    )
    declarations = problem.split("(:objects")[1].split("(:init")[0]
    assert "item1 - item" in declarations
    assert "robot_side" not in declarations
    assert "opposite_side" not in declarations
    assert "(Ready robot_side)" in problem
    assert "(At item1 opposite_side)" in problem


@pytest.mark.parametrize("name", [_LEFT.name, _LEFT.name.upper()])
def test_conflicting_constant_types_across_skills_are_rejected(*, name):
    other_type = Type(name="other", feature_names=())
    variable = Variable(name="other", type=other_type)
    effect = ConditionalAddEffect(
        conditions=frozenset(),
        add_effects=frozenset(),
        constant_bindings=((variable, Object(name=name, type=other_type)),),
    )
    other = _skill().model_copy(
        update={"name": "Other", "conditional_add_effects": frozenset({effect})}
    )
    with pytest.raises(ValueError, match="Conflicting domain constants"):
        PddlWriter.domain_str(skills=(_skill(), other), predicates=_PREDICATES, types=_TYPES)


def test_problem_rejects_conflicting_object_and_constant_declarations():
    with pytest.raises(ValueError, match="conflicts with a domain constant"):
        PddlWriter.problem_str(
            objects=(Object(name=_LEFT.name, type=_ITEM_TYPE),),
            constants=(_LEFT,),
            init_atoms=frozenset(),
            goal=frozenset(),
        )


def test_pddl_rejects_case_insensitive_parameter_overlap():
    skill = _skill().model_copy(
        update={"parameters": (_ITEM, Variable(name=_SIDE.name.upper(), type=_SIDE_TYPE))}
    )
    with pytest.raises(ValueError, match="cannot shadow"):
        PddlWriter.domain_str(skills=(skill,), predicates=_PREDICATES, types=_TYPES)


def test_constant_names_cannot_carry_variable_sigils():
    with pytest.raises(ValueError, match="already carries"):
        PddlWriter.domain_str(
            skills=(_skill(constants=(Object(name="?bad", type=_SIDE_TYPE),)),),
            predicates=_PREDICATES,
            types=_TYPES,
        )


def test_constants_do_not_multiply_groundings():
    skill = _skill()
    assert SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=(_OBJECT, _LEFT, _RIGHT), true_atoms=frozenset()
    ) == [GroundSkill(skill=skill, objects=(_OBJECT,))]


@pytest.mark.parametrize("side", [_LEFT, _RIGHT])
@pytest.mark.parametrize("include_constants_in_objects", [False, True])
def test_fast_downward_executes_constant_conditional_effects(*, side, include_constants_in_objects):
    skill = _skill()
    ground = GroundSkill(skill=skill, objects=(_OBJECT,))
    before = frozenset({GroundAtom(predicate=_READY, objects=(side,))})
    goal = frozenset({GroundAtom(predicate=_AT, objects=(_OBJECT, side))})
    kwargs = {
        "skills": (skill,),
        "predicates": _PREDICATES,
        "types": _TYPES,
        "objects": (_OBJECT, _LEFT, _RIGHT) if include_constants_in_objects else (_OBJECT,),
        "init_atoms": before,
    }
    plan = FastDownwardPlanner.plan(**kwargs, goal=goal)
    assert plan == [ground]
    assert plan[0].expected_add_effects(before_atoms=before) == goal
    opposite = _RIGHT if side == _LEFT else _LEFT
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(
            **kwargs, goal=frozenset({GroundAtom(predicate=_AT, objects=(_OBJECT, opposite))})
        )


@pytest.mark.parametrize("side", [_LEFT, _RIGHT])
def test_fast_downward_evaluates_constant_guards_after_an_earlier_action(*, side):
    enable = Skill(
        name="Enable",
        parameters=(),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset(),
                add_effects=frozenset({LiftedAtom(predicate=_READY, variables=(_SIDE,))}),
                constant_bindings=((_SIDE, side),),
            )
        }),
        param_dim=0,
    )
    place = _skill()
    goal = frozenset({GroundAtom(predicate=_AT, objects=(_OBJECT, side))})
    plan = FastDownwardPlanner.plan(
        skills=(enable, place),
        predicates=_PREDICATES,
        types=_TYPES,
        objects=(_OBJECT, _LEFT, _RIGHT),
        init_atoms=frozenset(),
        goal=goal,
    )
    assert plan == [
        GroundSkill(skill=enable, objects=()),
        GroundSkill(skill=place, objects=(_OBJECT,)),
    ]
    atoms = frozenset()
    for step in plan:
        atoms |= step.expected_add_effects(before_atoms=atoms)
    assert goal <= atoms


def test_nonconstant_conditional_pddl_remains_byte_compatible():
    ready = LiftedAtom(predicate=_READY, variables=(_SIDE,))
    skill = Skill(
        name="Mark",
        parameters=(_SIDE,),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(conditions=frozenset({ready}), add_effects=frozenset({ready}))
        }),
        param_dim=0,
    )
    assert (
        PddlWriter.domain_str(skills=(skill,), predicates=(_READY,), types=(_SIDE_TYPE,))
        == """(define (domain hitlpmp)
  (:requirements :typing :conditional-effects)
  (:types side)

  (:predicates
    (Ready ?x0 - side)
  )

  (:action Mark
    :parameters (?local_side - side)
    :precondition (and )
    :effect (and (when (and (Ready ?local_side)) (and (Ready ?local_side))))
  )
)"""
    )
    assert PddlWriter.problem_str(
        objects=(_LEFT,), init_atoms=frozenset(), goal=frozenset()
    ) == "\n".join([
        "(define (problem hitlpmpproblem) (:domain hitlpmp)",
        "  (:objects",
        "    robot_side - side",
        "  )",
        "  (:init",
        "    ",
        "  )",
        "  (:goal (and ))",
        ")",
        "",
    ])
