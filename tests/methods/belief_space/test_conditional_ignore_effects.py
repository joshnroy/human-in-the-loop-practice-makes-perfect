"""Search metadata and symmetry must account for guarded predicate forgetting."""

from hitl_pmp.core.method.types import (
    ConditionalIgnoreEffect,
    GroundSkill,
    LiftedAtom,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.object_symmetry import InterchangeableObjects, _Renamer
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel

_TYPE = Type(name="cube", feature_names=())
_GUARD = Predicate(name="DropGuard", types=(_TYPE,), holds=lambda state, objects: False)
_GEOMETRY = Predicate(name="DropGeometry", types=(_TYPE,), holds=lambda state, objects: False)
_VARIABLE = Variable(name="cube", type=_TYPE)
_FIRST = Object(name="first", type=_TYPE)
_SECOND = Object(name="second", type=_TYPE)


def _ground(*, obj, ignored=_GEOMETRY):
    return GroundSkill(
        skill=Skill(
            name="OpenGripper",
            parameters=(_VARIABLE,),
            preconditions=frozenset(),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            conditional_ignore_effects=frozenset({
                ConditionalIgnoreEffect(
                    conditions=frozenset({LiftedAtom(predicate=_GUARD, variables=(_VARIABLE,))}),
                    ignore_effects=frozenset({ignored}),
                )
            }),
            param_dim=0,
            practice_cost=1.0,
        ),
        objects=(obj,),
    )


def test_search_indexes_conditions_and_marks_conditionally_forgotten_predicates_dynamic():
    skill = _ground(obj=_FIRST)
    model = Tossing3DPracticeModel(ground_skills=(skill,))
    assert GroundAtom(predicate=_GUARD, objects=(_FIRST,)) in model._atom_indexes
    assert _GEOMETRY.name in model._dynamic_predicates


def test_symmetry_signature_includes_guard_objects_and_forgotten_predicates():
    renamer = _Renamer(
        dynamic_predicates=frozenset({_GUARD.name, _GEOMETRY.name}),
        cost_of=lambda skill: 1.0,
    )
    first = renamer.signature(skill=_ground(obj=_FIRST))
    second = renamer.signature(skill=_ground(obj=_SECOND))
    assert first != second
    assert renamer.rename_signature(signature=first, mapping={_FIRST: _SECOND}) == second
    assert first != renamer.signature(skill=_ground(obj=_FIRST, ignored=_GUARD))
    assert GroundAtom(predicate=_GUARD, objects=(_FIRST,)) in set(
        renamer.signature_atoms(signature=first)
    )


def test_static_conditional_ignore_guards_disable_unsafe_symmetry_reduction():
    symmetry = InterchangeableObjects.detect(
        ground_skills=(_ground(obj=_FIRST), _ground(obj=_SECOND)),
        dynamic_predicates=frozenset({_GEOMETRY.name}),
        static_atoms=frozenset({GroundAtom(predicate=_GUARD, objects=(_FIRST,))}),
        cost_of=lambda skill: 1.0,
    )
    assert not symmetry.classes
