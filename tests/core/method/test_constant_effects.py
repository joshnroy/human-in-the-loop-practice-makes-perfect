import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    ConditionalAddEffect,
    ConditionalIgnoreEffect,
    GroundSkill,
    LiftedAtom,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

_TYPE = Type(name="item", feature_names=())
_OTHER_TYPE = Type(name="other", feature_names=())
_PARAMETER = Variable(name="item", type=_TYPE)
_LOCAL = Variable(name="local", type=_TYPE)
_OBJECT = Object(name="item1", type=_TYPE)
_LEFT = Object(name="left", type=_TYPE)
_RIGHT = Object(name="right", type=_TYPE)
_READY = Predicate(name="Ready", types=(_TYPE,), holds=lambda state, objects: False)
_AT = Predicate(name="At", types=(_TYPE, _TYPE), holds=lambda state, objects: False)


def _effect(*, constant=_LEFT):
    return ConditionalAddEffect(
        conditions=frozenset({LiftedAtom(predicate=_READY, variables=(_LOCAL,))}),
        add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(_PARAMETER, _LOCAL))}),
        constant_bindings=((_LOCAL, constant),),
    )


def _skill(**updates):
    return Skill(**{
        "name": "Place",
        "parameters": (_PARAMETER,),
        "preconditions": frozenset(),
        "add_effects": frozenset(),
        "delete_effects": frozenset(),
        "conditional_add_effects": frozenset({_effect(), _effect(constant=_RIGHT)}),
        "param_dim": 0,
        **updates,
    })


def test_bindings_ground_both_guards_and_additions_without_mutating_substitution():
    substitution = {_PARAMETER: _OBJECT}
    effect = _effect().ground(substitution=substitution)
    assert substitution == {_PARAMETER: _OBJECT}
    assert effect.conditions == {GroundAtom(predicate=_READY, objects=(_LEFT,))}
    assert effect.add_effects == {GroundAtom(predicate=_AT, objects=(_OBJECT, _LEFT))}


@pytest.mark.parametrize("constant", [_LEFT, _RIGHT])
def test_reusing_local_variable_in_other_effects_does_not_change_action_identity(*, constant):
    skill = _skill()
    ground = GroundSkill(skill=skill, objects=(_OBJECT,))
    assert skill.parameters == (_PARAMETER,)
    assert ground.objects == (_OBJECT,)
    assert len(ground.conditional_add_effects) == 2
    assert ground.expected_add_effects(
        before_atoms=frozenset({GroundAtom(predicate=_READY, objects=(constant,))})
    ) == {GroundAtom(predicate=_AT, objects=(_OBJECT, constant))}
    assert ground.expected_add_effects(before_atoms=frozenset()) == frozenset()


def test_constant_bindings_are_hashable_frozen_and_roundtrip_through_json():
    effect = ConditionalAddEffect(
        conditions=frozenset(), add_effects=frozenset(), constant_bindings=((_LOCAL, _LEFT),)
    )
    restored = ConditionalAddEffect.model_validate_json(effect.model_dump_json())
    assert restored == effect
    assert restored.constant_bindings == ((_LOCAL, _LEFT),)
    assert {restored} == {effect}
    skill = _skill(conditional_add_effects=frozenset({effect}))
    assert Skill.model_validate_json(skill.model_dump_json()) == skill
    with pytest.raises(ValidationError, match="frozen"):
        effect.constant_bindings = ()


@pytest.mark.parametrize("constant", [_LEFT, _RIGHT])
def test_duplicate_variable_bindings_are_rejected_even_when_values_agree(*, constant):
    with pytest.raises(ValidationError, match="Duplicate constant binding"):
        ConditionalAddEffect(
            conditions=frozenset(),
            add_effects=frozenset(),
            constant_bindings=((_LOCAL, _LEFT), (_LOCAL, constant)),
        )


def test_binding_types_must_match_including_feature_schema():
    for object_type in (_OTHER_TYPE, Type(name=_TYPE.name, feature_names=("position",))):
        with pytest.raises(ValidationError, match="but variable declares"):
            _effect(constant=Object(name="wrong", type=object_type))


@pytest.mark.parametrize("different_type", [False, True])
def test_bindings_cannot_shadow_parameter_names(*, different_type):
    parameter = Variable(name=_LOCAL.name, type=_OTHER_TYPE if different_type else _TYPE)
    with pytest.raises(ValidationError, match="cannot shadow"):
        _skill(parameters=(_PARAMETER, parameter))
    with pytest.raises(ValueError, match="cannot shadow"):
        _effect().ground(substitution={parameter: _LEFT})


@pytest.mark.parametrize("field", ["preconditions", "add_effects", "delete_effects"])
def test_local_binding_does_not_declare_variables_in_unconditional_atoms(*, field):
    with pytest.raises(ValidationError, match="not in its own parameters"):
        _skill(**{field: frozenset({LiftedAtom(predicate=_READY, variables=(_LOCAL,))})})


@pytest.mark.parametrize("field", ["conditions", "add_effects"])
def test_local_binding_does_not_declare_variables_in_other_add_effects(*, field):
    unbound = ConditionalAddEffect(**{
        "conditions": frozenset(),
        "add_effects": frozenset(),
        field: frozenset({LiftedAtom(predicate=_READY, variables=(_LOCAL,))}),
    })
    with pytest.raises(ValidationError, match="not in its own parameters"):
        _skill(conditional_add_effects=frozenset({_effect(), unbound}))


def test_local_binding_does_not_declare_variables_in_conditional_ignore_guards():
    effect = ConditionalIgnoreEffect(
        conditions=frozenset({LiftedAtom(predicate=_READY, variables=(_LOCAL,))}),
        ignore_effects=frozenset({_AT}),
    )
    with pytest.raises(ValidationError, match="not in its own parameters"):
        _skill(conditional_ignore_effects=frozenset({effect}))


def test_unbound_variables_inside_a_constant_effect_are_still_rejected():
    with pytest.raises(ValidationError, match="not in its own parameters"):
        _skill(parameters=())


def test_legacy_effect_defaults_to_no_bindings():
    effect = ConditionalAddEffect(
        conditions=frozenset({LiftedAtom(predicate=_READY, variables=(_PARAMETER,))}),
        add_effects=frozenset(),
    )
    assert effect.constant_bindings == ()
    assert effect.ground(substitution={_PARAMETER: _OBJECT}).conditions == {
        GroundAtom(predicate=_READY, objects=(_OBJECT,))
    }
