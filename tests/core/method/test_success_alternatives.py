"""A skill may declare several mutually exclusive things a success leaves behind."""

import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    Skill,
    SuccessAlternative,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.pddl import PddlWriter

_ITEM = Type(name="item", feature_names=())
_PLACE = Type(name="place", feature_names=())
_A = Variable(name="a", type=_ITEM)
_B = Variable(name="b", type=_ITEM)
_WHERE = Variable(name="where", type=_PLACE)
_ONE = Object(name="one", type=_ITEM)
_TWO = Object(name="two", type=_ITEM)
_SHELF = Object(name="shelf", type=_PLACE)
_HELD = Predicate(name="Held", types=(_ITEM,), holds=lambda state, objects: False)
_AT = Predicate(name="At", types=(_ITEM, _PLACE), holds=lambda state, objects: False)
_DONE = Predicate(name="Done", types=(), holds=lambda state, objects: False)


def _alternative(*, moved: Variable) -> SuccessAlternative:
    return SuccessAlternative(
        add_effects=frozenset({LiftedAtom(predicate=_AT, variables=(moved, _WHERE))}),
        delete_effects=frozenset({LiftedAtom(predicate=_HELD, variables=(moved,))}),
        constant_bindings=((_WHERE, _SHELF),),
    )


def _skill(**updates: object) -> Skill:
    return Skill(**{
        "name": "PutOneDown",
        "parameters": (_A, _B),
        "preconditions": frozenset({
            LiftedAtom(predicate=_HELD, variables=(_A,)),
            LiftedAtom(predicate=_HELD, variables=(_B,)),
        }),
        "add_effects": frozenset({LiftedAtom(predicate=_DONE, variables=())}),
        "delete_effects": frozenset(),
        "param_dim": 0,
        "success_alternatives": (_alternative(moved=_A), _alternative(moved=_B)),
        **updates,
    })


def test_alternatives_ground_with_the_skills_objects_and_their_own_constants() -> None:
    ground = GroundSkill(skill=_skill(), objects=(_ONE, _TWO))
    first, second = ground.success_alternatives
    assert first.add_effects == {GroundAtom(predicate=_AT, objects=(_ONE, _SHELF))}
    assert first.delete_effects == {GroundAtom(predicate=_HELD, objects=(_ONE,))}
    assert second.add_effects == {GroundAtom(predicate=_AT, objects=(_TWO, _SHELF))}
    assert second.delete_effects == {GroundAtom(predicate=_HELD, objects=(_TWO,))}


def test_a_skill_declares_no_alternatives_unless_it_says_so() -> None:
    plain = _skill(success_alternatives=())
    assert Skill.model_fields["success_alternatives"].default == ()
    assert GroundSkill(skill=plain, objects=(_ONE, _TWO)).success_alternatives == ()


def test_alternatives_change_neither_the_judged_effects_nor_the_planners_pddl() -> None:
    with_alternatives, plain = _skill(), _skill(success_alternatives=())
    ground = GroundSkill(skill=with_alternatives, objects=(_ONE, _TWO))
    assert ground.expected_add_effects(before_atoms=frozenset()) == {
        GroundAtom(predicate=_DONE, objects=())
    }
    assert ground.delete_effects == frozenset()

    def domain(*, skill: Skill) -> str:
        return PddlWriter.domain_str(
            skills=(skill,), predicates=(_HELD, _AT, _DONE), types=(_ITEM, _PLACE)
        )

    assert domain(skill=with_alternatives) == domain(skill=plain)


def test_alternatives_do_not_change_the_text_logs_and_cost_tables_are_keyed_by() -> None:
    with_alternatives, plain = _skill(), _skill(success_alternatives=())
    assert with_alternatives != plain
    assert str(with_alternatives) == str(plain)
    assert str(GroundSkill(skill=with_alternatives, objects=(_ONE, _TWO))) == str(
        GroundSkill(skill=plain, objects=(_ONE, _TWO))
    )


def test_an_alternative_may_only_name_parameters_and_its_own_constants() -> None:
    stray = Variable(name="stray", type=_ITEM)
    with pytest.raises(ValidationError, match="stray"):
        _skill(
            success_alternatives=(
                SuccessAlternative(
                    add_effects=frozenset({LiftedAtom(predicate=_HELD, variables=(stray,))})
                ),
            )
        )
    with pytest.raises(ValidationError, match="shadow"):
        _skill(
            success_alternatives=(
                SuccessAlternative(
                    add_effects=frozenset({LiftedAtom(predicate=_HELD, variables=(_A,))}),
                    constant_bindings=((_A, _ONE),),
                ),
            )
        )
