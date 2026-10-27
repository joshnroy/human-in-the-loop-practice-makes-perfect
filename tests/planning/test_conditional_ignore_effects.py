"""Conditional forgetting grounds its guard, not its universally cleared objects."""

import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import (
    ConditionalIgnoreEffect,
    GroundSkill,
    LiftedAtom,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.pddl import PddlWriter

_TYPE = Type(name="thing", feature_names=())
_HELD = Predicate(name="Held", types=(_TYPE,), holds=lambda state, objects: False)
_SIDE = Predicate(name="Side", types=(_TYPE, _TYPE), holds=lambda state, objects: False)
_READY = Predicate(name="Ready", types=(), holds=lambda state, objects: False)
_DONE = Predicate(name="Done", types=(), holds=lambda state, objects: False)
_VARIABLE = Variable(name="x0", type=_TYPE)
_OBJECT = Object(name="cube", type=_TYPE)
_OTHER = Object(name="barrier", type=_TYPE)
_CONDITION = LiftedAtom(predicate=_HELD, variables=(_VARIABLE,))


def _skill(*, conditions=frozenset({_CONDITION})):
    return Skill(
        name="Release",
        parameters=(_VARIABLE,),
        preconditions=frozenset(),
        add_effects=frozenset({LiftedAtom(predicate=_DONE, variables=())}),
        delete_effects=frozenset({_CONDITION}),
        conditional_ignore_effects=frozenset({
            ConditionalIgnoreEffect(
                conditions=conditions, ignore_effects=frozenset({_SIDE, _READY})
            )
        }),
        param_dim=0,
        practice_cost=1.0,
    )


def test_conditional_ignore_variables_must_be_declared():
    undeclared = Variable(name="other", type=_TYPE)
    with pytest.raises(ValidationError, match="not in its own parameters"):
        _skill(conditions=frozenset({LiftedAtom(predicate=_HELD, variables=(undeclared,))}))


@pytest.mark.parametrize("held", [False, True])
def test_grounding_and_transition_evaluate_guard_before_deletion(*, held):
    ground = GroundSkill(skill=_skill(), objects=(_OBJECT,))
    holding = GroundAtom(predicate=_HELD, objects=(_OBJECT,))
    side = GroundAtom(predicate=_SIDE, objects=(_OBJECT, _OTHER))
    other_side = GroundAtom(predicate=_SIDE, objects=(_OTHER, _OBJECT))
    ready = GroundAtom(predicate=_READY, objects=())
    before = frozenset({side, other_side, ready}) | ({holding} if held else set())
    (effect,) = ground.conditional_ignore_effects
    assert effect.conditions == frozenset({holding})
    assert effect.ignore_effects == frozenset({_SIDE, _READY})
    assert ground.expected_ignore_effects(before_atoms=before) == (
        frozenset({_SIDE, _READY}) if held else frozenset()
    )
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=ground,
        effects={ground: (ground.add_effects, ground.delete_effects, ground.ignore_effects)},
    )
    assert holding not in after
    assert (side in after) == (not held)
    assert (other_side in after) == (not held)
    assert (ready in after) == (not held)
    assert ground.add_effects <= after


def test_conditional_ignore_pddl_is_quantified_without_parameter_capture():
    domain = PddlWriter.domain_str(
        skills=(_skill(),), predicates=(_HELD, _SIDE, _READY, _DONE), types=(_TYPE,)
    )
    assert ":conditional-effects" in domain
    assert "(when (and (Held ?x0))" in domain
    assert (
        "(forall (?xx0 - thing ?xx1 - thing) (when (and (Held ?x0)) (not (Side ?xx0 ?xx1))))"
        in domain
    )
    assert "(not (Ready))" in domain
    assert domain == PddlWriter.domain_str(
        skills=(_skill(),), predicates=(_HELD, _SIDE, _READY, _DONE), types=(_TYPE,)
    )
    assert FastDownwardPlanner._search_arguments(skills=(_skill(),), alias="seq-opt-lmcut") == (
        [],
        ["--search", "astar(hmax())"],
    )


@pytest.mark.parametrize("held", [False, True])
def test_classical_planning_preserves_side_only_without_holding(*, held):
    side = GroundAtom(predicate=_SIDE, objects=(_OBJECT, _OTHER))
    before = frozenset({side})
    if held:
        before |= {GroundAtom(predicate=_HELD, objects=(_OBJECT,))}
    skill = _skill().model_copy(
        update={
            "preconditions": frozenset({
                LiftedAtom(predicate=_SIDE, variables=(_VARIABLE, _VARIABLE))
            })
        }
    )
    before |= {GroundAtom(predicate=_SIDE, objects=(_OBJECT, _OBJECT))}
    kwargs = {
        "skills": (skill,),
        "predicates": (_HELD, _SIDE, _READY, _DONE),
        "types": (_TYPE,),
        "objects": (_OBJECT, _OTHER),
        "init_atoms": before,
        "goal": frozenset({side, GroundAtom(predicate=_DONE, objects=())}),
    }
    if held:
        with pytest.raises(PlanningFailure):
            FastDownwardPlanner.plan(**kwargs)
    else:
        plan = FastDownwardPlanner.plan(**kwargs)
        assert plan == [GroundSkill(skill=skill, objects=(_OBJECT,))]


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_multiple_guards_merge_with_unconditional_ignores_using_incoming_state(*, held, ready):
    lifted = _skill().model_copy(
        update={
            "ignore_effects": frozenset({_HELD}),
            "conditional_ignore_effects": frozenset({
                ConditionalIgnoreEffect(
                    conditions=frozenset({_CONDITION, LiftedAtom(predicate=_READY, variables=())}),
                    ignore_effects=frozenset({_SIDE}),
                ),
                ConditionalIgnoreEffect(
                    conditions=frozenset({LiftedAtom(predicate=_DONE, variables=())}),
                    ignore_effects=frozenset({_READY}),
                ),
            }),
        }
    )
    ground = GroundSkill(skill=lifted, objects=(_OBJECT,))
    before = frozenset({GroundAtom(predicate=_SIDE, objects=(_OBJECT, _OTHER))})
    if held:
        before |= {GroundAtom(predicate=_HELD, objects=(_OBJECT,))}
    if ready:
        before |= {GroundAtom(predicate=_READY, objects=())}
    expected = frozenset({_HELD, _SIDE}) if held and ready else frozenset({_HELD})
    assert ground.expected_ignore_effects(before_atoms=before) == expected
    assert _READY not in ground.expected_ignore_effects(before_atoms=before)
    assert _READY in ground.expected_ignore_effects(
        before_atoms=before | ground.expected_add_effects(before_atoms=before)
    )


def test_empty_guard_and_legacy_default_ignore_effects():
    unconditional = GroundSkill(skill=_skill(conditions=frozenset()), objects=(_OBJECT,))
    assert unconditional.expected_ignore_effects(before_atoms=frozenset()) == {_SIDE, _READY}
    legacy = GroundSkill(
        skill=Skill(
            name="Legacy",
            parameters=(),
            preconditions=frozenset(),
            add_effects=frozenset(),
            delete_effects=frozenset(),
            ignore_effects=frozenset({_SIDE}),
            param_dim=0,
        ),
        objects=(),
    )
    assert not legacy.conditional_ignore_effects
    assert legacy.expected_ignore_effects(before_atoms=frozenset()) == legacy.ignore_effects


def test_conditional_ignore_preserves_explicit_planner_alias():
    assert FastDownwardPlanner._search_arguments(skills=(_skill(),), alias="lama-first") == (
        ["--alias", "lama-first"],
        [],
    )


def test_reasserted_atoms_survive_conditional_forgetting_in_both_planners():
    target = LiftedAtom(predicate=_SIDE, variables=(_VARIABLE, _VARIABLE))
    lifted = _skill().model_copy(
        update={
            "preconditions": frozenset({_CONDITION}),
            "add_effects": _skill().add_effects | {target},
        }
    )
    ground = GroundSkill(skill=lifted, objects=(_OBJECT,))
    old_side = GroundAtom(predicate=_SIDE, objects=(_OBJECT, _OTHER))
    before = ground.preconditions | {old_side}
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=ground,
        effects={ground: (ground.add_effects, ground.delete_effects, ground.ignore_effects)},
    )
    assert after == ground.add_effects
    kwargs = {
        "skills": (lifted,),
        "predicates": (_HELD, _SIDE, _READY, _DONE),
        "types": (_TYPE,),
        "objects": (_OBJECT, _OTHER),
        "init_atoms": before,
    }
    assert FastDownwardPlanner.plan(**kwargs, goal=ground.add_effects) == [ground]
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(**kwargs, goal=ground.add_effects | {old_side})
