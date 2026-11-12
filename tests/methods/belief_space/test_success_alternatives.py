"""The forecast branches a success over the skill's declared alternatives."""

import math

import pytest

from hitl_pmp.core.method.types import (
    GroundSkill,
    LiftedAtom,
    Skill,
    SuccessAlternative,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.skill_plans_domain import SkillPlansPracticeDomain
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource

_ITEM = Type(name="item", feature_names=())
_A = Variable(name="a", type=_ITEM)
_B = Variable(name="b", type=_ITEM)
_ONE = Object(name="one", type=_ITEM)
_TWO = Object(name="two", type=_ITEM)
_HELD = Predicate(name="Held", types=(_ITEM,), holds=lambda state, objects: False)
_DOWN = Predicate(name="Down", types=(_ITEM,), holds=lambda state, objects: False)
_DONE = Predicate(name="Done", types=(), holds=lambda state, objects: False)
_HELD_ONE = GroundAtom(predicate=_HELD, objects=(_ONE,))
_HELD_TWO = GroundAtom(predicate=_HELD, objects=(_TWO,))
_DONE_ATOM = GroundAtom(predicate=_DONE, objects=())
_START = frozenset({_HELD_ONE, _HELD_TWO})


def _alternative(*, moved: Variable) -> SuccessAlternative:
    return SuccessAlternative(
        add_effects=frozenset({LiftedAtom(predicate=_DOWN, variables=(moved,))}),
        delete_effects=frozenset({LiftedAtom(predicate=_HELD, variables=(moved,))}),
    )


def _ground(*, alternatives: bool, param_dim: int) -> GroundSkill:
    return GroundSkill(
        skill=Skill(
            name="PutOneDown",
            parameters=(_A, _B),
            preconditions=frozenset({
                LiftedAtom(predicate=_HELD, variables=(_A,)),
                LiftedAtom(predicate=_HELD, variables=(_B,)),
            }),
            add_effects=frozenset({LiftedAtom(predicate=_DONE, variables=())}),
            delete_effects=frozenset(),
            param_dim=param_dim,
            success_alternatives=(_alternative(moved=_A), _alternative(moved=_B))
            if alternatives
            else (),
        ),
        objects=(_ONE, _TWO),
    )


def _model(*, ground: GroundSkill, source: PracticeExampleSource, **fields: object):  # noqa: ANN202
    domain = SkillPlansPracticeDomain(
        deployment_horizon=1,
        deployment_plans=(("PutOneDown",),),
        belief_sources=(("PutOneDown", source),),
    )
    model = Tossing3DPracticeModel(domain=domain, ground_skills=(ground,), **fields)
    state = make_default_tossing3d_belief(domain=domain, num_particles=16)
    return model, state, make_tossing3d_search_state(state=state, true_atoms=_START)


SOURCES = pytest.mark.parametrize(
    ("source", "param_dim"),
    [(PracticeExampleSource.SAMPLER, 3), (PracticeExampleSource.OUTCOME, 0)],
)


@SOURCES
def test_a_success_becomes_one_equally_likely_branch_per_alternative(
    *, source: PracticeExampleSource, param_dim: int
) -> None:
    ground = _ground(alternatives=True, param_dim=param_dim)
    model, state, search = _model(ground=ground, source=source)
    plain_model, plain_state, plain_search = _model(
        ground=_ground(alternatives=False, param_dim=param_dim), source=source
    )
    outcomes = model.outcomes(environment_state=search, state=state, action=ground)
    plain = plain_model.outcomes(
        environment_state=plain_search,
        state=plain_state,
        action=plain_model.ground_skills[0],
    )
    assert sum(p for p, _, _ in outcomes) == pytest.approx(1.0)
    by_atoms: dict[frozenset[GroundAtom], float] = {}
    for probability, _, atoms in outcomes:
        by_atoms[atoms] = by_atoms.get(atoms, 0.0) + probability
    first = frozenset({_DONE_ATOM, _HELD_TWO, GroundAtom(predicate=_DOWN, objects=(_ONE,))})
    second = frozenset({_DONE_ATOM, _HELD_ONE, GroundAtom(predicate=_DOWN, objects=(_TWO,))})
    assert set(by_atoms) == {first, second, _START}
    plain_success = sum(p for p, _, atoms in plain if _DONE_ATOM in atoms)
    assert by_atoms[first] == pytest.approx(plain_success / 2)
    assert by_atoms[second] == pytest.approx(plain_success / 2)
    assert by_atoms[_START] == pytest.approx(1.0 - plain_success)


@SOURCES
def test_the_belief_update_of_a_success_does_not_depend_on_the_alternative(
    *, source: PracticeExampleSource, param_dim: int
) -> None:
    ground = _ground(alternatives=True, param_dim=param_dim)
    model, state, search = _model(ground=ground, source=source)
    plain_model, plain_state, plain_search = _model(
        ground=_ground(alternatives=False, param_dim=param_dim), source=source
    )
    successes = [
        after
        for _, after, atoms in model.outcomes(environment_state=search, state=state, action=ground)
        if _DONE_ATOM in atoms
    ]
    plain_successes = [
        after
        for _, after, atoms in plain_model.outcomes(
            environment_state=plain_search,
            state=plain_state,
            action=plain_model.ground_skills[0],
        )
        if _DONE_ATOM in atoms
    ]
    assert len(successes) == 2 * len(plain_successes)
    assert successes[0::2] == successes[1::2] == plain_successes


@pytest.mark.parametrize("variant", ["curtis", "algorithm3"])
def test_each_alternative_is_its_own_search_child_and_the_chances_sum_to_one(
    *, variant: str
) -> None:
    ground = _ground(alternatives=True, param_dim=3)
    model, state, search = _model(ground=ground, source=PracticeExampleSource.SAMPLER)
    children = model.transition_outcomes(
        environment_state=search, practice_action=ground, belief_state=state
    )
    assert sum(probability for _, _, probability in children) == pytest.approx(1.0)
    succeeded = [child for child, _, _ in children if _DONE_ATOM in child.true_atoms]
    assert len({child.true_atoms for child in succeeded}) == 2
    # The planner asserts every chance distribution it samples or branches on.
    DeterminizedAStarPlanner(max_iterations=20, seed=0, variant=variant).solve(  # type: ignore[arg-type]
        environment_state=search,
        summed_cost=state.accumulated_cost,
        belief_state=state,
        remaining_actions=2,
        model=model,
        num_samples=1,
    )


def test_expectimax_integrates_over_the_alternatives() -> None:
    ground = _ground(alternatives=True, param_dim=3)
    model, state, search = _model(ground=ground, source=PracticeExampleSource.SAMPLER)
    # Expectimax asserts each chance node's probabilities sum to one.
    value, _ = ExpectimaxPlanner(use_model_j=True).solve(
        environment_state=search,
        summed_cost=state.accumulated_cost,
        belief_state=state,
        horizon=2,
        model=model,
    )
    assert math.isfinite(value)


def test_exact_object_symmetry_refuses_a_skill_it_does_not_model() -> None:
    ground = _ground(alternatives=True, param_dim=3)
    with pytest.raises(ValueError, match="success alternatives"):
        _model(ground=ground, source=PracticeExampleSource.SAMPLER, object_symmetry=True)
