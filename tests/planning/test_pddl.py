import numpy as np

from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.pddl import PddlWriter

_BLOCK = Type(name="block", feature_names=("x",))
_ROBOT_TYPE = Type(name="robot", feature_names=())
_ROBOT = Object(name="robot", type=_ROBOT_TYPE)
_BLOCK_A = Object(name="block_a", type=_BLOCK)
_BLOCK_B = Object(name="block_b", type=_BLOCK)

# Trivially-true/false predicates are enough here: abstract_state only needs to
# call predicate.holds and record which combinations return True, so these
# tests don't need State to carry any real feature data.
_ROBOT_NEAR_A = Predicate(name="NearA", types=(_ROBOT_TYPE,), holds=lambda state, objects: True)
_ROBOT_NEAR_B = Predicate(name="NearB", types=(_ROBOT_TYPE,), holds=lambda state, objects: False)


def _simple_state() -> State:
    return State(data={_ROBOT: np.array([]), _BLOCK_A: np.array([0.0]), _BLOCK_B: np.array([0.0])})


def test_abstract_state_finds_true_ground_atoms_and_excludes_false_ones() -> None:
    state = _simple_state()
    atoms = PddlWriter.abstract_state(
        state=state, objects=(_ROBOT, _BLOCK_A, _BLOCK_B), predicates=(_ROBOT_NEAR_A, _ROBOT_NEAR_B)
    )
    assert atoms == frozenset({GroundAtom(predicate=_ROBOT_NEAR_A, objects=(_ROBOT,))})


def test_abstract_state_never_repeats_an_object_within_one_atom() -> None:
    always_true = Predicate(
        name="SameBlock", types=(_BLOCK, _BLOCK), holds=lambda state, objects: True
    )
    state = _simple_state()
    atoms = PddlWriter.abstract_state(
        state=state, objects=(_BLOCK_A, _BLOCK_B), predicates=(always_true,)
    )
    # (block_a, block_a) and (block_b, block_b) are skipped -- only the two
    # distinct-object orderings remain.
    assert atoms == frozenset({
        GroundAtom(predicate=always_true, objects=(_BLOCK_A, _BLOCK_B)),
        GroundAtom(predicate=always_true, objects=(_BLOCK_B, _BLOCK_A)),
    })


def _move_skill() -> Skill:
    robot = Variable(name="robot", type=_ROBOT_TYPE)
    source = Variable(name="source", type=_BLOCK)
    target = Variable(name="target", type=_BLOCK)
    return Skill(
        name="Move",
        parameters=(robot, source, target),
        preconditions=frozenset({LiftedAtom(predicate=_ROBOT_NEAR_A, variables=(robot,))}),
        add_effects=frozenset({LiftedAtom(predicate=_ROBOT_NEAR_B, variables=(robot,))}),
        delete_effects=frozenset({LiftedAtom(predicate=_ROBOT_NEAR_A, variables=(robot,))}),
        param_dim=0,
    )


def test_write_domain_declares_types_and_predicates() -> None:
    domain = PddlWriter.write_domain(
        domain_name="mydomain",
        types=(_ROBOT_TYPE, _BLOCK),
        predicates=(_ROBOT_NEAR_A, _ROBOT_NEAR_B),
        skills=(),
    )
    assert "(define (domain mydomain)" in domain
    assert "(:types robot block)" in domain
    assert "(NearA ?x0 - robot)" in domain
    assert "(NearB ?x0 - robot)" in domain


def test_write_domain_declares_an_action_block_with_precondition_and_effect() -> None:
    skill = _move_skill()
    domain = PddlWriter.write_domain(
        domain_name="mydomain",
        types=(_ROBOT_TYPE, _BLOCK),
        predicates=(_ROBOT_NEAR_A, _ROBOT_NEAR_B),
        skills=(skill,),
    )
    assert "(:action Move" in domain
    assert ":parameters (?robot - robot ?source - block ?target - block)" in domain
    assert ":precondition (and (NearA ?robot))" in domain
    assert "(NearB ?robot)" in domain
    assert "(not (NearA ?robot))" in domain


def test_write_problem_declares_objects_init_and_goal() -> None:
    init_atoms = frozenset({GroundAtom(predicate=_ROBOT_NEAR_A, objects=(_ROBOT,))})
    goal_atoms = frozenset({GroundAtom(predicate=_ROBOT_NEAR_B, objects=(_ROBOT,))})
    problem = PddlWriter.write_problem(
        problem_name="myproblem",
        domain_name="mydomain",
        objects=(_ROBOT, _BLOCK_A, _BLOCK_B),
        init_atoms=init_atoms,
        goal_atoms=goal_atoms,
    )
    assert "(define (problem myproblem)" in problem
    assert "(:domain mydomain)" in problem
    assert "robot - robot" in problem
    assert "block_a - block" in problem
    assert "(NearA robot)" in problem
    assert "(:goal (and (NearB robot)))" in problem


def test_write_problem_with_multiple_goal_atoms_ands_them_all() -> None:
    goal_atoms = frozenset({
        GroundAtom(predicate=_ROBOT_NEAR_A, objects=(_ROBOT,)),
        GroundAtom(predicate=_ROBOT_NEAR_B, objects=(_ROBOT,)),
    })
    problem = PddlWriter.write_problem(
        problem_name="myproblem",
        domain_name="mydomain",
        objects=(_ROBOT,),
        init_atoms=frozenset(),
        goal_atoms=goal_atoms,
    )
    assert "(NearA robot)" in problem
    assert "(NearB robot)" in problem
    assert problem.count("(:goal (and") == 1
