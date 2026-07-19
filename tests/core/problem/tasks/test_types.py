from unittest.mock import Mock

import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate, Task

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


def test_goal_is_satisfied_true_when_no_atoms() -> None:
    goal = Goal(atoms=frozenset())
    assert goal.is_satisfied(state=_state(x=0.0)) is True


def test_goal_is_satisfied_true_when_all_atoms_hold() -> None:
    is_one = Predicate(name="is-one", types=(_BLOCK,), holds=lambda s, objs: s[objs[0]][0] == 1.0)
    state = _state(x=1.0)
    goal = Goal(atoms=frozenset({is_one(state=state, objects=(_OBJ,))}))
    assert goal.is_satisfied(state=state) is True


def test_goal_is_satisfied_false_when_one_atom_fails() -> None:
    is_one = Predicate(name="is-one", types=(_BLOCK,), holds=lambda s, objs: s[objs[0]][0] == 1.0)
    is_two = Predicate(name="is-two", types=(_BLOCK,), holds=lambda s, objs: s[objs[0]][0] == 2.0)
    state = _state(x=1.0)
    goal = Goal(
        atoms=frozenset({
            is_one(state=state, objects=(_OBJ,)),
            is_two(state=state, objects=(_OBJ,)),
        })
    )
    assert goal.is_satisfied(state=state) is False


def test_predicate_call_grounds_without_evaluating_holds() -> None:
    holds = Mock(return_value=True)
    predicate = Predicate(name="never-checked", types=(_BLOCK,), holds=holds)
    atom = predicate(state=_state(x=0.0), objects=(_OBJ,))
    assert isinstance(atom, GroundAtom)
    assert atom.predicate is predicate
    assert atom.objects == (_OBJ,)
    holds.assert_not_called()


def test_ground_atoms_with_equal_content_are_equal_and_hashable() -> None:
    predicate = Predicate(name="p", types=(_BLOCK,), holds=lambda s, objs: True)
    state = _state(x=0.0)
    atom1 = predicate(state=state, objects=(_OBJ,))
    atom2 = predicate(state=state, objects=(_OBJ,))
    assert atom1 == atom2
    assert hash(atom1) == hash(atom2)


def test_task_bundles_initial_state_and_goal() -> None:
    state = _state(x=0.0)
    goal = Goal(atoms=frozenset())
    task = Task(initial_state=state, goal=goal)
    assert task.initial_state is state
    assert task.goal is goal
