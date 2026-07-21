import numpy as np

from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.grounding import SkillGrounder

_ROBOT_TYPE = Type(name="robot", feature_names=())
_CELL_TYPE = Type(name="cell", feature_names=())
_ROBOT = Object(name="robot", type=_ROBOT_TYPE)
_CELL0 = Object(name="cell0", type=_CELL_TYPE)
_CELL1 = Object(name="cell1", type=_CELL_TYPE)
_CELL2 = Object(name="cell2", type=_CELL_TYPE)

_IN_CELL = Predicate(
    name="InCell", types=(_ROBOT_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)
_ADJACENT = Predicate(
    name="Adjacent", types=(_CELL_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)


def _move_skill() -> tuple[Skill, Variable, Variable, Variable]:
    robot = Variable(name="robot", type=_ROBOT_TYPE)
    current = Variable(name="current", type=_CELL_TYPE)
    target = Variable(name="target", type=_CELL_TYPE)
    skill = Skill(
        name="Move",
        parameters=(robot, current, target),
        preconditions=frozenset({
            LiftedAtom(predicate=_IN_CELL, variables=(robot, current)),
            LiftedAtom(predicate=_ADJACENT, variables=(current, target)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=_IN_CELL, variables=(robot, target))}),
        delete_effects=frozenset({LiftedAtom(predicate=_IN_CELL, variables=(robot, current))}),
        param_dim=0,
    )
    return skill, robot, current, target


def test_applicable_ground_skills_finds_only_bindings_consistent_with_true_atoms() -> None:
    skill, *_ = _move_skill()
    true_atoms = frozenset({
        GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
        # cell1<->cell2 adjacency exists too, but the robot isn't in cell1,
        # so no grounding should use it.
        GroundAtom(predicate=_ADJACENT, objects=(_CELL1, _CELL2)),
    })
    ground_skills = SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=(_ROBOT, _CELL0, _CELL1, _CELL2), true_atoms=true_atoms
    )
    assert [gs.objects for gs in ground_skills] == [(_ROBOT, _CELL0, _CELL1)]


def test_applicable_ground_skills_returns_empty_when_nothing_satisfies_preconditions() -> None:
    skill, *_ = _move_skill()
    ground_skills = SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=(_ROBOT, _CELL0, _CELL1), true_atoms=frozenset()
    )
    assert ground_skills == []


def test_applicable_ground_skills_finds_every_consistent_binding() -> None:
    skill, *_ = _move_skill()
    true_atoms = frozenset({
        GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL2)),
    })
    ground_skills = SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=(_ROBOT, _CELL0, _CELL1, _CELL2), true_atoms=true_atoms
    )
    assert {gs.objects for gs in ground_skills} == {
        (_ROBOT, _CELL0, _CELL1),
        (_ROBOT, _CELL0, _CELL2),
    }


def test_applicable_ground_skills_covers_multiple_skills() -> None:
    move_skill, *_ = _move_skill()
    robot = Variable(name="robot2", type=_ROBOT_TYPE)
    noop_skill = Skill(
        name="Noop",
        parameters=(robot,),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        param_dim=0,
    )
    true_atoms = frozenset({
        GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
    })
    ground_skills = SkillGrounder.applicable_ground_skills(
        skills=(move_skill, noop_skill), objects=(_ROBOT, _CELL0, _CELL1), true_atoms=true_atoms
    )
    assert len(ground_skills) == 2
    assert {gs.skill.name for gs in ground_skills} == {"Move", "Noop"}


_BLOCK = Type(name="block", feature_names=("x",))
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
    atoms = SkillGrounder.abstract_state(
        state=state, objects=(_ROBOT, _BLOCK_A, _BLOCK_B), predicates=(_ROBOT_NEAR_A, _ROBOT_NEAR_B)
    )
    assert atoms == frozenset({GroundAtom(predicate=_ROBOT_NEAR_A, objects=(_ROBOT,))})


def test_abstract_state_never_repeats_an_object_within_one_atom() -> None:
    always_true = Predicate(
        name="SameBlock", types=(_BLOCK, _BLOCK), holds=lambda state, objects: True
    )
    state = _simple_state()
    atoms = SkillGrounder.abstract_state(
        state=state, objects=(_BLOCK_A, _BLOCK_B), predicates=(always_true,)
    )
    # (block_a, block_a) and (block_b, block_b) are skipped -- only the two
    # distinct-object orderings remain.
    assert atoms == frozenset({
        GroundAtom(predicate=always_true, objects=(_BLOCK_A, _BLOCK_B)),
        GroundAtom(predicate=always_true, objects=(_BLOCK_B, _BLOCK_A)),
    })
