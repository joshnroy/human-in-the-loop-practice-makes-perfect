"""SweepOffShelf3D's lifted skills: six for the robot, two for the human.

Only `SweepShelf` has continuous parameters: the wiper head's height above the board and
the pull speed. The height range runs past a cube's full height on purpose -- a head
riding above the cubes passes over them -- so the sampled range holds parameters that
fail as well as ones that work, and there is something for practice to learn.

`SweepShelf` takes all five cubes so one sweep can move however many are on the shelf:
each cube that is `OnShelf` is promised `InBox` through a conditional effect, and the
`CubeOrder` chain pins those five slots to a single grounding. The human resets are
modelled at their nominal effect (every cube back on the shelf; the wiper home). How
often that effect actually happens -- `reset_cubes` returns only 5 - k cubes -- is not
written into the model: the robot learns it, as it learns every robot skill's competence.
"""

from typing import ClassVar

from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable

from .objects import NUM_CUBES
from .objects import SweepOffShelf3DObjects as O
from .predicates import (
    CUBE_ORDER,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    IN_BOX,
    ON_FLOOR,
    ON_SHELF,
    WIPER_HOME,
    WIPER_ON_FLOOR,
)

RESET_CUBES_NAME = "reset_cubes"
RESET_WIPER_NAME = "reset_wiper"

# Module level because a class-body comprehension cannot see class attributes.
_ROBOT = Variable(name="robot", type=O.robot_type)
_BOX = Variable(name="box", type=O.box_type)
_SLOTS = tuple(Variable(name=f"cube{i}", type=O.cube_type) for i in range(NUM_CUBES))
_EMPTY = LiftedAtom(predicate=HAND_EMPTY, variables=(_ROBOT,))


class SweepOffShelf3DSkills:
    """A static-method container, never instantiated."""

    _robot: ClassVar[Variable] = _ROBOT
    _cube: ClassVar[Variable] = Variable(name="cube", type=O.cube_type)
    _wiper: ClassVar[Variable] = Variable(name="wiper", type=O.wiper_type)
    _box: ClassVar[Variable] = _BOX
    _slots: ClassVar[tuple[Variable, ...]] = _SLOTS

    _empty: ClassVar[LiftedAtom] = _EMPTY
    _held_wiper: ClassVar[LiftedAtom] = LiftedAtom(
        predicate=HOLDING_WIPER, variables=(_robot, _wiper)
    )
    _held_cube: ClassVar[LiftedAtom] = LiftedAtom(predicate=HOLDING_CUBE, variables=(_robot, _cube))
    _home: ClassVar[LiftedAtom] = LiftedAtom(predicate=WIPER_HOME, variables=(_wiper,))
    _chain: ClassVar[frozenset[LiftedAtom]] = frozenset(
        LiftedAtom(predicate=CUBE_ORDER, variables=(a, b))
        for a, b in zip(_SLOTS, _SLOTS[1:], strict=False)
    )

    # (head height above the board [m], pull speed [m/s]).
    SWEEP_BOUNDS: ClassVar[tuple[tuple[float, float], tuple[float, float]]] = (
        (0.0, 0.04),
        (0.05, 0.25),
    )

    PICK_WIPER: ClassVar[Skill] = Skill(
        name="PickWiper",
        parameters=(_robot, _wiper),
        preconditions=frozenset({_empty, _home}),
        add_effects=frozenset({_held_wiper}),
        delete_effects=frozenset({_empty, _home}),
        param_dim=0,
        practice_cost=1.0,
    )

    SWEEP_SHELF: ClassVar[Skill] = Skill(
        name="SweepShelf",
        parameters=(_robot, _wiper, _box, *_slots),
        preconditions=frozenset({_held_wiper, *_chain}),
        add_effects=frozenset(),
        delete_effects=frozenset(LiftedAtom(predicate=ON_SHELF, variables=(c,)) for c in _slots),
        conditional_add_effects=frozenset(
            ConditionalAddEffect(
                conditions=frozenset({LiftedAtom(predicate=ON_SHELF, variables=(c,))}),
                add_effects=frozenset({LiftedAtom(predicate=IN_BOX, variables=(c, _BOX))}),
            )
            for c in _slots
        ),
        param_dim=2,
        practice_cost=1.0,
    )

    RETURN_WIPER: ClassVar[Skill] = Skill(
        name="ReturnWiper",
        parameters=(_robot, _wiper),
        preconditions=frozenset({_held_wiper}),
        add_effects=frozenset({_empty, _home}),
        delete_effects=frozenset({_held_wiper}),
        param_dim=0,
        practice_cost=1.0,
    )

    PICK_CUBE_FROM_SHELF: ClassVar[Skill] = Skill(
        name="PickCubeFromShelf",
        parameters=(_robot, _cube),
        preconditions=frozenset({_empty, LiftedAtom(predicate=ON_SHELF, variables=(_cube,))}),
        add_effects=frozenset({_held_cube}),
        delete_effects=frozenset({_empty, LiftedAtom(predicate=ON_SHELF, variables=(_cube,))}),
        param_dim=0,
        practice_cost=1.0,
    )

    PICK_CUBE_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="PickCubeFromFloor",
        parameters=(_robot, _cube),
        preconditions=frozenset({_empty, LiftedAtom(predicate=ON_FLOOR, variables=(_cube,))}),
        add_effects=frozenset({_held_cube}),
        delete_effects=frozenset({_empty, LiftedAtom(predicate=ON_FLOOR, variables=(_cube,))}),
        param_dim=0,
        practice_cost=1.0,
    )

    PLACE_CUBE_IN_BOX: ClassVar[Skill] = Skill(
        name="PlaceCubeInBox",
        parameters=(_robot, _cube, _box),
        preconditions=frozenset({_held_cube}),
        add_effects=frozenset({_empty, LiftedAtom(predicate=IN_BOX, variables=(_cube, _box))}),
        delete_effects=frozenset({_held_cube}),
        param_dim=0,
        practice_cost=1.0,
    )

    # Human skills. Practice cost is set by the skill provider.
    RESET_CUBES: ClassVar[Skill] = Skill(
        name=RESET_CUBES_NAME,
        parameters=(_robot, _box, *_slots),
        preconditions=frozenset(_chain),
        add_effects=frozenset(LiftedAtom(predicate=ON_SHELF, variables=(c,)) for c in _slots),
        delete_effects=frozenset(
            atom
            for c in _slots
            for atom in (
                LiftedAtom(predicate=IN_BOX, variables=(c, _BOX)),
                LiftedAtom(predicate=ON_FLOOR, variables=(c,)),
                LiftedAtom(predicate=HOLDING_CUBE, variables=(_ROBOT, c)),
            )
        ),
        conditional_add_effects=frozenset(
            ConditionalAddEffect(
                conditions=frozenset({LiftedAtom(predicate=HOLDING_CUBE, variables=(_ROBOT, c))}),
                add_effects=frozenset({_EMPTY}),
            )
            for c in _slots
        ),
        param_dim=0,
        practice_cost=1.0,
    )

    RESET_WIPER: ClassVar[Skill] = Skill(
        name=RESET_WIPER_NAME,
        parameters=(_robot, _wiper),
        preconditions=frozenset(),
        add_effects=frozenset({_home}),
        delete_effects=frozenset({
            _held_wiper,
            LiftedAtom(predicate=WIPER_ON_FLOOR, variables=(_wiper,)),
        }),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset({_held_wiper}), add_effects=frozenset({_empty})
            )
        }),
        param_dim=0,
        practice_cost=1.0,
    )

    @staticmethod
    def robot_skills() -> tuple[Skill, ...]:
        s = SweepOffShelf3DSkills
        return (
            s.PICK_WIPER,
            s.SWEEP_SHELF,
            s.RETURN_WIPER,
            s.PICK_CUBE_FROM_SHELF,
            s.PICK_CUBE_FROM_FLOOR,
            s.PLACE_CUBE_IN_BOX,
        )

    @staticmethod
    def human_skills() -> tuple[Skill, ...]:
        return (SweepOffShelf3DSkills.RESET_CUBES, SweepOffShelf3DSkills.RESET_WIPER)
