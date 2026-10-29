"""Binary skills for table delivery and restoring the practice scene.

SweepTable promises all five balls settled in the tray after one stroke. Its parameters
are [angle_deg, speed_m_s]: angle from outward -x, positive toward +y.
Recovery operators remain applicable after failure.
The final individual return establishes BallsRestored and, with the tool home,
TaskActive. Summary deletions are refreshed from observations, not conditional effects.
"""

from typing import ClassVar

from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable

from .objects import NUM_CUBES
from .objects import SweepTable3DObjects as O
from .predicates import (
    ANY_CUBE_OFF_TABLE,
    ANY_CUBE_ON_TABLE,
    BALLS_RESTORED,
    CUBE_ORDER,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    IN_BOX,
    NO_BALL_HELD,
    ON_FLOOR,
    ON_TABLE,
    RESET_BALLS_NEEDED,
    SWEEP_SUCCEEDED,
    TASK_ACTIVE,
    WIPER_AWAY,
    WIPER_HOME,
    WIPER_ON_FLOOR,
)

RESET_CUBES_NAME = "reset_cubes"
RESET_WIPER_NAME = "reset_wiper"
SWEEP_ANGLE_BOUNDS = (-45.0, 45.0)
SWEEP_SPEED_BOUNDS = (0.05, 0.30)
SKILL_IDS: dict[str, int] = {
    "PickWiper": 0,
    "SweepTable": 1,
    "ReturnWiper": 2,
    "PickCubeFromTable": 3,
    "PickCubeFromFloor": 4,
    "PlaceCubeInBox": 5,
    "ReturnCubeToTableFromBox": 6,
    "ReturnCubeToTableFromFloor": 7,
}
NOOP_ID = -1

_ROBOT = Variable(name="robot", type=O.robot_type)
_WIPER = Variable(name="wiper", type=O.wiper_type)
_HOME = LiftedAtom(predicate=WIPER_HOME, variables=(_WIPER,))
_BOX = Variable(name="box", type=O.box_type)
_SLOTS = tuple(Variable(name=f"cube{index}", type=O.cube_type) for index in range(NUM_CUBES))
_EMPTY = LiftedAtom(predicate=HAND_EMPTY, variables=(_ROBOT,))
_ANY_ON = LiftedAtom(predicate=ANY_CUBE_ON_TABLE, variables=())
_ANY_OFF = LiftedAtom(predicate=ANY_CUBE_OFF_TABLE, variables=())
_ACTIVE = LiftedAtom(predicate=TASK_ACTIVE, variables=())
_SUCCEEDED = LiftedAtom(predicate=SWEEP_SUCCEEDED, variables=())
_RESTORED = LiftedAtom(predicate=BALLS_RESTORED, variables=())
_RESET_BALLS = LiftedAtom(predicate=RESET_BALLS_NEEDED, variables=())
_NO_BALL = LiftedAtom(predicate=NO_BALL_HELD, variables=())
_RETURN_COMPLETION_CONDITIONS = tuple(
    frozenset(
        LiftedAtom(predicate=ON_TABLE, variables=(ball,)) for ball in _SLOTS if ball != target
    )
    for target in _SLOTS
)
_RETURN_RESTORATION_EFFECTS = frozenset({
    *(
        ConditionalAddEffect(conditions=conditions, add_effects=frozenset({_RESTORED}))
        for conditions in _RETURN_COMPLETION_CONDITIONS
    ),
    *(
        ConditionalAddEffect(conditions=conditions | {_HOME}, add_effects=frozenset({_ACTIVE}))
        for conditions in _RETURN_COMPLETION_CONDITIONS
    ),
})


class SweepTable3DSkills:
    """Lifted robot and human operators with all-or-nothing promises."""

    _robot: ClassVar[Variable] = _ROBOT
    _cube: ClassVar[Variable] = Variable(name="cube", type=O.cube_type)
    _wiper: ClassVar[Variable] = _WIPER
    _box: ClassVar[Variable] = _BOX
    _slots: ClassVar[tuple[Variable, ...]] = _SLOTS
    _empty: ClassVar[LiftedAtom] = _EMPTY
    _held_wiper: ClassVar[LiftedAtom] = LiftedAtom(
        predicate=HOLDING_WIPER, variables=(_robot, _wiper)
    )
    _held_cube: ClassVar[LiftedAtom] = LiftedAtom(predicate=HOLDING_CUBE, variables=(_robot, _cube))
    _home: ClassVar[LiftedAtom] = LiftedAtom(predicate=WIPER_HOME, variables=(_wiper,))
    _away: ClassVar[LiftedAtom] = LiftedAtom(predicate=WIPER_AWAY, variables=(_wiper,))
    _chain: ClassVar[frozenset[LiftedAtom]] = frozenset(
        LiftedAtom(predicate=CUBE_ORDER, variables=(first, second))
        for first, second in zip(_SLOTS, _SLOTS[1:], strict=False)
    )
    SWEEP_BOUNDS: ClassVar[tuple[tuple[float, float], tuple[float, float]]] = (
        SWEEP_ANGLE_BOUNDS,
        SWEEP_SPEED_BOUNDS,
    )

    PICK_WIPER: ClassVar[Skill] = Skill(
        name="PickWiper",
        parameters=(_robot, _wiper),
        preconditions=frozenset({_empty, _home, _ACTIVE}),
        add_effects=frozenset({_held_wiper, _away}),
        delete_effects=frozenset({_empty, _home}),
        param_dim=0,
        practice_cost=1.0,
    )
    SWEEP_TABLE: ClassVar[Skill] = Skill(
        name="SweepTable",
        parameters=(_robot, _wiper, _box, *_slots),
        preconditions=frozenset({
            _held_wiper,
            _ACTIVE,
            *_chain,
            *(LiftedAtom(predicate=ON_TABLE, variables=(cube,)) for cube in _SLOTS),
        }),
        add_effects=frozenset({
            _ANY_OFF,
            _SUCCEEDED,
            _RESET_BALLS,
            *(LiftedAtom(predicate=IN_BOX, variables=(cube, _BOX)) for cube in _SLOTS),
        }),
        delete_effects=frozenset({
            _ANY_ON,
            _ACTIVE,
            _RESTORED,
            *(LiftedAtom(predicate=ON_TABLE, variables=(cube,)) for cube in _SLOTS),
        }),
        param_dim=2,
        practice_cost=1.0,
    )
    RETURN_WIPER: ClassVar[Skill] = Skill(
        name="ReturnWiper",
        parameters=(_robot, _wiper),
        preconditions=frozenset({_held_wiper}),
        add_effects=frozenset({_empty, _home}),
        delete_effects=frozenset({_held_wiper, _away}),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset({_RESTORED}), add_effects=frozenset({_ACTIVE})
            )
        }),
        param_dim=0,
        practice_cost=1.0,
    )
    PICK_CUBE_FROM_TABLE: ClassVar[Skill] = Skill(
        name="PickCubeFromTable",
        parameters=(_robot, _cube),
        preconditions=frozenset({
            _empty,
            _ACTIVE,
            LiftedAtom(predicate=ON_TABLE, variables=(_cube,)),
        }),
        add_effects=frozenset({_held_cube, _ANY_OFF}),
        delete_effects=frozenset({
            _empty,
            _RESTORED,
            _NO_BALL,
            LiftedAtom(predicate=ON_TABLE, variables=(_cube,)),
        }),
        param_dim=0,
        practice_cost=1.0,
    )
    PICK_CUBE_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="PickCubeFromFloor",
        parameters=(_robot, _cube),
        preconditions=frozenset({
            _empty,
            _ACTIVE,
            LiftedAtom(predicate=ON_FLOOR, variables=(_cube,)),
        }),
        add_effects=frozenset({_held_cube}),
        delete_effects=frozenset({
            _empty,
            _NO_BALL,
            LiftedAtom(predicate=ON_FLOOR, variables=(_cube,)),
        }),
        param_dim=0,
        practice_cost=1.0,
    )
    PLACE_CUBE_IN_BOX: ClassVar[Skill] = Skill(
        name="PlaceCubeInBox",
        parameters=(_robot, _cube, _box),
        preconditions=frozenset({_held_cube, _ACTIVE}),
        add_effects=frozenset({
            _empty,
            _NO_BALL,
            LiftedAtom(predicate=IN_BOX, variables=(_cube, _box)),
        }),
        delete_effects=frozenset({_held_cube}),
        param_dim=0,
        practice_cost=1.0,
    )
    RETURN_CUBE_TO_TABLE_FROM_BOX: ClassVar[Skill] = Skill(
        name="ReturnCubeToTableFromBox",
        parameters=(_robot, _cube, _box, _wiper, *_slots),
        preconditions=frozenset({
            _empty,
            *_chain,
            LiftedAtom(predicate=IN_BOX, variables=(_cube, _box)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=ON_TABLE, variables=(_cube,)), _ANY_ON}),
        delete_effects=frozenset({LiftedAtom(predicate=IN_BOX, variables=(_cube, _box))}),
        conditional_add_effects=_RETURN_RESTORATION_EFFECTS,
        param_dim=0,
        practice_cost=1.0,
    )
    RETURN_CUBE_TO_TABLE_FROM_FLOOR: ClassVar[Skill] = Skill(
        name="ReturnCubeToTableFromFloor",
        parameters=(_robot, _cube, _wiper, *_slots),
        preconditions=frozenset({
            _empty,
            *_chain,
            LiftedAtom(predicate=ON_FLOOR, variables=(_cube,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=ON_TABLE, variables=(_cube,)), _ANY_ON}),
        delete_effects=frozenset({LiftedAtom(predicate=ON_FLOOR, variables=(_cube,))}),
        conditional_add_effects=_RETURN_RESTORATION_EFFECTS,
        param_dim=0,
        practice_cost=1.0,
    )
    RETURN_CUBE_SKILLS: ClassVar[tuple[Skill, ...]] = (
        RETURN_CUBE_TO_TABLE_FROM_BOX,
        RETURN_CUBE_TO_TABLE_FROM_FLOOR,
    )
    RESET_CUBES: ClassVar[Skill] = Skill(
        name=RESET_CUBES_NAME,
        parameters=(_robot, _box, _wiper, *_slots),
        preconditions=frozenset({_RESET_BALLS, *_chain}),
        add_effects=frozenset({
            _ANY_ON,
            _RESTORED,
            _NO_BALL,
            *(LiftedAtom(predicate=ON_TABLE, variables=(cube,)) for cube in _SLOTS),
        }),
        delete_effects=frozenset({
            _ANY_OFF,
            _SUCCEEDED,
            _RESET_BALLS,
            *(
                atom
                for cube in _SLOTS
                for atom in (
                    LiftedAtom(predicate=IN_BOX, variables=(cube, _BOX)),
                    LiftedAtom(predicate=ON_FLOOR, variables=(cube,)),
                    LiftedAtom(predicate=HOLDING_CUBE, variables=(_ROBOT, cube)),
                )
            ),
        }),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset({_home, _empty}), add_effects=frozenset({_ACTIVE})
            ),
            *(
                ConditionalAddEffect(
                    conditions=frozenset({
                        LiftedAtom(predicate=HOLDING_CUBE, variables=(_ROBOT, cube))
                    }),
                    add_effects=frozenset({_EMPTY}),
                )
                for cube in _SLOTS
            ),
            *(
                ConditionalAddEffect(
                    conditions=frozenset({
                        _HOME,
                        LiftedAtom(predicate=HOLDING_CUBE, variables=(_ROBOT, cube)),
                    }),
                    add_effects=frozenset({_ACTIVE}),
                )
                for cube in _SLOTS
            ),
        }),
        param_dim=0,
        practice_cost=1.0,
    )
    RESET_WIPER: ClassVar[Skill] = Skill(
        name=RESET_WIPER_NAME,
        parameters=(_robot, _wiper),
        preconditions=frozenset({_away, _NO_BALL}),
        add_effects=frozenset({_home}),
        delete_effects=frozenset({
            _held_wiper,
            _away,
            LiftedAtom(predicate=WIPER_ON_FLOOR, variables=(_wiper,)),
        }),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset({_held_wiper}), add_effects=frozenset({_empty})
            ),
            ConditionalAddEffect(
                conditions=frozenset({_RESTORED, _held_wiper}), add_effects=frozenset({_ACTIVE})
            ),
            ConditionalAddEffect(
                conditions=frozenset({_RESTORED, _empty}), add_effects=frozenset({_ACTIVE})
            ),
        }),
        param_dim=0,
        practice_cost=1.0,
    )

    @staticmethod
    def robot_skills() -> tuple[Skill, ...]:
        skills = SweepTable3DSkills
        return (
            skills.PICK_WIPER,
            skills.SWEEP_TABLE,
            skills.RETURN_WIPER,
            skills.PICK_CUBE_FROM_TABLE,
            skills.PICK_CUBE_FROM_FLOOR,
            skills.PLACE_CUBE_IN_BOX,
            *skills.RETURN_CUBE_SKILLS,
        )

    @staticmethod
    def human_skills() -> tuple[Skill, ...]:
        return (SweepTable3DSkills.RESET_CUBES, SweepTable3DSkills.RESET_WIPER)
