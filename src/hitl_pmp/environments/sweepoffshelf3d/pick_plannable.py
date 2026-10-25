"""Whether the floor and box pick's own planning finds a grasp: the rule behind
`PickPlannable`, Tossing3D's atom of the same name carried over to this scene.

The verdict itself is not computed here. It is a dry run of the pick that starts
`PickCubeFromFloor` and both `ReturnCubeToShelf*` skills (`FloorPicks.plannable`): the
same grasp x base candidates, checked in MuJoCo, and the same `PickShelfController`
planning, stopped before the first control step. This module holds the atom's name and
the one decision that is ours: in which states the dry run is made at all.

## Why

A sweep that misses a narrow box can wedge a cube where no grasp exists -- in the
pocket between the box's outer wall and the cupboard's bottom board, on that board, on
the box's rim (2026-10-24 floor-pick-strip diagnosis: 11/25 of the PoC's refused poses
are still refused with the grid-36 grasps). Every pick precondition holds there, so
without this atom the planners retry a pick that refuses at dispatch with zero control
steps for the rest of the run. With it, those operators are inapplicable and the way
forward is the human's `reset_cubes`.

## The three cases

| state | dry run | atom |
| --- | --- | --- |
| the hand is not empty | not made | present for every cube |
| the cube is on the cube shelf | not made | present |
| the cube is in the box or on the floor, the hand empty | made | present iff a grasp plans |

The atom is defined for off-shelf cubes only; it gates `PickCubeFromFloor` and both
returns, never `PickCubeFromShelf`. Where it is not made it is carried present, for the
reason Tossing3D carries it present while the cube is held: no operator re-asserts it,
so a cube swept or placed into the box in a forecast keeps the atom it had, and
carrying it absent would make every imagined return from the box inapplicable. While
the hand holds the wiper or a cube no pick can be dispatched, and the dry run would
plan around a hand the pick never has; the next observed state with the hand empty
decides. `reset_cubes` is unchanged: it puts cubes back on the shelf, where the dry
run is not made.
"""

from collections.abc import Callable

from .objects import SweepOffShelf3DObjects as O
from .types import AbstractAtom

# The atom name the environment emits and `PICK_PLANNABLE` looks up.
KB_PICK_PLANNABLE = "PickPlannable"


class PickPlannableAtoms:
    """A static-method container, never instantiated."""

    @staticmethod
    def with_verdicts(
        *, atoms: frozenset[AbstractAtom], plannable: Callable[[str], bool]
    ) -> frozenset[AbstractAtom]:
        """`atoms` plus `PickPlannable(robot, cube)` for every cube the table above
        makes it present for; `plannable(cube)` is the dry run, called only for an
        off-shelf cube with the hand empty."""
        robot = O.robot.name
        hand_empty = ("HandEmpty", (robot,)) in atoms
        present = []
        for cube in O.cubes:
            name = cube.name
            off_shelf = ("OnFloor", (name,)) in atoms or ("InBox", (name, O.box.name)) in atoms
            if not hand_empty or not off_shelf or plannable(name):
                present.append((KB_PICK_PLANNABLE, (robot, name)))
        return atoms | frozenset(present)
