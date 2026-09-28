"""Whether the pick controller's own planner finds a plan: the rule behind `PickPlannable`.

The verdict itself is not computed here. It is a dry run of upstream's
`PickCubeController.reset` -- grasp yaw, base path, hover, descent and lift -- made by
`KinderBackend.pick_cube_plan_failure`, which builds the controller exactly as a
dispatched pick does and stops before the first simulator step. This module holds the
atom's name and the one decision that is ours: in which states the dry run is made at
all.

## Why a dry run and not another clearance constant

Since PickCube stopped requiring `PickupUnblocked`, any state that satisfies every pick
precondition while the controller refuses at dispatch ("No collision-free cube grasp",
zero controller steps) is one the planners retry for the rest of the run. Four such
geometries were found one at a time -- a tipped bin, a cube beside the barrier, a cube
inside the bin near a wall, a cube between the bin and the room's south wall -- and each
clearance constant fitted to one of them was silent about the next. The controller's
planner is the thing being predicted, so it is asked directly. A robot may always query
its own motion planner before acting; nothing here reads simulator state the robot
could not observe.

## Why the verdict agrees with dispatch

Every planning call inside `PickCubeController.reset` is seeded with the constant 0 and
draws from a generator local to that call, and the collision model is a PyBullet client
built fresh from the state handed in. The dry run and the dispatch therefore run the
same code on the same input. Dispatch re-plans the descent and lift only once it is
executing, from the cube pose observed then; that is a mid-execution failure, which
this predicate does not describe and the planners treat as an ordinary failed attempt.

Measured on 2026-09-28 by replaying every PickCube dispatch logged by 30 EXP-22b runs
(kindergarden f0d554b, kinder-baselines 427ad6c) from its pre-pick state: of 12994
picks the controller refused at dispatch the dry run found no plan for 12994/12994,
and of 21122 it executed the dry run found a plan for 21122/21122. The refusals are
16 distinct states, the executed picks 4313; for 2581 of the executed picks the
pre-pick state was never logged and the pick's own first tick stood in for it.

## The three cases

| state | dry run | atom |
| --- | --- | --- |
| the robot holds the cube | not made | present |
| the cube is across the barrier from the robot | not made | absent |
| anything else | made, cached per state | present iff a plan was found |

While the cube is held there is no pick to plan, and the atom is carried as present:
the imagined toss leaves `PickPlannable` as it found it (it is one of the toss's
`ignore_effects`, not re-asserted, like `GraspClear`), so carrying it absent would make
every imagined landing unpickable and every imagined toss end in a reset. Across the
barrier the base has no path, so the verdict is known without the planner's search --
which is also the slowest one, about 10 s against a median 0.7 s, since a
bidirectional RRT exhausts its budget before reporting that no path exists. No robot
skill can change either condition except through the states the dry run is made in.
"""

from collections.abc import Mapping

# The atom name `KinderBackend.abstract_atoms` emits and `PICK_PLANNABLE` looks up.
KB_PICK_PLANNABLE = "PickPlannable"


class PickPlannableGate:
    """A static-method container, never instantiated."""

    @staticmethod
    def cube_across_barrier(
        *, robot_x: float, cube: Mapping[str, float], barrier: Mapping[str, float]
    ) -> bool:
        """Whether the barrier's centre plane separates the cube from the robot's base.

        Deliberately the bare plane, without `CubeAtSide`'s footprint or grasp-band
        margins: a cube touching the barrier from the robot's side is a state the dry
        run is made in, so its verdict is the planner's and not this gate's.
        """
        return (cube["x"] - barrier["x"]) * (robot_x - barrier["x"]) <= 0.0
