"""Tossing3D's symbolic layer: upstream physical atoms plus typed barrier sides.

The five manipulation predicates are KINDER's classifiers, not ours. Each is a lookup
into the atom set upstream's own `Tossing3DStateAbstractor`
(`kinder_models.dynamic3d.tossing.state_abstractions`) derived from the state being
asked about. Nothing here re-implements a threshold, so nothing here can drift out of
agreement with upstream. ``RobotAtSide``, ``CubeAtSide``, and ``BinAtSide`` are the
small geometric vocabulary added here so PDDL can bind a reset destination. Their sides
are robot-relative: ``robot_side`` denotes the robot's halfspace of the barrier and
``opposite_side`` denotes the other one, independent of world-frame orientation.

That is the change this module records. It previously carried six classifiers of its own,
three of them ported from upstream's `shelf` abstractions with thresholds copied across,
and three written here. Keeping them in agreement with upstream was a *testing*
obligation rather than a structural guarantee, and an audit found it had already failed
on one of the six -- the throw-pose one, which is also the one that no longer exists at
all (see below).

## What the swap changed, measured

| this domain's | upstream's | agreed before the swap? |
| --- | --- | --- |
| `HAND_EMPTY` | `HandEmpty` | yes -- 2000/2001 grid points; the miss is a |
| | | float32 artifact at the tolerance edge |
| `HOLDING` | `Holding` | upstream strictly stronger (an FK conjunct we |
| | | could not evaluate); 12/12 on real states |
| `ON_GROUND` | `OnGround` | yes -- 755/755 |
| `IN_BIN` | `MovableInGoalRegion` | yes -- 12/12, and both match KINDER's |
| | | own `_check_goals()` 12/12 |
| `REACHABLE` | `MovableIsDownX` | yes -- 201/201; both are literally |
| | | `cube.x < barrier.x` |

All five were already equivalent, so the swap is structural for every one of them.

## The sixth predicate is gone, and it is the one the swap actually moved

`ROBOT_AT_SUCCESSFUL_THROW_POSE` named the pose between `MoveToThrowPose` and `Toss`, and
it was the one predicate the swap changed behaviourally -- an audit found this domain's
band and upstream's `RobotAtThrowPose` disagreed on **478/1805** poses. Upstream then
composed the base move into the toss and **deleted the classifier**, so at the pin this
branch carries there is no `RobotAtThrowPose` to look up and no state between the two
skills for any predicate to describe. The disagreement is not resolved; the question is
retired. `THROW_RANGE` and the whole margin calibration that backed our band went with it,
and stay readable in git history and in the experiment logs that cite them.

## Where the classifiers actually run

Not here. `core.Predicate.holds` is a positional `(state, objects)` callable with no
simulator handle, and two of upstream's five classifiers need one -- `Holding` does
forward kinematics through a `PyBulletSim`, and `MovableInGoalRegion` reads the scored
region off the live env's ground fixture. So the whole abstraction is computed **once per
state, at the boundary**, by `KinderBackend.abstract_atoms`, and travels on the state.
See `types.py` for the design and for the measurement showing a stale state still yields
its own answers.

One consequence worth stating: a hand-built `core.State` can no longer answer a predicate
here. It raises rather than quietly returning `False` for everything. Upstream's three
*pure* classifiers are `@staticmethod`s over an `ObjectCentricState`, which is
constructible with no MuJoCo, so the offline boundary probes moved to calling those
directly -- see `tests/environments/tossing3d/object_centric.py`.

## What this module no longer owns

Nothing. It used to carry this domain's own sampler and controller constants -- a
`THROW_STANDOFF_BOUNDS` for `MoveToThrowPose` to draw from, `TOSS_SPEED_BOUNDS`,
`TOSS_RELEASE_MS_BOUNDS`, and upstream's two shipped defaults. Every one of them belonged
to the three-skill decomposition, and the composed controller declares its own bounds on
itself, so they moved to `skills.py` as reads of upstream's constants rather than numbers
of ours. See `skills.py` for the ranges and `test_kinder_pin.py` for the assertions that
hold them to upstream's.
"""

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import Tossing3DEnvironment
from .sides import Tossing3DSides


class Tossing3DAtoms:
    """Looks one of upstream's abstract atoms up on a `Tossing3DState`.

    Every predicate below is this and nothing else. The classifiers themselves are
    upstream's -- `kinder_models.dynamic3d.tossing.state_abstractions` -- and are
    evaluated once per state, at the boundary, by `KinderBackend.abstract_atoms`. See
    `types.py` for why the abstraction travels on the state rather than being computed
    here, and for the measurement showing that is honest rather than a stale cache.

    A static-method container, never instantiated, same as every other business-logic
    class in this project.
    """

    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        """Whether upstream's abstractor said `name(objects)` held, for this state."""
        atoms = getattr(state, "abstract_atoms", None)
        if atoms is None:
            raise ValueError(
                f"cannot evaluate {name} on a state with no abstraction attached. "
                "Tossing3D's predicates are upstream's classifiers, which need a live "
                "scene to evaluate (forward kinematics for Holding, the ground fixture "
                "for MovableInGoalRegion), so they are computed once by "
                "KinderBackend.abstract_atoms when the state is built. A hand-built "
                "core.State cannot answer them; build one through "
                "Tossing3DEnvironment.take_action/reset_to_seed, or test upstream's "
                "classifiers directly (see tests/environments/tossing3d/object_centric.py)."
            )
        return (name, tuple(obj.name for obj in objects)) in atoms


# Upstream's own predicate names, which are what `KinderBackend.abstract_atoms` keys the
# atom set by. Named here so the mapping from this domain's vocabulary to upstream's is
# in one readable place rather than spread across five lambdas.
KB_IN_GOAL_REGION = "MovableInGoalRegion"
KB_HAND_EMPTY = "HandEmpty"
KB_HOLDING = "Holding"
KB_ON_GROUND = "OnGround"
KB_IS_DOWN_X = "MovableIsDownX"


# `Predicate.holds` is a positional `(state, objects)` callable per its interface contract
# (`Goal.is_satisfied` calls it that way), so each lambda below adapts that into a call to
# the keyword-only lookup above -- exactly as Light Switch and Tossing Room do.
#
# **This domain's predicate names are kept, and upstream's classifiers are what now backs
# them.** The names are this domain's symbolic vocabulary: they appear in the PDDL
# `skills.py` emits, in the operator model, and in `Tossing3DTasks`' goal. Renaming them
# to upstream's would be an operator-model change, which this is deliberately not.
IN_BIN = Predicate(
    name="InBin",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    # **Upstream's is unary and this is binary, so the bin argument is dropped.**
    # `MovableInGoalRegion(cube)` reads the scored region off the live scene's ground
    # fixture, so it takes no target object at all. The binary shape is kept because it
    # is what `Tossing3DTasks`' goal and this domain's operators are written against, and
    # because "the cube is in *the bin*" is the sentence this domain means. The dropped
    # argument is real, though: under this predicate a second bin would be
    # indistinguishable from the first.
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_IN_GOAL_REGION, objects=(objects[0],)
    ),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(Tossing3DEnvironment.robot_type,),
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_HAND_EMPTY, objects=(objects[0],)
    ),
)

HOLDING = Predicate(
    name="Holding",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    # Strictly stronger than what this domain carried before: upstream adds a
    # forward-kinematics conjunct (the end effector within
    # `END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE` of the object) that a pure function of a
    # flat `core.State` could not evaluate, and which our own version therefore dropped.
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_HOLDING, objects=(objects[0], objects[1])
    ),
)

CLOSED_EMPTY = Predicate(
    name="ClosedEmpty",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    holds=lambda state, objects: (
        not HAND_EMPTY.holds(state, (objects[0],)) and not HOLDING.holds(state, objects)
    ),
)
NOT_HOLDING = Predicate(
    name="NotHolding",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    holds=lambda state, objects: not HOLDING.holds(state, objects),
)

ON_GROUND = Predicate(
    name="OnGround",
    types=(Tossing3DEnvironment.cube_type,),
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_ON_GROUND, objects=(objects[0],)
    ),
)

REACHABLE = Predicate(
    name="Reachable",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
    # Upstream's `MovableIsDownX(cube, barrier)` is literally `cube.x < barrier.x`, which
    # is what this domain's `Reachable` always was -- the one-way door that makes the
    # domain interesting. Audited at 201/201 agreement before the swap.
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_IS_DOWN_X, objects=(objects[0], objects[1])
    ),
)


def _same_barrier_side(*, state: State, x_object: Object, side: Object) -> bool:
    """Whether ``x_object`` is on the named robot-relative halfspace."""
    robot_x = state.get(obj=Tossing3DEnvironment.robot, feature_name="pos_base_x")
    barrier_x = state.get(obj=Tossing3DEnvironment.barrier, feature_name="x")
    object_x = state.get(obj=x_object, feature_name="x")
    robot_delta = robot_x - barrier_x
    object_delta = object_x - barrier_x
    # An object centred on or straddling the barrier is not in either open
    # halfspace. Include both physical footprints: using only the cube half-width
    # incorrectly classifies overlap with the nonzero-width barrier as reachable.
    barrier_half_width = state.get(obj=Tossing3DEnvironment.barrier, feature_name="bb_x") / 2.0
    object_half_width = (
        state.get(obj=x_object, feature_name="bb_x") / 2.0
        if "bb_x" in x_object.type.feature_names
        else 0.0
    )
    clearance = barrier_half_width + object_half_width
    if abs(object_delta) <= clearance:
        return False
    same_as_robot = object_delta * robot_delta > 0.0
    if side == Tossing3DSides.robot:
        return same_as_robot
    if side == Tossing3DSides.opposite:
        return object_delta * robot_delta < 0.0
    raise ValueError(f"unknown Tossing3D side object {side.name!r}")


ROBOT_AT_SIDE = Predicate(
    name="RobotAtSide",
    types=(
        Tossing3DEnvironment.robot_type,
        Tossing3DEnvironment.barrier_type,
        Tossing3DSides.type,
    ),
    holds=lambda state, objects: objects[2] == Tossing3DSides.robot,
)

CUBE_AT_SIDE = Predicate(
    name="CubeAtSide",
    types=(
        Tossing3DEnvironment.cube_type,
        Tossing3DEnvironment.barrier_type,
        Tossing3DSides.type,
    ),
    holds=lambda state, objects: _same_barrier_side(
        state=state, x_object=objects[0], side=objects[2]
    ),
)

BIN_AT_SIDE = Predicate(
    name="BinAtSide",
    types=(
        Tossing3DEnvironment.bin_type,
        Tossing3DEnvironment.barrier_type,
        Tossing3DSides.type,
    ),
    holds=lambda state, objects: _same_barrier_side(
        state=state, x_object=objects[0], side=objects[2]
    ),
)


# The bin's physical outer footprint half-extent. Not a state feature: the bin's
# `x_min..z_max` features carry the SCORING box, which is strictly inside the walls
# and whose width varies with the scene (0.15 m in the stock task, 0.22 m after a
# bin relocation), so it cannot stand in for the walls. The 0.30 m footprint is the
# one `test_the_live_scoring_window_lies_inside_the_bins_live_footprint` measures
# against the compiled model, so this constant cannot drift from the scene unnoticed.
BIN_FOOTPRINT_HALF_M = 0.15

# The lateral clearance a graspable cube needs from the bin's footprint planes.
# Measured at the live pick controller by bisection (2026-09-22 trap diagnosis): a
# cube whose face sits 0.040 m inside the footprint plane is REFUSED ("No
# collision-free cube grasp", zero steps) and one at 0.050 m is planned and picked.
# The boundary decomposes into upstream's named `GRASP_OBSTACLE_CLEARANCE = 0.01`
# (kinder_models `parameterized_skills.PickCubeController`, the mesh distance the
# grasp planner reserves from non-target obstacles, added by upstream c18056d2) plus
# the ~0.04 m wall band between the scoring box and the footprint that the 2F-85's
# fingers must clear -- the composite is what the bisection measures directly. The
# original home for this classifier would be upstream `state_abstractions`, like the
# five classifier-backed predicates; keeping it local follows #346's side-atom
# precedent, and migrating it upstream is deliberate follow-up work.
GRASP_CLEARANCE_M = 0.05


def _grasp_clear(*, state: State, cube: Object, bin_: Object) -> bool:
    """Whether every cube face clears the bin's wall band by `GRASP_CLEARANCE_M`.

    Two-dimensional and lateral only: the wall band is the square annulus between
    the footprint rectangle and itself (walls have no reachable interior in this
    projection), so the signed quantity that matters is the distance from the cube's
    rectangle to the footprint *boundary*. Inside the bin that is the smallest
    face-to-plane gap; outside it is the rectangle-to-rectangle distance; a cube
    straddling a wall is never clear.
    """
    cube_half = state.get(obj=cube, feature_name="bb_x") / 2.0
    inside_gaps: list[float] = []
    outside_gaps: list[float] = []
    straddles = 0
    for axis in ("x", "y"):
        delta = abs(state.get(obj=cube, feature_name=axis) - state.get(obj=bin_, feature_name=axis))
        near_face = delta - cube_half
        far_face = delta + cube_half
        if far_face <= BIN_FOOTPRINT_HALF_M:
            inside_gaps.append(BIN_FOOTPRINT_HALF_M - far_face)
        elif near_face >= BIN_FOOTPRINT_HALF_M:
            outside_gaps.append(near_face - BIN_FOOTPRINT_HALF_M)
        else:
            straddles += 1
    if straddles and not outside_gaps:
        # A wall passes through the cube's rectangle: touching, never clear.
        return False
    # Exactly the margin counts as clear (the measured 0.050 m case was accepted
    # live); the epsilon keeps float arithmetic from flipping that boundary.
    epsilon = 1e-9
    if outside_gaps:
        # Outside the footprint: rectangle-to-rectangle distance, to which only the
        # axes actually beyond the footprint contribute.
        distance = sum(gap**2 for gap in outside_gaps) ** 0.5
        return distance >= GRASP_CLEARANCE_M - epsilon
    return min(inside_gaps) >= GRASP_CLEARANCE_M - epsilon


GRASP_CLEAR = Predicate(
    name="GraspClear",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: _grasp_clear(state=state, cube=objects[0], bin_=objects[1]),
)
