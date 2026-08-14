"""Tossing3D's symbolic layer: six predicates, all of them upstream's.

**These are kinder-baselines' classifiers, not ours.** Every predicate below is a lookup
into the atom set upstream's own `Tossing3DStateAbstractor`
(`kinder_models.dynamic3d.tossing.state_abstractions`) derived from the state being
asked about. Nothing here re-implements a threshold, so nothing here can drift out of
agreement with upstream.

That is the change this module records. It previously carried six classifiers of its own,
three of them ported from upstream's `shelf` abstractions with thresholds copied across,
and three written here. Keeping them in agreement with upstream was a *testing*
obligation rather than a structural guarantee, and an audit found it had already failed
on one of the six.

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
| `ROBOT_AT_SUCCESSFUL_THROW_POSE` | `RobotAtThrowPose` | **no -- 478/1805 disagreed** |

So five of six were already equivalent and the swap is structural for them. The sixth
genuinely moves behaviour: our band accepted achieved standoffs [0.210, 1.400] against
upstream's measured [1.09, 1.375], our lateral tolerance was 0.04 against upstream's
0.08, and upstream has a heading conjunct we had none of.

## Where the classifiers actually run

Not here. `core.Predicate.holds` is a positional `(state, objects)` callable with no
simulator handle, and two of upstream's six classifiers need one -- `Holding` does
forward kinematics through a `PyBulletSim`, and `MovableInGoalRegion` reads the scored
region off the live env's ground fixture. So the whole abstraction is computed **once per
state, at the boundary**, by `KinderBackend.abstract_atoms`, and travels on the state.
See `types.py` for the design and for the measurement showing a stale state still yields
its own answers.

One consequence worth stating: a hand-built `core.State` can no longer answer a predicate
here. It raises rather than quietly returning `False` for everything. Upstream's four
*pure* classifiers are `@staticmethod`s over an `ObjectCentricState`, which is
constructible with no MuJoCo, so the offline boundary probes moved to calling those
directly -- see `tests/environments/tossing3d/object_centric.py`.

## What this module still owns

The constants below, which are **sampler** and **controller** parameters rather than
classifier thresholds. They were never upstream's and are not part of the swap.

> **`THROW_STANDOFF_BOUNDS` here is not upstream's constant of the same name.** Ours is
> the interval `MoveToThrowPose`'s sampler *draws from*, (1.10, 1.75). Upstream's, in
> `state_abstractions.py`, is the band `RobotAtThrowPose` *accepts*, (1.09, 1.375).
> Upstream keeps the same separation under different names -- its sampler draws from
> `TOSS_TARGET_DISTANCE_BOUNDS` = (1.25, 1.45) -- and for the same reason: a predicate
> that accepts exactly what the sampler can draw is a predicate whose add effect can
> never fail, which is what made this domain's sampler unlearnable once already. **Do not
> couple them, and do not "reconcile" the two same-named constants.**
"""

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import Tossing3DEnvironment

# Ours: the range of throw standoffs, in metres -- the interval the `MoveToThrowPose`
# sampler draws from. There is no upstream number to borrow: upstream simply hardcodes
# 1.35 in its own test, and its own `MOVE_TO_TARGET_DISTANCE_BOUNDS` of (0.5, 0.6) is a
# *grasping* standoff.
#
# It is the **feasible** range -- where the episode is still a throw problem -- not the
# range that solves. The feasible range is `[0.40, 2.06]`, measured over five scene seeds
# by running `Pick` -> `MoveToThrowPose(standoff)` -> `Toss` and recording each upstream
# controller's own outcome:
#
# - **Below 0.40 m** `move_to_target` reports success but the base drives into the bin and
#   shoves it across the floor: 5/5 seeds displaced the bin at 0.35 m (by up to 0.069 m,
#   and by 0.99 m at 0.05 m), against 0/5 at 0.40 m.
# - **Above ~2.06 m** the base reaches the pose cleanly, but `Toss`'s windup
#   `move_arm_to_conf` fails to motion-plan (`AssertionError: Motion planning failed`), so
#   `Toss` executes zero steps and the cube never leaves the gripper: 2/5 seeds failed to
#   plan at 2.08 m, against 0/5 at 2.06 m.
#
# The bounds inset that range at both ends. **The bottom is not belt-and-braces -- it is
# load-bearing, and used to be wrong.** An earlier version of this comment claimed
# "nothing observed stops that short", which is false: `move_to_target`'s base motion
# planner has collision-checking hardcoded off upstream, and `cuboid_barrier` -- a real
# dynamic MuJoCo body the base must walk past to reach a short standoff -- is not a
# static waypoint check, so a short-enough `MoveToThrowPose` drives straight through it
# and knocks it over. `test_move_to_throw_pose_at_the_lower_standoff_bound_does_not_
# disturb_the_barrier` would have caught this: at the old 0.45 m lower bound the barrier
# visibly moves.
#
# Measured three independent ways, all using the real oracle Pick parameters
# (`ORACLE_PICK_DISTANCE=0.5682351863248143`, `ORACLE_PICK_ROTATION=-0.7008563047585579`
# -- a placeholder-param probe earlier gave a wrong, lower number and was caught and
# corrected before landing here): the worst colliding standoff, `WORST_BARRIER_
# COLLISION_STANDOFF` below, is **1.00 m**, never exceeded anywhere tested --
#
# - 10 scene seeds x 0.005 m resolution over [0.98, 1.03] (fine boundary pinpoint),
# - 10 scene seeds x 0.02 m resolution over [0.90, 1.40] (dense sweep),
# - all four corners of `Pick`'s own full sampling box (`skills.PICK_DISTANCE_BOUNDS`,
#   `skills.PICK_ROTATION_BOUNDS`) plus the oracle point -- `Pick` also samples during
#   practice, not just at its oracle default, so the base pose `MoveToThrowPose` starts
#   from is not fixed to the oracle's own draw.
#
# The first fully-clear standoff at 0.005 m resolution is 1.005 m. `BARRIER_COLLISION_
# MARGIN` (0.10 m) sits above the worst measured collision rather than that first clear
# point, so the new bound is `1.00 + 0.10 = 1.10` -- a confirming sweep of exactly that
# proposed range, [1.10, 1.75] at 0.05 m resolution across 10 scene seeds, scored
# 140/140 clear with the barrier's pose bit-exact every time. Only this lower bound
# moves; the top, below, is unrelated to this defect and stays as measured. It is
# computed from the two constants below rather than written as the literal `1.10`, so a
# future re-measurement of either the collision point or the margin cannot silently drift
# out of sync with the bound it justifies -- the same discipline
# `RobotAtSuccessfulThrowPoseClassifier`'s band is held to below.
#
# The top is set by a *hazard* rather than by feasibility: the predicate must stay false
# at the pose `Pick` leaves the base in, or the oracle -- and any planner reading it --
# would believe it was already at a throw pose and skip `MoveToThrowPose`. Over 30 scene
# seeds the post-`Pick` base sits 1.364-1.971 m from the bin, and seed 14 sits at
# 1.8592 m only 0.0074 m off-axis, i.e. inside the lateral tolerance, so nothing but the
# standoff would exclude it. An upper bound of 1.90 admits it; 1.75 does not.
#
# **This is the sampler's range only. It is deliberately NOT what the predicate accepts.**
# Those two used to be the same symbol -- this constant was imported into
# `NearBinClassifier` and used as its acceptance interval -- and that identity is what made
# this domain's sampler unlearnable. `MoveToThrowPose`'s only add effect was `NearBin`, so
# every standoff the sampler could draw satisfied its own add effect by construction: the
# label was constant-true (16/16 attempts labelled success in a probe run), EES's per-skill
# success classifier saw a single class, and every draw fell back to uniform forever. No
# amount of data can fix a label that cannot vary.
#
# `RobotAtSuccessfulThrowPose` now derives its own, much narrower acceptance band from live
# scene geometry and `THROW_RANGE`. **Do not re-couple them**; widening this constant must
# not widen the predicate, or the tautology returns silently.
#
# The widening from `(1.20, 1.65)` to the feasible range is what makes the separation
# *measurable* rather than merely correct. At the old bounds the derived band covered
# 0.225 of a 0.45 m range, so a uniform draw was right about half the time and a learned
# sampler had little headroom over its own prior. After the barrier-collision tightening
# the range is 0.65 m wide and the (untrimmed, geometric) 0.300 m band is 0.300/0.65 =
# 6/13 of it: still a substantial fraction wrong more often than right, though a smaller
# margin over the prior than the 3/13 the wider range had -- see this module's
# `RobotAtSuccessfulThrowPoseClassifier` docstring for the actually-accepted (margin-
# trimmed) 0.225 m band and its own share of the range.
WORST_BARRIER_COLLISION_STANDOFF = 1.00
BARRIER_COLLISION_MARGIN = 0.10
THROW_STANDOFF_BOUNDS = (WORST_BARRIER_COLLISION_STANDOFF + BARRIER_COLLISION_MARGIN, 1.75)

# `Toss`'s release speed, in joint-path deg/s -- the rate along the path between upstream's
# two arm configurations (L2 norm 148.8288 deg). `KinderBackend.run_toss` is the one site
# that converts to upstream's rad/s.
#
# 140 is what the real TidyBot primitive commands
# (`movej_primitive.execute(..., max_vel=140, ...)`); 60 is the bottom of PR #213's grid.
# No feasibility clamp below 140: `_ARM_MAX_VEL[5] = 70` deg/s is kinder-baselines' own
# conservative constant, not a hardware limit (PR #221).
TOSS_SPEED_BOUNDS = (60.0, 140.0)

# `Toss`'s second dial: the millisecond from the start of the swing at which the gripper
# opens. Absolute rather than a swing fraction because that is what the real TidyBot's
# `movej_primitive.execute()` takes. Its own 600 ms does not transfer -- it normalises on
# the L-infinity norm and finishes in 1476 ms, so 600 is fraction 0.4107 of its swing
# against 0.3449 of this one.
#
# Swing duration is a function of `release_speed`, so the two dials are coupled. Measured
# from `toss_profile_limits` + `_trapezoidal_motion_profile`:
#
# | `release_speed` deg/s | 60 | 80 | 100 | 120 | 140 |
# | --- | --- | --- | --- | --- | --- |
# | swing duration ms | 3100 | 2500 | 2100 | 1900 | 1700 |
# | path fraction at 720 ms | 0.196 | 0.262 | 0.327 | 0.392 | 0.458 |
# | ms at path fraction 0.46 | 1375 | 1090 | 918 | 804 | 723 |
#
# 1400 is set by the shortest swing (1700 ms, at 140 deg/s). `gripper_release_ms` is not
# clamped upstream: at or past the end of the swing the gripper never opens. Raising
# `TOSS_SPEED_BOUNDS` invalidates this interval -- at a 240 deg/s cap the shortest swing is
# 1300 ms, below the slow end's 1375 ms crossing.
TOSS_RELEASE_MS_BOUNDS = (300.0, 1400.0)

# Upstream's shipped default: the millisecond path fraction 0.46 falls at, for the shipped
# windup->release path at 140 deg/s.
#
# **720, not the 723 the nominal `TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF` difference
# gives**: `TossController.reset` profiles both endpoints out of `run_motion_planning`, and
# the bent path moves the crossing 3 ms earlier -- 52 mm of landing distance. Re-derive by
# running the swing, never from the two arm configurations.
UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS = 720.0

# Upstream's own default, and the speed every committed Tossing3D number was measured at:
# what `toss_profile_limits()` returns when passed nothing.
UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S = 140.0


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
# in one readable place rather than spread across six lambdas.
KB_IN_GOAL_REGION = "MovableInGoalRegion"
KB_HAND_EMPTY = "HandEmpty"
KB_HOLDING = "Holding"
KB_ON_GROUND = "OnGround"
KB_IS_DOWN_X = "MovableIsDownX"
KB_AT_THROW_POSE = "RobotAtThrowPose"


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

ROBOT_AT_SUCCESSFUL_THROW_POSE = Predicate(
    name="RobotAtSuccessfulThrowPose",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.bin_type),
    # **The one predicate whose behaviour actually moves.** Upstream accepts an achieved
    # standoff in its own measured `THROW_STANDOFF_BOUNDS` = (1.09, 1.375), against the
    # [0.210, 1.400] this domain accepted; its lateral tolerance is
    # `2 * WAYPOINT_TOLERANCE` = 0.08 against our 0.04; and it adds a heading conjunct we
    # had none of, so the base must *face* the bin rather than merely sit on its axis.
    # An audit over 1805 poses found 478 disagreements. Everything downstream of this --
    # including `skill_oracle_policy.py`'s branch -- has to hold against upstream's band,
    # not ours.
    holds=lambda state, objects: Tossing3DAtoms.holds(
        state=state, name=KB_AT_THROW_POSE, objects=(objects[0], objects[1])
    ),
)
