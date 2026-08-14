"""Tossing3D's symbolic layer: six predicates, all pure arithmetic over `core.State`.

**KINDER ships no symbolic model for Tossing3D.** `kinder_bilevel_planning.env_models.
dynamic3d` has one for base motion, one for `Shelf3D` and one for `Sweep3D`, and that is
all -- so unlike the controllers, which are upstream's verbatim, these are ours. The
shape they follow is `tidybot3d_shelf3D.py`'s: a small set of `Predicate`s over the
scene's objects, consumed by `LiftedOperator`s that are paired to upstream's controllers
(there, `LiftedSkill(PickTargetOperator, LiftedPickShelfController)`; here, `skills.py`'s
`Skill`s plus `skill_provider.py`).

Three of the six are ported from upstream's own `kinder_models.dynamic3d.shelf.
state_abstractions`, whose `HandEmpty`/`Holding`/`OnGround` classify the same TidyBot
state this domain reads -- thresholds included, so they are upstream's numbers rather
than ours. Two are genuinely new, and are called out below.

## The stated assumption: the bin's interior **is** the scored region

Every predicate below that needs a landing box reads it off the **bin**, and no predicate
and no operator takes a goal region as an argument. That is an assumption this domain
makes, deliberately, and it is **false in general**.

What is true in general is that KINDER scores `["on", "cube_0", "blocks_goal_region"]` --
a ground *region*, which is a box in the scene, entirely independent of where the bin is.
Under upstream's stock `Tossing3D-o1` the two are 23 cm apart: the bin sits at x = 2.2305
while `blocks_goal_region` inflates to x in [1.85, 2.15], so **a cube that lands in the bin
scores a failure** and only one that misses the bin and lands on bare floor scores a
success. There is a commit in this repo's history titled "Tossing3D: the bin is scenery"
for exactly that reason. Our default `scripts/task_configs/Tossing3D-o1-coincident.json`
is *named* for putting the bin back at x = 2.0 so that the two coincide -- measured live,
they then agree to 0.1 mm.

So this is a modelling choice, not an observation. Three consequences, stated rather than
left to be discovered:

1. **Fidelity is preserved, because the number is unchanged.** The box these classifiers
   read is still the live `Region.bbox` of `blocks_goal_region`
   (`KinderBackend.goal_region_bbox`), carried in the `State` on the bin object. Nothing is
   re-derived from the bin's pose or from a literal, so `InBin` still agrees with
   `_check_goals()` exactly, on **both** task configs. What was dropped is the *symbolic*
   dependence -- an object a planner had to bind and no skill could act on -- not the
   classifier's access to scene geometry.
2. **Under `Tossing3DTaskConfig.STOCK` the name lies, and the bin object is internally
   inconsistent.** Its own `x` (2.2305) sits outside its own `x_max` (2.15). `InBin` is then
   true exactly when the cube is *not* in the bin. The predicate remains *correct* -- it
   still scores what KINDER scores -- but it is misnamed there, and any reading of a stock
   run's symbolic trace has to account for that. Stock stays selectable because every
   number in `docs/kinder-environment-validation.md` was measured on it.
3. **Wherever the bin is taken to be movable, moving it moves the goal.** kindergarden#126
   moves the bin, and the rest of this module is written so the accepted band follows live
   scene geometry rather than a literal; under this assumption the scored box is *part of*
   that geometry, so relocating the bin retargets the throw rather than leaving a stale
   target behind. That is the intended reading of a move-the-bin task -- "put the cube in
   the bin, wherever the bin is" -- rather than a problem to work around. It is worth
   stating because the opposite reading, a fixed goal the bin merely decorates, is exactly
   what the stock config implements.

Where the port deviates from upstream, once, and deliberately: upstream's `Holding` also
requires the end-effector to be within 5 cm of the object, computed by forward kinematics
through a live `PyBulletSim`. A `core.Predicate.holds` is a pure function of `State` with
no simulator handle -- that is the whole reason planning can happen off-simulator -- so
that third conjunct is dropped, leaving upstream's other two (gripper closed past
`GRASP_THRESHOLD`, object lifted past `HOLDING_HEIGHT`). The result is *weaker* than
upstream's: it can call a cube "held" that is closed-gripper and airborne without being
grasped. With one cube and a grasp that either works or drops it on the floor, no
observed state distinguishes them, but it is a real difference and not a restatement.
"""

import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import Tossing3DEnvironment

# Upstream's own thresholds, from `kinder_models/dynamic3d/shelf/state_abstractions.py`.
# Named here rather than inlined so a future upstream bump has one place to check.
HANDEMPTY_TOL = 1e-3  # `handempty_tol`
GRASP_THRESHOLD = 0.1  # `GraspThreshold`
HOLDING_HEIGHT = 0.1  # the `state.get(target, "z") > 0.1` conjunct
ON_GROUND_TOL = 0.05  # `on_ground_tol`

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

# **The one calibrated constant in this module, and the only thing the success band is not
# derived from.** `run_toss` executes `move_arm_to_conf(windup_conf_deg)` and then
# `toss(toss_conf_deg)`, both fixed joint configurations, so a throw displaces the cube by
# the same distance in the base's facing direction every time. That displacement is this
# number, in metres.
#
# > **Scope note, 2026-08-12.** This is the impact range at 140 deg/s only, and
# > `TOSS_SPEED_BOUNDS` spans 60-140. `RobotAtSuccessfulThrowPoseClassifier` derives its
# > acceptance band from it, so that predicate is correct only at the oracle's speed until
# > it is reformulated as a union over the range.
#
# Because it is a property of the *controller* -- upstream's arm configurations and the
# cube's 0.1 kg mass -- and not of the scene, the success band can be recomputed from live
# state on every call and stays correct when the bin or the goal region moves. That is the
# whole point: kindergarden#126 moves the bin, and nothing here needs to change.
#
# **Calibrated against where success breaks, not measured in free flight, and the
# difference is a trap.** The obvious measurement -- throw onto open floor and record where
# the cube comes to rest -- gives **1.3499 m** (sd 0.0024, n = 12, standoffs 0.45-1.00 over
# 3 seeds) and is **wrong for this purpose**: it includes post-impact roll along the floor.
# What the goal-region test needs is the *impact* range, because on the coincident config
# the bin sits on the goal region and a cube that impacts inside it is caught by the bin
# walls instead of rolling on. The two differ by the ~0.075 m of roll.
#
# Two independent sweeps agree on the impact range:
#
# - A 48-episode grid (16 standoffs x 3 scene seeds, coincident config, Pick ->
#   MoveToThrowPose(d) -> Toss): solved 0/3 at d = 1.10, 3/3 at 1.15, 3/3 through 1.35,
#   2/3 at 1.40, 0/3 at 1.45. Reading the *measured* base poses against the goal box's own
#   edges brackets the constant to (1.2608, 1.3090).
# - PR #105's finer sweep (5 seeds, 0.025 m resolution) put the edges of partial solving at
#   1.125 and 1.425. Those two edges imply **1.2749 from each end independently** -- which
#   is the strongest evidence here that the constant-displacement model is right at all,
#   since nothing forced the two ends to agree.
#
# 1.275 sits inside both brackets. `test_throw_range_predicts_where_the_cube_lands`
# re-measures it against the real simulator, so upstream changing the toss controller, the
# windup conf or the physics fails loudly rather than silently.
#
# > **Provisional as of 2026-08-13; left as published, NOT recomputed.** The brackets above
# > were measured with the gripper opening on the first control step past fraction 0.46.
# > Scheduling on an absolute millisecond moves the oracle's landing +41.6 mm, so read
# > 1.275 as approximate. The guard tests still pass on the current pins; re-deriving means
# > redoing PR #105's 5-seed, 0.025 m sweep.
THROW_RANGE = 1.275

# The interval `RobotAtSuccessfulThrowPose` unions over: shortest and longest reach
# across the toss parameter box, as distance covered before first ground contact (the
# quantity `THROW_RANGE` is calibrated in, not resting x). Measured over 3359
# unobstructed rows of PR #240's grid and this stack's standoff grid, envelopes
# [0.3330, 1.3465] and [0.3386, 1.3552]. Resting x instead jumps 244 mm at ~710 ms on
# bin contact, and dropping those rows censors the longest throws: the two grids then
# disagree by 178 mm, against 8.7 mm on reach.
#
# Derived from the toss parameter box, never from `THROW_STANDOFF_BOUNDS` -- otherwise
# the `NearBin` tautology returns by another door.
THROW_RANGE_MIN = 0.333

# Measured maximum 1.3552, trimmed 55.2 mm to put the accepted band's far edge on
# standoff 1.400, the last observed to score (`0/1050` beyond it, `3/150` at it):
# `1.400 - (bin_x - x_min - THROW_SHORTFALL_MARGIN)` = `1.400 - (2.0 - 1.85 - 0.05)`.
# Scene-coupled: re-derive by re-running the standoff grid, not by editing the literal.
THROW_RANGE_MAX = 1.300

# Upstream's `WAYPOINT_TOLERANCE` (`kinder_models/dynamic3d/utils.py`), which is how close
# `MoveToTargetGroundController.terminated()` requires the base to be to its own planned
# waypoint.
WAYPOINT_TOLERANCE = 4 * 1e-2

# Upstream's own `THROW_POSE_TOLERANCE`, restated: twice the waypoint tolerance, on
# upstream's stated grounds that "the sampler already spends half of it off-axis". It
# bounds both the lateral offset and the heading error below.
#
# **Restated rather than imported.** Every classifier in this module is a pure function of
# `core.State` with no simulator handle, and importing anything from `kinder_models` here
# would make this module -- and through `tasks.py`, the whole `--env tossing3d`
# registration -- require MuJoCo. `test_kinder_pin.py` asserts these constants still equal
# upstream's, so a drift fails loudly wherever KINDER is installed while CI keeps running
# the offline predicate tests.
THROW_POSE_TOLERANCE = 2 * WAYPOINT_TOLERANCE

# The band of **achieved** standoffs `RobotAtSuccessfulThrowPose` accepts, in metres.
#
# **Not `THROW_STANDOFF_BOUNDS` above, and the two must never become one symbol again.**
# That one is the interval the sampler *draws* from; this one is what the predicate
# *accepts*. Their identity is what made `MoveToThrowPose`'s add effect constant-true and
# its sampler unlearnable. Inside the sampler's (1.10, 1.75) the band below covers
# 0.300/0.650 of the range, so the label stays two-class.
#
# The near edge is upstream's own measured 1.09. **The far edge is 1.400, this repo's,
# where upstream's is 1.375** -- see `RobotAtSuccessfulThrowPoseClassifier`'s docstring
# for why, and for the question that was deliberately left open when it moved.
ACCEPTED_THROW_STANDOFF_BOUNDS = (1.09, 1.400)

# PR #105's finer sweep (5 scene seeds, 0.025 m resolution) measured this per-point
# solving table, which the band above still has to contain:
#
#     standoff   <=1.100   1.125   1.150-1.375   1.400   1.425   >=1.450
#     solved       0/5      2/5     5/5 (every)    3/5     2/5      0/5
#
# **A further tightening to roughly `[1.150, 1.325]` was investigated and rejected**, and
# the reason is the same one that keeps the far edge at 1.400 today. A wider grid over
# `MoveToThrowPose`'s standoff (PR #196) labelled by this classifier alone -- `Toss` never
# executed -- found only 8/12 solving at 1.350 and 2/12 at 1.375. A fresh oracle-driven
# sweep that *does* execute `Toss` (`docs/experiment-logs/
# 2026-08-10-tossing3d-throw-band-retightening-sweep.md`) found 10/10 at every standoff up
# to and including 1.375 on an independent seed set. The two disagree because PR #196
# measured whether the achieved pose clears the classifier's own threshold, and at
# 1.350/1.375 that threshold sits inside `move_to_target`'s own stopping noise, so ordinary
# seed-to-seed pose variance flips the *label* without the *throw* becoming less reliable.
# Tightening to 1.325 would also have pushed `ORACLE_THROW_STANDOFF` (1.35) outside the
# band, rejecting a pose the committed oracle arm throws from on every episode.


class InBinClassifier:
    """The cube's centre lies inside the bin's interior.

    This is the domain's success criterion, and it is written to agree with KINDER's own
    `_check_goals()` **exactly**, not approximately. That is checkable rather than
    aspirational: for `["on", "cube_0", "blocks_goal_region"]` on a ground region,
    `_check_goals` (`envs.py:1053-1167`) builds `[x, y, z]` from the object's own state
    features and hands it to `Region.check_in_region`, which is a plain inclusive
    point-in-box test against `Region.bbox` (`objects/base.py:148-185`). Both halves are
    reproduced here: the box arrives in the `State` as the live `bbox`
    (`KinderBackend.goal_region_bbox`), carried on the bin, and the comparison below is the
    same six inclusive bounds.

    **"The bin's interior" is this domain's assumption, not KINDER's arrangement** -- see
    this module's docstring for what it buys, and for the stock config where the name is
    wrong while the arithmetic stays right. The name is chosen to read honestly under the
    default config every committed number on this domain is measured at, where the bin and
    the scored region coincide to 0.1 mm and "the cube is in the bin" is exactly what
    KINDER scores. The alternative -- keeping `InGoalRegion` -- would name a thing that no
    longer exists as an object anywhere in the domain, which is worse: it would invite a
    reader to look for a goal-region argument that is gone.

    Inclusive on purpose (`<=`, not `<`), because upstream's is.

    The box is read from the state rather than re-derived from the task JSON, and *never*
    from the bin's own pose. The JSON's range is inflated by `ground_placement_threshold`
    (0.05 m per side, z clamped at 0) before it becomes a region, so the literal in the
    file is 2/3 of the true width on x -- the axis a toss controls -- and a predicate
    written against it scores KINDER successes as failures. That defect has already shipped
    once in this project's history. Deriving the box from the bin's pose plus a half-extent
    would be the same defect in a new costume.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, target: Object) -> bool:
        position = [state.get(obj=cube, feature_name=name) for name in ("x", "y", "z")]
        lower = [state.get(obj=target, feature_name=name) for name in ("x_min", "y_min", "z_min")]
        upper = [state.get(obj=target, feature_name=name) for name in ("x_max", "y_max", "z_max")]
        return all(
            low <= value <= high for value, low, high in zip(position, lower, upper, strict=True)
        )


class HandEmptyClassifier:
    """The gripper is open. Upstream's `HandEmpty`, verbatim including `handempty_tol`."""

    @staticmethod
    def holds(*, state: State, robot: Object) -> bool:
        gripper = state.get(obj=robot, feature_name="pos_gripper")
        return bool(np.isclose(gripper, 0.0, atol=HANDEMPTY_TOL))


class HoldingClassifier:
    """The gripper is closed and the cube is off the floor.

    Upstream's `Holding` minus its forward-kinematics conjunct -- see this module's
    docstring for why that one cannot be evaluated here and what it costs.
    """

    @staticmethod
    def holds(*, state: State, robot: Object, cube: Object) -> bool:
        gripper = state.get(obj=robot, feature_name="pos_gripper")
        return bool(
            gripper > GRASP_THRESHOLD and state.get(obj=cube, feature_name="z") > HOLDING_HEIGHT
        )


class OnGroundClassifier:
    """The cube is resting flat on the floor. Upstream's `OnGround`, verbatim.

    Flatness (`qx`/`qy` near zero) is upstream's own condition and is load-bearing rather
    than decorative: `pick_shelf` builds its grasp pose from the object's orientation, so
    a cube that came to rest on a corner is not a cube this grasp is modelled on.
    """

    @staticmethod
    def holds(*, state: State, cube: Object) -> bool:
        z = state.get(obj=cube, feature_name="z")
        bb_z = state.get(obj=cube, feature_name="bb_z")
        return bool(
            np.isclose(z - bb_z / 2, 0.0, atol=ON_GROUND_TOL)
            and np.isclose(state.get(obj=cube, feature_name="qx"), 0.0, atol=ON_GROUND_TOL)
            and np.isclose(state.get(obj=cube, feature_name="qy"), 0.0, atol=ON_GROUND_TOL)
        )


class ReachableClassifier:
    """The cube is on the robot's side of the barrier.

    **Ours, and the one that carries this domain's whole point.** The barrier is a 5 m
    immovable wall the base cannot pass, so a cube past it can never be picked up again --
    the irreversibility that makes this domain interesting is exactly "`Reachable` is a
    one-way door". Making `Pick` require it, and `Toss` delete it, is what stops a planner
    emitting the retrieve-and-retry plan the dynamics can never execute.

    Compared against the barrier's own live x rather than a constant: the barrier's pose
    is sampled from `barrier_init_region` per episode, so a literal would be right for one
    seed and quietly wrong for the next.
    """

    @staticmethod
    def holds(*, state: State, cube: Object, barrier: Object) -> bool:
        return bool(
            state.get(obj=cube, feature_name="x") < state.get(obj=barrier, feature_name="x")
        )


class RobotAtSuccessfulThrowPoseClassifier:
    """The base is standing somewhere a throw from here lands the cube in the bin: on the
    bin's axis, and at a standoff from which the throw's fixed displacement carries the
    cube into the scored box.

    **Ours.** `move_to_target` has no symbolic model upstream, and its own termination
    condition (`_robot_is_close_to_pose`) is about the base having reached *its own
    planned waypoint*, which says nothing about where that waypoint was. So the effect a
    planner needs has to be stated here.

    **Why this is a success test and not a reachability test.** This was previously
    `NearBin`, whose docstring called it "an exact characterisation of what
    `MoveToThrowPose` produces" -- and that is exactly what was wrong with it. It accepted
    every standoff in `THROW_STANDOFF_BOUNDS`, the interval the sampler draws from, so
    `MoveToThrowPose`'s only add effect held after *every* attempt. EES trains one success
    classifier per skill on precisely that label, so the label was constant-true, the
    classifier saw a single class, and the sampler fell back to uniform on every draw
    forever -- 16/16 attempts labelled success with 0/16 informed draws in a probe run,
    against 7/20 informed draws for `Pick` in the same run. A skill whose add effect cannot
    fail is a skill whose sampler cannot learn.

    **The standoff conjunct stays even though `Toss` has parameters of its own.**
    Dropping it would hand `MoveToThrowPose` a constant-true label again.

    ## Converged onto kinder-baselines' `RobotAtThrowPose`, with one edge moved

    The three conjuncts below are upstream's, in upstream's own formulation: the standoff
    is `dx = target.x - pos_base_x` tested against a **literal** band, the lateral offset
    and the heading error share one tolerance, and the heading error is the shortest signed
    angle between the bearing to the target and the base's own yaw.

    **The classifier is reimplemented here rather than imported, and that is deliberate.**
    Upstream's lives on a `Tossing3DStateAbstractor` that needs a live
    `ObjectCentricTidyBot3DEnv` and reads `relational_structs.ObjectCentricState`. A
    `core.Predicate.holds` is a pure function of `core.State` with no simulator handle,
    which is what lets ~900 lines of offline predicate and skill tests run on CI, where
    KINDER is never installed. Converging the *definition* keeps that; converging the
    *call path* would not. An audit measured the two agreeing on `12/12` real rollout
    states for four of the six predicates before this one was reconciled.

    **What converging cost: the band no longer follows the scored region.** The previous
    band was derived per call from the live `Region.bbox` plus a calibrated throw range, so
    retargeting the throw in the task JSON needed no change here -- written that way
    precisely because kindergarden#126 moves the bin. Upstream's is a literal, and its own
    comment concedes the consequence: "A literal, not a live read of the region, so a scene
    whose bin or goal region moves needs remeasuring." The band is still coupled to the
    bin's own pose, since `dx` is measured from it;
    `test_the_accepted_band_no_longer_follows_the_goal_region` asserts the half that was
    given up, so it reads as a decision rather than a regression.

    ## The far edge is 1.400, not upstream's 1.375

    `move_to_target` stops long of its commanded standoff by a per-seed constant, measured
    at 0.7-27.7 mm across 5 scene seeds, so the oracle's commanded 1.35 arrives at an
    *achieved* `dx` of up to 1.3777. Upstream's 1.375 edge falls inside that spread.
    `SkillOraclePolicy` branches directly on this predicate, so at 1.375 the oracle would
    find itself not-at-a-throw-pose after a `MoveToThrowPose` that did exactly what it was
    asked, and would re-issue it forever rather than throw -- `2/5` seeds, deterministically.

    That alone does not say which side is wrong, so it was measured: 40 oracle rollouts
    (8 scene seeds x commanded standoffs 1.325-1.425), run through to the throw, reading
    KINDER's own `_check_goals()`. **Every throw from an achieved `dx` up to and including
    1.4462 scored; the first miss is at 1.4464.** Throws from 1.3777 therefore score, so
    upstream's band rejects poses that demonstrably work -- its own comment claims both
    edges move "~9 mm across seeds", and eight seeds show 28 mm of stopping error alone.

    **1.400 is deliberately conservative rather than fitted to that measurement.** It is
    this repo's own previous far edge -- the last standoff observed to score on the
    committed standoff grid (`3/150` at 1.400, `0/1050` beyond) -- and it clears the worst
    measured achieved `dx` at the oracle standoff by 22.3 mm. The open question it leaves
    is the opposite of the one it closes: the band could be widened towards ~1.446 and is
    not, so the predicate still calls some genuinely-scoring poses a failure. That is the
    safe direction for a precondition, and widening it should be its own change with its own
    sweep. Note the committed grid's `0/1050` was measured under the pre-1 kHz release
    scheduling, which lands the cube ~41.6 mm shorter, so the two are not in conflict.

    **The lateral conjunct was measured against a real failure.** An earlier version
    tested only `1.0 <= hypot(dx, dy) <= 1.8`. After `Pick` -- which drives the base to the
    *cube*, off to one side -- the base sat at 1.76 m from the bin, inside that band, so
    the predicate was already true, the oracle skipped `MoveToThrowPose` entirely and threw
    from a pose facing 40 degrees away from the bin: the cube landed at `(0.9969, -0.7196)`
    and the episode scored a failure. `test_widening_the_lateral_conjunct_does_not_readmit_
    the_recorded_failure` asserts that pose stays rejected under the widened tolerance --
    it fails both conjuncts, `|dy| = 0.366` against 0.08 and `dx = 1.721` against a band
    topping out at 1.400.

    **What the heading conjunct adds, stated at the strength it was measured.**
    `MoveToThrowPose` pins `rot = 0`, so a base it placed always faces the bin. Measured
    over 75 real states: `pos_base_rot` ranges 0.0003-1.3919 rad, and after `Pick` the
    heading error is 0.29-0.92 rad against the 0.08 tolerance -- so this is a genuine
    function of a free state variable, not a tautology. But it was **never** the conjunct
    that decided an observed state (`0/75` where lateral and standoff pass while heading
    fails), and at post-`MoveToThrowPose` states -- the ones that label the sampler --
    heading and lateral both pass `25/25` while only the standoff conjunct varies
    (`13/25`). It is kept because it is upstream's, and because it costs nothing.
    """

    @staticmethod
    def _signed_angle_distance(*, target: float, source: float) -> float:
        """The shortest signed angle `d` with `source + d = target`.

        Upstream's `prpl_utils.utils.get_signed_angle_distance`, restated for the same
        reason the constants above are -- see `THROW_POSE_TOLERANCE`. Reproducing the wrap
        rather than subtracting matters: a base facing `+x` reported as `-pi` instead of
        `+pi` is the same heading, and a naive difference makes it `2*pi` wrong.
        """
        return float((target - source + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def holds(*, state: State, robot: Object, target: Object) -> bool:
        dx = state.get(obj=target, feature_name="x") - state.get(
            obj=robot, feature_name="pos_base_x"
        )
        dy = state.get(obj=target, feature_name="y") - state.get(
            obj=robot, feature_name="pos_base_y"
        )
        heading_error = abs(
            RobotAtSuccessfulThrowPoseClassifier._signed_angle_distance(
                target=float(np.arctan2(dy, dx)),
                source=float(state.get(obj=robot, feature_name="pos_base_rot")),
            )
        )
        if abs(dy) > THROW_POSE_TOLERANCE or heading_error > THROW_POSE_TOLERANCE:
            return False
        low, high = ACCEPTED_THROW_STANDOFF_BOUNDS
        return bool(low <= dx <= high)


# `Predicate.holds` is a positional `(state, objects)` callable per its interface contract
# (`Goal.is_satisfied` calls it that way), so each lambda below adapts that into a call to
# the relevant class's keyword-only `holds` -- exactly as Light Switch and Tossing Room do.
IN_BIN = Predicate(
    name="InBin",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: InBinClassifier.holds(
        state=state, cube=objects[0], target=objects[1]
    ),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(Tossing3DEnvironment.robot_type,),
    holds=lambda state, objects: HandEmptyClassifier.holds(state=state, robot=objects[0]),
)

HOLDING = Predicate(
    name="Holding",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.cube_type),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], cube=objects[1]
    ),
)

ON_GROUND = Predicate(
    name="OnGround",
    types=(Tossing3DEnvironment.cube_type,),
    holds=lambda state, objects: OnGroundClassifier.holds(state=state, cube=objects[0]),
)

REACHABLE = Predicate(
    name="Reachable",
    types=(Tossing3DEnvironment.cube_type, Tossing3DEnvironment.barrier_type),
    holds=lambda state, objects: ReachableClassifier.holds(
        state=state, cube=objects[0], barrier=objects[1]
    ),
)

ROBOT_AT_SUCCESSFUL_THROW_POSE = Predicate(
    name="RobotAtSuccessfulThrowPose",
    types=(Tossing3DEnvironment.robot_type, Tossing3DEnvironment.bin_type),
    holds=lambda state, objects: RobotAtSuccessfulThrowPoseClassifier.holds(
        state=state, robot=objects[0], target=objects[1]
    ),
)
