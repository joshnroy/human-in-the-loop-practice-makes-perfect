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

# `Toss`'s own dial: the release speed, in **joint-path deg/s**.
#
# ## Why degrees per second, and why "joint-path"
#
# Every measurement of this quantity is written in deg/s -- PR #213's range fit, PR #221's
# 660-cell grid, upstream's own `140` literal, and the real TidyBot primitive
# (`yixuanhuang98/tidybot_real`, `robot/kinova.py:120-124`). Carrying the dial in the same
# unit means a logged action vector reading `140.0` can be checked against those records
# directly. Upstream's `TossController.reset` takes rad/s, so exactly one site converts --
# `KinderBackend.run_toss` -- and `test_run_toss_converts_the_release_speed_to_radians_
# exactly_once` pins it there. A missing conversion drives the arm 57x fast and a doubled
# one 57x slow; neither raises.
#
# "Joint-path" because the speed is the rate along the *path* between upstream's two arm
# configurations, not any single joint's rate and not a Cartesian speed. The path is
# `TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF`, whose L2 norm is 148.8288 deg; joint 6
# carries 0.8399 of it, so joint 6 turns at 0.8399x whatever this number says.
#
# **An m/s figure was considered and deliberately rejected.** Converting needs a Jacobian
# gain, and that gain is not a constant over this range: measured at the realised release
# configurations it runs 0.6156 (at 60) to 0.6622 (at 240) and *sawtooths* rather than
# drifting cleanly, because the realised release fraction quantises to the 10 Hz control
# step. A single-constant m/s label would therefore be wrong by several percent in a
# direction that is not even monotone. Angular rate is exact and needs no constant at all.
#
# ## The range
#
# `(60, 240)` is the span PR #213's grid actually drove, so every draw is a speed something
# has been observed at rather than an extrapolation. There is **no hardware-feasibility
# clamp**, and that is deliberate rather than an oversight: `_ARM_MAX_VEL[5] = 70` deg/s is
# kinder-baselines' own conservative constant, *not* a hardware limit -- the real TidyBot
# primitive this ports runs that joint at 140 deg/s, twice it. See PR #221, which
# established that and retracted the "83.34 deg/s ceiling" an earlier design study asserted.
#
# ## The sawtooth artifact this range used to carry is gone
#
# Release used to fire on the first control step past `fraction_covered >= 0.46`, tested
# once per `_CONTROL_DT = 0.1 s`, so the *realised* release fraction sawtoothed with speed:
# measured paired on (standoff, seed), n = 110, `65 -> 70` deg/s made the throw land
# **0.0329 m shorter**, paired t = +4.51, p = 1.6e-05. That was a fidelity defect of our
# simulation rather than a property of the toss -- the real primitive runs at 1 kHz.
#
# `joshnroy/kinder-baselines` PR #12 replaced the fraction test with a scheduled release on
# an absolute millisecond, so the trigger no longer quantises to 100 ms. The published
# numbers above stand as what was measured; they are simply evidence about the old trigger.
# What is **not** yet established is monotonicity under the new one -- see
# `TOSS_RELEASE_MS_BOUNDS` below, and do not read a draw from this range as monotone in
# landing distance without checking.
TOSS_SPEED_BOUNDS = (60.0, 240.0)

# `Toss`'s second dial: the wall-clock millisecond, measured from the start of the swing,
# at which the gripper opens.
#
# ## Why an absolute millisecond rather than a fraction of the swing
#
# Because that is what the real robot takes. `movej_primitive.execute()` on the TidyBot
# (`yixuanhuang98/tidybot_real`, `robot/kinova.py`) takes exactly `max_vel` and
# `gripper_release_ms`, and kb#12 gave upstream's `TossController.reset` the same two, so
# both of this skill's parameters are now upstream's knobs rather than quantities this
# package synthesised. Carrying a fraction here would re-introduce a derived unit on our
# side of the boundary.
#
# **The real robot's own literal 600 is deliberately not copied across.** That primitive
# normalises its joint profile on the **L-infinity** norm (125.0 deg) and finishes in
# 1476 ms, so its 600 ms is fraction 0.4107 of its swing; 600 ms here would be 0.3449 of
# this one. The two swings are not the same length, so the literal does not transfer --
# only the parameterisation does.
#
# ## The range, and why it is this narrow
#
# **The swing's duration is itself a function of `release_speed`**, so the two dials are
# coupled and an absolute-millisecond range cannot mean the same thing at both ends.
# Measured from `toss_profile_limits` + `_trapezoidal_motion_profile` on upstream's own
# path (`TOSS_RELEASE_ARM_CONF - TOSS_WINDUP_ARM_CONF`, L2 norm 148.8288 deg):
#
# | `release_speed` deg/s | 60 | 100 | 140 | 180 | 240 |
# | --- | --- | --- | --- | --- | --- |
# | swing duration ms | 3100 | 2100 | 1700 | 1500 | 1300 |
# | path fraction at 723 ms | 0.197 | 0.329 | 0.461 | 0.591 | 0.732 |
#
# So the swing is **2.4x longer at the bottom of `TOSS_SPEED_BOUNDS` than at the top**, and
# one fixed millisecond walks from a fifth of the way through the swing to three quarters
# of the way through it as the speed rises.
#
# The upper edge is set by the *shortest* swing in `TOSS_SPEED_BOUNDS`, 1300 ms at
# 240 deg/s. `gripper_release_ms` is deliberately **not clamped** upstream: a value at or
# past the end of the swing means the gripper never opens and the cube is never thrown.
# That is a real and reachable corner of the space rather than an error, and the
# characterisation sweep has to be able to land in it -- but a *sampler* drawing there
# would spend those draws on throws that cannot score, so the bounds stop at the point
# where every speed in range still releases.
#
# > **What this range gives up, stated plainly.** At 60 deg/s the canonical release point
# > -- path fraction 0.46, which is where the retired trigger fired and where every
# > committed Tossing3D number was measured -- falls at **1375 ms**, outside this range.
# > So a slow throw drawn from these bounds always releases *earlier* in its swing than a
# > fast one does, and the two dials are not independent in effect even though they are
# > independent in the sampler. Making the second dial speed-relative would fix that at the
# > cost of no longer being upstream's own parameter; that trade is recorded here rather
# > than taken, and is the obvious follow-up if the sweep shows the slow end starved.
TOSS_RELEASE_MS_BOUNDS = (300.0, 1300.0)

# Upstream's own shipped default, and the millisecond the retired 0.46 fraction fell at for
# the shipped windup->release path at 140 deg/s -- so a caller passing nothing reproduces
# the throw every earlier result was measured against. Not the real robot's 600; see
# `TOSS_RELEASE_MS_BOUNDS` for why that literal does not transfer.
UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS = 723.0

# The speed every committed Tossing3D number was measured at, and still upstream's own
# default: the literal that was inline in `TossController.reset` before kb#8 turned it into
# a parameter, and what `toss_profile_limits()` returns when passed nothing.
UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S = 140.0

# **The one calibrated constant in this module, and the only thing the success band is not
# derived from.** `run_toss` executes `move_arm_to_conf(windup_conf_deg)` and then
# `toss(toss_conf_deg)`, both fixed joint configurations, so a throw displaces the cube by
# the same distance in the base's facing direction every time. That displacement is this
# number, in metres.
#
# > **Scope note, 2026-08-12.** This constant is the impact range **at 140 deg/s only**.
# > It was calibrated when `Toss` had no parameters and every throw was the same throw;
# > `TOSS_SPEED_BOUNDS` now lets a caller ask for 60-240 deg/s, over which the range
# > genuinely varies. Nothing below has been recomputed and the measurements it cites stand
# > exactly as published -- they are simply evidence about one speed.
# >
# > **The consequence is live, not hypothetical**: `RobotAtSuccessfulThrowPoseClassifier`
# > derives its acceptance band from this single number, so it currently answers "is this a
# > pose a *140 deg/s* throw scores from", while `Toss` may be executed at any speed in the
# > bounds. Reformulating that predicate as a union over the speed range is the next PR in
# > this stack; until it lands, the predicate is correct only for the oracle's speed. This
# > is recorded here rather than left implicit because a silently speed-conditional
# > precondition is exactly the kind of thing
# > `tests/environments/test_operator_dynamics_fidelity.py` exists to forbid.
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
THROW_RANGE = 1.275

# Upstream's `WAYPOINT_TOL` (`kinder_models/dynamic3d/utils.py:54`), which is how close
# `MoveToTargetGroundController.terminated()` requires the base to be to its own planned
# waypoint. Using upstream's own number means the lateral conjunct admits exactly the poses
# that controller is willing to stop at, rather than a tolerance invented here.
THROW_POSE_LATERAL_TOLERANCE = 4 * 1e-2

# The two margins that tighten `RobotAtSuccessfulThrowPoseClassifier`'s standoff band from
# the full geometric prediction to the band PR #105's finer sweep (5 scene seeds, 0.025 m
# resolution, coincident config) actually found reliable. Its per-point solving table:
#
#     standoff   <=1.100   1.125   1.150-1.375   1.400   1.425   >=1.450
#     solved       0/5      2/5     5/5 (every)    3/5     2/5      0/5
#
# The 5/5 core is `[1.150, 1.375]` -- narrower than the geometric band `[1.125, 1.425]` on
# *both* ends, and not symmetrically: "short standoffs overshoot [the goal box] ... long
# ones fall short" (PR #105's own wording). Overshoot happens at the box's far edge
# (`x_max`), so the margin that excludes it trims `x_max`; falling short happens at the
# near edge (`x_min`), so the margin that excludes it enlarges `x_min`.
#
# **The two spaces run in opposite directions, and that is the trap.** `landing_x =
# base_x + THROW_RANGE` and `base_x = bin_x - standoff`, so a *larger* standoff produces a
# *smaller* `landing_x`. Recovering the standoff band `[1.150, 1.375]` from the box
# therefore means: `x_max` shrinks by `1.150 - 1.125 = 0.025` (this is the short-standoff/
# overshoot margin, even though it is the *upper* box edge), and `x_min` grows by
# `1.425 - 1.375 = 0.050` (the long-standoff/shortfall margin, on the *lower* box edge).
# Naming the margins by which standoff failure mode they exclude, rather than by which box
# edge they touch, is deliberate -- the box-edge framing is exactly what inverts under the
# sign flip. `test_the_accepted_band_matches_the_measured_five_of_five_core` pins the
# resulting band directly so a reintroduced inversion fails loudly.
#
# **A further tightening to roughly `[1.150, 1.325]` was investigated and rejected.** A
# later, wider grid over `MoveToThrowPose`'s standoff (PR #196) labelled by this classifier
# alone -- `Toss` never executed -- found only 8/12 solving at 1.350 and 2/12 at 1.375,
# suggesting the band above 1.325 was over-permissive. A fresh oracle-driven sweep that
# *does* execute `Toss` and reads the real episode outcome (`docs/experiment-logs/
# 2026-08-10-tossing3d-throw-band-retightening-sweep.md`, same methodology as the sweep
# above) found 10/10 at every standoff up to and including 1.375 on an independent seed
# set -- no real degradation. The two disagree because PR #196's grid measured whether the
# achieved `landing_x` clears this classifier's own fixed threshold, and at 1.350/1.375 that
# threshold sits inside `move_to_target`'s own several-millimetre stopping noise (closest
# miss at 1.375: 0.1 mm), so ordinary seed-to-seed pose variance flips the *label* without
# the *throw* ever becoming less reliable. Tightening to 1.325 would also have pushed
# `ORACLE_THROW_STANDOFF` (1.35, below) outside the accepted band, rejecting a pose the
# committed oracle arm throws from on every episode. The margins below are unchanged.
THROW_OVERSHOOT_MARGIN = 0.025
THROW_SHORTFALL_MARGIN = 0.05


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
    fail is a skill whose sampler cannot learn, and this skill's standoff is the one
    continuous parameter in the domain that decides the outcome: `Toss`, which does decide
    it, has `param_dim = 0`.

    **The standoff conjunct is derived from live state, not measured and pinned.** `Toss`
    takes no parameters -- fixed windup conf, fixed toss conf -- so a throw displaces the
    cube by the constant `THROW_RANGE` in the base's facing direction. `MoveToThrowPose`
    pins `rot = 0` and the bin's yaw range is `[[0, 0]]`, so a base satisfying the lateral
    conjunct faces `+x` and the cube's predicted resting place is `base_x + THROW_RANGE`.
    `InBin` tests the cube's centre against the scored box, which the `State` already
    carries on the bin as the live `Region.bbox`. So the test below is *that same box*,
    read off *that same object*, trimmed by the two margins above and applied to the
    predicted landing point -- and the accepted band of standoffs

        [bin_x + THROW_RANGE - (x_max - THROW_OVERSHOOT_MARGIN),
         bin_x + THROW_RANGE - (x_min + THROW_SHORTFALL_MARGIN)]

    falls out rather than being written down. Move the bin, resize the scored box, or
    change `ground_placement_threshold`, and the band follows on its own -- which matters,
    because kindergarden#126 moves the bin. A hard-coded band would be silently wrong the
    moment that lands. The two margins are fixed metres, not a fraction of the box, so they
    do not move with it.

    That this predicate and `InBin` now read their box off the same object is what makes
    "the pose a throw succeeds from" and "the place a throw must land" provably the same
    geometry rather than two things kept in step by hand.

    On the coincident config the *geometric* band -- before the two margins below -- works
    out to `[1.125, 1.425]`: **0.300 m wide, which is exactly the scored box's own
    x-extent**, as it must be for a constant-displacement throw. That geometric band is
    over-permissive: PR #105's finer sweep (5 scene seeds, 0.025 m resolution) found the
    band solving on *every* seed is `[1.150, 1.375]`, 0.225 m wide, with the geometric
    band's own edges only partially solving (2/5 at 1.125, 3/5 at 1.400, 2/5 at 1.425).
    `THROW_OVERSHOOT_MARGIN`/`THROW_SHORTFALL_MARGIN` trim the box used below by exactly
    that 0.025 m / 0.050 m so the accepted standoff band matches the measured 5/5 core
    instead of the wider, partially-reliable geometric prediction. Training `MoveToThrowPose`'s
    sampler against the untrimmed band taught it that the outer ~0.075 m sliver on each
    side was as good as the centre, when empirically it is not -- a plausible mechanistic
    account of `Toss`'s own residual failures at the trained EES plateau (`#178`).

    **The lateral conjunct is unchanged, and was measured.** An earlier version tested only
    `1.0 <= hypot(dx, dy) <= 1.8`. After `Pick` -- which drives the base to the *cube*, off
    to one side -- the base sat at 1.76 m from the bin, inside that band, so the predicate
    was already true, the oracle skipped `MoveToThrowPose` entirely and threw from a pose
    facing 40 degrees away from the bin: the cube landed at `(0.9969, -0.7196)` and the
    episode scored a failure. The lateral conjunct rules that out by 0.72 m rather than by
    the 7 cm the distance test had to spare. This is exactly the over-permissive-operator-
    model defect class that `tests/environments/test_operator_dynamics_fidelity.py` exists
    for, caught here by
    `test_the_oracle_reproduces_the_recorded_coincident_landing_and_step_counts`.

    The standoff conjunct now independently excludes those poses too -- the post-`Pick`
    base is far enough back that its predicted landing falls short of the box -- but the
    lateral conjunct stays, because it is the one measured against the real failure, and
    because a base off the axis does not throw along `+x` at all.
    """

    @staticmethod
    def holds(*, state: State, robot: Object, target: Object) -> bool:
        lateral_offset = abs(
            state.get(obj=robot, feature_name="pos_base_y")
            - state.get(obj=target, feature_name="y")
        )
        if lateral_offset > THROW_POSE_LATERAL_TOLERANCE:
            return False
        landing_x = state.get(obj=robot, feature_name="pos_base_x") + THROW_RANGE
        x_min = state.get(obj=target, feature_name="x_min") + THROW_SHORTFALL_MARGIN
        x_max = state.get(obj=target, feature_name="x_max") - THROW_OVERSHOOT_MARGIN
        return bool(x_min <= landing_x <= x_max)


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
