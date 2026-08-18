"""Tossing3D's privileged solver: the three-skill sequence and its parameters.

The whole policy is a three-way branch on the symbolic state -- pick, walk to the throw
pose, throw -- because the domain admits exactly one plan shape. What makes it an
*oracle* rather than a fixed script is the continuous parameters, which are the part a
learner would have to find -- all upstream's own published values except
`ORACLE_GRIPPER_RELEASE_MS`, which is ours and measured:

- **`ORACLE_PICK_DISTANCE` / `ORACLE_PICK_ROTATION`** are the pair upstream's own
  `PickShelfController.sample_parameters` draws from `np.random.default_rng(123)` -- the
  rng that upstream's `test_pick_ground_toss` constructs to parameterize this exact
  grasp. With one cube in the scene upstream's rejection loop has nothing to reject
  against, so its first draw is accepted and the sampler reduces to two uniforms over
  `MOVE_TO_TARGET_DISTANCE_BOUNDS` and `MOVE_TO_TARGET_ROT_BOUNDS`. They are written out
  as literals here so the oracle is deterministic without importing KINDER, and
  `test_oracle_pick_parameters_match_upstreams_own_sampler` pins them against the real
  sampler whenever the simulator is installed.
- **`ORACLE_THROW_STANDOFF = 1.35`** is upstream's own `target_distance` in the same
  test, and is the standoff every measured number in `docs/kinder-environment-validation.md`
  and `docs/tossing3d-integration-status.md` was taken at.
- **`ORACLE_RELEASE_SPEED_DEG_S = 140`** is upstream's own shipped default, aliased from
  `predicates.UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S`: what `toss_profile_limits()` returns
  when passed nothing, and the speed every committed Tossing3D number was measured at --
  including the `10/10` this oracle scores at standoff 1.35. Not a tuned value; moving it
  would be a new measurement.

- **`ORACLE_GRIPPER_RELEASE_MS = 792`** is ours, not upstream's: the midpoint of the
  `5/5` band PR #240 measured at `ORACLE_RELEASE_SPEED_DEG_S`, 763.2-821.1 ms, rounded
  to whole ms by `KinderBackend.run_toss`. Upstream's 720 is below that band.

These are **one operating point measured together**, not independently valid constants.
PR #221 measured the best standoffs over 60-83.34 deg/s at 1.050-1.075, below
`THROW_STANDOFF_BOUNDS`'s floor. Moving any one alone changes the throw.

**1.35 lands the cube at x = 1.9902, inside the bin and inside the goal box, and scores
`True`.**

> **Staleness note, 2026-08-13.** 1.9902 is left as published and is correct for the throw
> it measured, the release firing on the first control step past path fraction 0.46. Under
> the scheduled 1 kHz release the same standoff, seed and speed rest at **x = 2.0318**:
> +41.6 mm, still inside the bin and still `True`.
> `tests/environments/tossing3d/test_kinder_fidelity.py` carries both values.

> **Second staleness note, 2026-08-13.** Both numbers above were measured at 720 ms; at
> 792 ms the same rollout rests at **x = 1.9926**, still in the bin and still `True`.

That used to be a contrast: on the scene KINDER shipped before the upstream bin
fix (`kindergarden` PR #126, now carried on this repo's `reference/kindergarden` pin) the
bin sat 23 cm further out, the same standoff put the cube *in* it at x = 2.2197, and
`_check_goals()` was `False` -- landing in the bin was a scored failure. There is one
scene now and no way to select the pre-fix one, so 1.35 simply solves it; see
`Tossing3DEnvironment.backend` for why the choice was retired rather than
preserved. `Tossing3DCli` still exposes `--oracle-throw-standoff`, which is what the
band-calibration tests drive off it.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import Tossing3DEnvironment
from .predicates import (
    HOLDING,
    ROBOT_AT_SUCCESSFUL_THROW_POSE,
    UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S,
)
from .skills import Tossing3DSkills

# Upstream's own draw; see the module docstring.
ORACLE_PICK_DISTANCE = 0.5682351863248143
ORACLE_PICK_ROTATION = -0.7008563047585579

# Upstream's own `target_distance` for the throw.
ORACLE_THROW_STANDOFF = 1.35

# Upstream's own shipped release speed; see the module docstring.
ORACLE_RELEASE_SPEED_DEG_S = UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S

# Ours, not upstream's; see the module docstring.
ORACLE_GRIPPER_RELEASE_MS = 792.0

# How close the base has to get to the standoff it was *commanded* before the oracle
# treats `MoveToThrowPose` as having converged and stops re-issuing it, in metres.
#
# **This is a control-loop criterion, not a classifier threshold**, and the distinction
# matters: nothing here decides whether a pose is good, only whether moving again could
# still change it. `RobotAtSuccessfulThrowPose` remains the sole judge of the former, and
# it is upstream's (see `predicates.py`).
#
# It exists because `move_to_target` overshoots its commanded standoff by a **per-seed
# constant** -- measured 0.7-27.7 mm over 8 scene seeds -- so a commanded 1.35 arrives at
# an achieved standoff of up to 1.3777, which is past the 1.375 upper edge of upstream's
# accepted band. Before this check the oracle read the predicate as false after a
# perfectly successful move, re-issued the identical skill, and never threw: an infinite
# loop rather than a failed episode.
#
# 0.05 is the measured overshoot envelope with room to spare.
MOVE_CONVERGENCE_TOLERANCE = 0.05

# How closely the base has to be lined up on the bin's axis, and facing it, before the
# standoff check above is even consulted -- metres and radians respectively.
#
# **The standoff check is not safe on its own, and this is why.** `predicates.py`'s
# `THROW_STANDOFF_BOUNDS` comment records the hazard in the predicate's own terms: the
# post-`Pick` base sits 1.364-1.971 m from the bin, so *a standoff-only test cannot tell
# "the move ran and converged" from "the move never ran"* whenever the commanded standoff
# happens to fall in that interval. Measured directly rather than argued: at a commanded
# 1.45 on scene seed 125 the post-`Pick` standoff is 1.4912, within 0.05 of the command,
# and a standoff-only check made the oracle skip `MoveToThrowPose` entirely and toss from
# the shelf pose. That is exactly the failure `predicates.py` says the predicate's own
# upper bound exists to prevent, reintroduced one layer up.
#
# The alignment conjuncts separate the two cases with an order of magnitude to spare,
# because `MoveToThrowPose` drives the base onto the bin's axis and turns it to face the
# bin while `Pick` does neither. Measured post-`Pick` on seeds 125 and 0: lateral offset
# 0.5269 and 0.4493 m against the 0.05 m here, heading error 0.8645 and 1.1506 rad
# against the 0.05 rad here.
#
# Deliberately *not* upstream's `THROW_POSE_TOLERANCE` (0.08 for both), even though these
# are the same two quantities upstream's classifier tests. Importing it would put a KINDER
# import in a second module -- `kinder_backend.py` is the only one allowed -- and would
# also blur the line this block exists to draw: upstream's constant is where a *throw
# scores from*, and nothing here is deciding that.
MOVE_ALIGNMENT_TOLERANCE_M = 0.05
MOVE_HEADING_TOLERANCE_RAD = 0.05


class SkillOraclePolicy:
    """Picks the next skill from ground-truth state. A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def get_labeled_action(
        *,
        state: State,
        env: Tossing3DEnvironment,
        goal: Goal,
        throw_standoff: float = ORACLE_THROW_STANDOFF,
    ) -> LabeledAction:
        """The next privileged action toward the one goal this domain has.

        Goal-agnostic: unlike Tossing Room, whose state cannot distinguish a
        throw-recycling task from a throw-trash one, there is exactly one goal family
        here (`InBin(cube_0, bin_0)`) and the state says everything.
        """
        del goal
        holding = HOLDING.holds(state, (env.robot, env.cube))
        at_throw_pose = ROBOT_AT_SUCCESSFUL_THROW_POSE.holds(state, (env.robot, env.bin))

        # Where the base actually ended up, relative to the bin, against what it was told.
        # `move_to_target` overshoots by a per-seed constant, so "the move succeeded" and
        # "the predicate accepts the resulting pose" are not the same question -- see
        # `MOVE_CONVERGENCE_TOLERANCE`. The two alignment conjuncts come first because the
        # standoff alone cannot tell a converged move from a base still at the shelf; see
        # `MOVE_ALIGNMENT_TOLERANCE_M`.
        achieved_standoff = state.get(obj=env.bin, feature_name="x") - state.get(
            obj=env.robot, feature_name="pos_base_x"
        )
        lateral_offset = state.get(obj=env.bin, feature_name="y") - state.get(
            obj=env.robot, feature_name="pos_base_y"
        )
        bearing_to_bin = np.arctan2(lateral_offset, achieved_standoff)
        base_heading = state.get(obj=env.robot, feature_name="pos_base_rot")
        # Wrapped into (-pi, pi] by hand rather than imported: this file must not import
        # KINDER (see `MOVE_ALIGNMENT_TOLERANCE_M`), and the identity is standard.
        heading_error = abs(
            np.arctan2(np.sin(bearing_to_bin - base_heading), np.cos(bearing_to_bin - base_heading))
        )
        move_has_converged = (
            abs(lateral_offset) <= MOVE_ALIGNMENT_TOLERANCE_M
            and heading_error <= MOVE_HEADING_TOLERANCE_RAD
            and abs(achieved_standoff - throw_standoff) <= MOVE_CONVERGENCE_TOLERANCE
        )

        ground_skill: GroundSkill
        params: np.ndarray
        # Tossing from a converged-but-unaccepted pose rather than re-issuing the move is
        # what makes this a policy that terminates. The throw may well miss, and that is
        # the honest outcome: the episode then scores a failure, which is information,
        # where an infinite `MoveToThrowPose` loop is not.
        if holding and (at_throw_pose or move_has_converged):
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.TOSS,
                objects=(env.robot, env.cube, env.bin, env.barrier),
            )
            params = np.array([ORACLE_RELEASE_SPEED_DEG_S, ORACLE_GRIPPER_RELEASE_MS])
        elif holding:
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.MOVE_TO_THROW_POSE,
                objects=(env.robot, env.cube, env.bin),
            )
            params = np.array([throw_standoff])
        else:
            # Covers both "hand empty, cube on the ground" and the unrecoverable state
            # after a missed toss. In the latter the grasp will fail to plan and
            # `take_action` records a no-op -- there is deliberately no fallback skill,
            # because there is nothing this domain can do to recover and pretending
            # otherwise would hide the irreversibility the domain exists to exhibit.
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.PICK,
                objects=(env.robot, env.cube, env.barrier, env.bin),
            )
            params = np.array([ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION])

        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(value), 2) for value in params]}"
        return LabeledAction(action=action, label=label)
