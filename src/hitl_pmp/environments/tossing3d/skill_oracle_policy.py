"""Tossing3D's privileged solver: the three-skill sequence, with upstream's parameters.

The whole policy is a three-way branch on the symbolic state -- pick, walk to the throw
pose, throw -- because the domain admits exactly one plan shape. What makes it an
*oracle* rather than a fixed script is the continuous parameters, which are the part a
learner would have to find, and every one of them here is a value upstream itself
publishes:

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
  `predicates.UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S` rather than written out again. It is
  the literal that sat inline in `TossController.reset` until `joshnroy/kinder-baselines`
  PR #8 made it a parameter, it is what `toss_profile_limits()` still returns when passed
  nothing, and it is the speed **every committed Tossing3D number was measured at** --
  including the `10/10` this oracle scores at standoff 1.35 on the coincident config.

  It is emphatically **not a tuned value**, and the distinction matters for what this
  oracle is evidence about. Naming it as the oracle's own constant is what keeps the
  oracle's throw byte-identical to the throw it made before `Toss` had a dial at all: the
  dial's arrival changes what a *learner* can do, and changes nothing about the baseline
  it is measured against. Moving this number would be a new measurement, not a tweak.

  Note that `ORACLE_THROW_STANDOFF = 1.35` is only known to solve **at this speed**. PR
  #221 measured the best standoffs over 60-83.34 deg/s at 1.050-1.075, *below*
  `THROW_STANDOFF_BOUNDS`'s floor, so the two constants are not independently valid --
  they are one operating point that was measured together.

- **`ORACLE_GRIPPER_RELEASE_MS = 723`** is upstream's own shipped default for the second
  dial, aliased from `predicates.UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS` for the same reason.
  723 ms is the millisecond the retired `_release_fraction = 0.46` trigger fell at for the
  shipped windup->release path **at 140 deg/s** -- which is the speed above it, so the
  pair reproduces the throw every earlier result was measured against rather than merely
  being near it.

  The same operating-point caveat applies, and more sharply here: 723 ms is fraction 0.461
  of the swing at 140 deg/s but 0.197 at 60 and 0.732 at 240, because the swing's duration
  is a function of the speed. So this constant is only the canonical release **paired with
  `ORACLE_RELEASE_SPEED_DEG_S`**; it is not a good release millisecond in general, and
  moving either one alone changes the throw.

**1.35 lands the cube at x = 1.9902, inside the bin and inside the goal box, and scores
`True`.** That used to be a contrast: on the scene KINDER shipped before the upstream bin
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
    UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS,
    UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S,
    HoldingClassifier,
    RobotAtSuccessfulThrowPoseClassifier,
)
from .skills import Tossing3DSkills

# Upstream's own draw; see the module docstring.
ORACLE_PICK_DISTANCE = 0.5682351863248143
ORACLE_PICK_ROTATION = -0.7008563047585579

# Upstream's own `target_distance` for the throw.
ORACLE_THROW_STANDOFF = 1.35

# Upstream's own shipped release speed; see the module docstring.
ORACLE_RELEASE_SPEED_DEG_S = UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S

# Upstream's own shipped default release millisecond; see the module docstring.
ORACLE_GRIPPER_RELEASE_MS = UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS


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
        holding = HoldingClassifier.holds(state=state, robot=env.robot, cube=env.cube)
        at_throw_pose = RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state, robot=env.robot, target=env.bin
        )

        ground_skill: GroundSkill
        params: np.ndarray
        if holding and at_throw_pose:
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
