"""Tossing3D's privileged solver: the two-skill sequence and its parameters.

The whole policy is a two-way branch on the symbolic state -- pick the cube up, then
drive-and-throw -- because the domain admits exactly one plan shape. What makes it an
*oracle* rather than a fixed script is the continuous parameters of the second skill,
which are the part a learner would have to find. The first skill has none at all:
`pick_cube` derives its standoff and its grasp rotation internally, so there is nothing
for an oracle to know that a random draw would not also get.

**That is a change from the three-skill decomposition, and it narrows this oracle.** It
used to supply `ORACLE_PICK_DISTANCE`/`ORACLE_PICK_ROTATION` -- the pair upstream's own
`PickShelfController.sample_parameters` drew from `np.random.default_rng(123)` in
`test_pick_ground_toss` -- and a separate `MoveToThrowPose` standoff. Both are gone: the
pick takes no parameters and the standoff is now the composed toss's first parameter.

The four the oracle does supply:

- **`ORACLE_THROW_STANDOFF = 1.35`** is upstream's own `target_distance` in
  `test_pick_ground_toss`, and is the standoff every measured number in
  `docs/kinder-environment-validation.md` and `docs/tossing3d-integration-status.md` was
  taken at. It lies inside upstream's own `TOSS_DISTANCE_BOUNDS` of `(1.25, 1.45)`.
- **`ORACLE_THROW_ROTATION = 0.0`** -- head-on. Upstream's own value in the same test,
  and the centre of `TOSS_ROTATION_BOUNDS`.
- **`ORACLE_RELEASE_SPEED_DEG_S = 140`** is upstream's own shipped default: what
  `toss_profile_limits()` returns when passed nothing, and the speed every committed
  Tossing3D number was measured at. Not a tuned value; moving it would be a new
  measurement.
- **`ORACLE_GRIPPER_RELEASE_MS = 792`** is ours, not upstream's: the midpoint of the
  `5/5` band PR #240 measured at `ORACLE_RELEASE_SPEED_DEG_S`, 763.2-821.1 ms. Upstream's
  own 720 is below that band, though inside `TOSS_RELEASE_MS_BOUNDS`.

> **Staleness note, and it applies to every number in this docstring.** All four were
> measured against the **three-skill** decomposition, where the base drove to the standoff
> under `move_to_target` and the swing then ran from a separately-commanded windup. The
> composed controller plans base motion, windup and swing together in one `reset`. The
> geometry is meant to be the same and the oracle is re-measured end to end by
> `test_kinder_fidelity.py`, but no number below is evidence about the composed
> controller until that test has run against it. Nothing here is recomputed; the earlier
> values stand as published.

These are **one operating point measured together**, not independently valid constants.
PR #221 measured the best standoffs over 60-83.34 deg/s at 1.050-1.075, below the
distance bounds upstream now samples from. Moving any one alone changes the throw.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import Tossing3DEnvironment
from .predicates import HoldingClassifier
from .skills import Tossing3DSkills

# Upstream's own `target_distance` for the throw.
ORACLE_THROW_STANDOFF = 1.35

# Head-on, upstream's own value.
ORACLE_THROW_ROTATION = 0.0

# Upstream's own shipped release speed; see the module docstring.
ORACLE_RELEASE_SPEED_DEG_S = 140.0

# Ours, not upstream's; see the module docstring.
ORACLE_GRIPPER_RELEASE_MS = 792.0


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

        The branch is on `Holding` alone now. It used to be on `Holding` *and*
        `RobotAtSuccessfulThrowPose`, which chose between walking to the throw pose and
        throwing from it; there is one composed skill for both, so there is nothing left
        to choose between.
        """
        del goal
        holding = HoldingClassifier.holds(state=state, robot=env.robot, cube=env.cube)

        ground_skill: GroundSkill
        params: np.ndarray
        if holding:
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
                objects=(env.robot, env.bin, env.cube, env.barrier),
            )
            params = np.array([
                throw_standoff,
                ORACLE_THROW_ROTATION,
                ORACLE_RELEASE_SPEED_DEG_S,
                ORACLE_GRIPPER_RELEASE_MS,
            ])
        else:
            # Covers both "hand empty, cube on the ground" and the unrecoverable state
            # after a missed toss. In the latter the grasp will fail to plan and
            # `take_action` records a no-op -- there is deliberately no fallback skill,
            # because there is nothing this domain can do to recover and pretending
            # otherwise would hide the irreversibility the domain exists to exhibit.
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.PICK_CUBE,
                objects=(env.robot, env.cube, env.barrier),
            )
            params = np.zeros(0)

        action = Tossing3DSkills.compute_action(
            ground_skill=ground_skill, params=params, state=state
        )
        objects_desc = ", ".join(obj.name for obj in ground_skill.objects)
        label = f"{ground_skill.skill.name}({objects_desc})"
        if params.size > 0:
            label += f", params={[round(float(value), 2) for value in params]}"
        return LabeledAction(action=action, label=label)
