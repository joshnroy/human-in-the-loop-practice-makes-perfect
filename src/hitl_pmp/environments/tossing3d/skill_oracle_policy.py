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

**1.35 does not solve the stock scene, and that is the point of the domain rather than a
defect in the oracle.** Under `--task-config stock` the bin sits 23 cm further out, the
cube lands *in* it at x = 2.2197, and `_check_goals()` is `False` -- landing in the bin is
a scored failure there. Under the default coincident config the same standoff lands at
x = 1.9902 and scores `True`. The oracle is tuned for the default; running it on stock is
expected to fail, and `Tossing3DCli` exposes `--oracle-throw-standoff` so the stock scene
can be driven at a standoff that does solve it (1.55, measured).
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal

from .environment import Tossing3DEnvironment
from .predicates import HoldingClassifier, RobotAtSuccessfulThrowPoseClassifier
from .skills import Tossing3DSkills

# Upstream's own draw; see the module docstring.
ORACLE_PICK_DISTANCE = 0.5682351863248143
ORACLE_PICK_ROTATION = -0.7008563047585579

# Upstream's own `target_distance` for the throw.
ORACLE_THROW_STANDOFF = 1.35


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
        here (`InGoalRegion(cube_0, blocks_goal_region)`) and the state says everything.
        """
        del goal
        holding = HoldingClassifier.holds(state=state, robot=env.robot, cube=env.cube)
        at_throw_pose = RobotAtSuccessfulThrowPoseClassifier.holds(
            state=state, robot=env.robot, target=env.bin, goal_region=env.goal_region
        )

        ground_skill: GroundSkill
        params: np.ndarray
        if holding and at_throw_pose:
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.TOSS,
                objects=(env.robot, env.cube, env.bin, env.barrier, env.goal_region),
            )
            params = np.zeros(0)
        elif holding:
            ground_skill = GroundSkill(
                skill=Tossing3DSkills.MOVE_TO_THROW_POSE,
                objects=(env.robot, env.cube, env.bin, env.goal_region),
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
                objects=(env.robot, env.cube, env.barrier, env.bin, env.goal_region),
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
