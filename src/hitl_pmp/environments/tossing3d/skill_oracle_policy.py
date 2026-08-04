from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Object, State

from .environment import Tossing3DEnvironment
from .predicates import AtThrowPoseClassifier, HandEmptyClassifier
from .skills import Tossing3DSkills


class SkillOraclePolicy:
    """Tossing3D's privileged, hand-authored solver: Pick, MoveToThrowPose, Toss, with
    the two continuous parameters set to values known to work rather than sampled.

    The swing constant is the point of this policy. A learned sampler's whole job here
    is to find the band of swing values that reaches KINDER's goal region, so measuring
    what a *known-good* swing scores is what turns "EES improved" into "EES improved
    towards, or short of, the achievable ceiling". A static-method container, never
    instantiated, same as every other business-logic class in this project.

    Goal-agnostic (unlike Tossing Room's oracle): this domain has exactly one goal, so
    the next action is a function of the state alone.
    """

    # Measured, not guessed: sweeping the swing dial over KINDER seeds 0-2 put the cube
    # in the goal region for every swing in [0.6, 0.9] and outside it (into the bin at
    # x ~ 2.22, past the region's 2.10 edge) from 1.0 up. 0.75 is the middle of the
    # band that worked on every seed tried.
    ORACLE_SWING: ClassVar[float] = 0.75
    # Mid-range of KINDER's own MOVE_TO_TARGET_DISTANCE_BOUNDS, and no rotation offset:
    # the pick pose its own sampler is centred on.
    ORACLE_PICK_DISTANCE: ClassVar[float] = 0.55
    ORACLE_PICK_ROT: ClassVar[float] = 0.0

    @staticmethod
    def get_labeled_action(*, state: State, env: Tossing3DEnvironment) -> LabeledAction:
        if HandEmptyClassifier.holds(state=state, robot=env.robot):
            # Nothing in hand: pick. If the cube is already beyond the barrier the task
            # is lost and no skill recovers it, so the oracle keeps issuing the pick and
            # the episode runs out its horizon -- the honest outcome rather than a
            # pretend one.
            return SkillOraclePolicy._labeled(
                skill=Tossing3DSkills.PICK,
                objects=(env.robot, env.cube, env.barrier, env.bin_object),
                params=np.array([
                    SkillOraclePolicy.ORACLE_PICK_DISTANCE,
                    SkillOraclePolicy.ORACLE_PICK_ROT,
                ]),
                state=state,
            )
        if not AtThrowPoseClassifier.holds(state=state, robot=env.robot, bin_object=env.bin_object):
            return SkillOraclePolicy._labeled(
                skill=Tossing3DSkills.MOVE_TO_THROW_POSE,
                objects=(env.robot, env.bin_object),
                params=np.empty(0),
                state=state,
            )
        return SkillOraclePolicy._labeled(
            skill=Tossing3DSkills.TOSS,
            objects=(env.robot, env.cube, env.bin_object, env.goal_region, env.barrier),
            params=np.array([SkillOraclePolicy.ORACLE_SWING]),
            state=state,
        )

    @staticmethod
    def _labeled(
        *, skill: Skill, objects: tuple[Object, ...], params: np.ndarray, state: State
    ) -> LabeledAction:
        ground_skill = GroundSkill(skill=skill, objects=objects)
        return LabeledAction(
            action=Tossing3DSkills.compute_action(
                ground_skill=ground_skill, params=params, state=state
            ),
            label=skill.name,
        )
