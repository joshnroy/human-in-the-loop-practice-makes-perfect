"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import (
    ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
    OraclePolicyProvider,
    SkillProvider,
)
from hitl_pmp.core.method.types import (
    ConditionalAddEffect,
    GroundSkill,
    LabeledAction,
    LiftedAtom,
    Skill,
    Variable,
)
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import Tossing3DEnvironment
from .layout import Tossing3DLayout
from .parameter_feasibility import TossParameterFeasibility
from .predicates import (
    CLOSED_EMPTY,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    REACHABLE,
)
from .recovery_skills import ON_BIN_RIM, ON_FLOOR, SameSideSkills
from .skill_oracle_policy import ORACLE_THROW_STANDOFF, SkillOraclePolicy
from .skills import Tossing3DSkills


class Tossing3DSkillProvider(SkillProvider):
    """Tossing3D's `SkillProvider`, mirroring `TossingRoomSkillProvider`.

    `objects()` is a fixed four: upstream's task JSON names exactly one cube, one bin and
    one barrier, plus the robot. There is no configuration that changes the cast -- `o2`
    would add a second cube, and this domain does not support it (see the README).

    It was five until the goal region stopped being a symbolic object. The scored box is
    still in the `State`, carried on the bin (see `predicates.py`'s module docstring); it
    is simply not something a planner binds a variable to, because no skill can act on it.
    """

    env: Tossing3DEnvironment
    human_reset_practice_cost: float = Field(default=5.0, ge=0.0, allow_inf_nan=False)
    non_human_reset_practice_cost: float = Field(default=5.0, ge=0.0, allow_inf_nan=False)

    def skills(self) -> tuple[Skill, ...]:
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.skills()
        return (
            Tossing3DSkills.PICK_CUBE,
            Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            # Robot-executed, unlike `human_cube_bin_reset_skill` -- always offered to the
            # planner (unconditionally, not gated behind `plan_to`'s `practicing`), since
            # it is a free real action rather than a costed human intervention. See its
            # docstring in skills.py for why it exists.
            Tossing3DSkills.OPEN_GRIPPER,
        )

    def predicates(self) -> tuple[Predicate, ...]:
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return (
                IN_BIN,
                HAND_EMPTY,
                HOLDING,
                ON_GROUND,
                ON_FLOOR,
                REACHABLE,
                CLOSED_EMPTY,
                ON_BIN_RIM,
            )
        return (CLOSED_EMPTY, IN_BIN, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE)

    def types(self) -> tuple[Type, ...]:
        return (
            Tossing3DEnvironment.robot_type,
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.bin_type,
            Tossing3DEnvironment.barrier_type,
        )

    def objects(self) -> tuple[Object, ...]:
        env = self.env
        return (env.robot, env.cube, env.bin, env.barrier)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.sample_params(ground_skill=ground_skill, rng=rng)
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.compute_action(
                ground_skill=ground_skill, params=params, state=state
            )
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)

    def parameter_rejection_reason(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> str | None:
        return TossParameterFeasibility.rejection_reason(
            ground_skill=ground_skill, params=params, state=state
        )

    def hand_selected_feature_transform(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """Describe a toss by robot-frame bin displacement and controller parameters.

        The parameters are already relative to the bin: standoff, yaw offset, joint
        speed, and release time. Expressing the observed displacement in the robot frame
        makes the full sampler row invariant to rigid changes in scene pose.
        """
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return None
        robot, bin_, _, _ = ground_skill.objects
        dx = state.get(obj=bin_, feature_name="x") - state.get(obj=robot, feature_name="pos_base_x")
        dy = state.get(obj=bin_, feature_name="y") - state.get(obj=robot, feature_name="pos_base_y")
        yaw = state.get(obj=robot, feature_name="pos_base_rot")
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        return [1.0, forward, lateral, *(float(param) for param in params)]

    def human_cube_bin_reset_skill(self) -> GroundSkill:
        """Relocate the cube and bin, preserving the robot's gripper command.

        Reset remains callable from any state, including failed grasps. Relocating
        a held cube deletes Holding and produces ClosedEmpty; an already-open
        gripper remains HandEmpty. The conditional addition is evaluated before
        Holding is deleted, in both classical and belief-space planning.
        """
        env = self.env
        robot = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
        cube = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
        bin_ = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
        barrier = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
        floor = (
            LiftedAtom(predicate=ON_FLOOR, variables=(cube, bin_))
            if env.layout == Tossing3DLayout.SAME_SIDE
            else LiftedAtom(predicate=ON_GROUND, variables=(cube,))
        )
        holding = LiftedAtom(predicate=HOLDING, variables=(robot, cube))
        removed = {LiftedAtom(predicate=IN_BIN, variables=(cube, bin_)), holding}
        if env.layout == Tossing3DLayout.SAME_SIDE:
            removed.add(LiftedAtom(predicate=ON_BIN_RIM, variables=(cube, bin_)))
        skill = Skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            parameters=(robot, cube, bin_, barrier),
            preconditions=frozenset(),
            add_effects=frozenset({
                floor,
                LiftedAtom(predicate=REACHABLE, variables=(cube, barrier)),
            }),
            delete_effects=frozenset(removed),
            conditional_add_effects=frozenset({
                ConditionalAddEffect(
                    conditions=frozenset({holding}),
                    add_effects=frozenset({
                        LiftedAtom(predicate=CLOSED_EMPTY, variables=(robot, cube))
                    }),
                )
            }),
            param_dim=0,
            practice_cost=self.human_reset_practice_cost,
        )
        return GroundSkill(skill=skill, objects=(env.robot, env.cube, env.bin, env.barrier))

    def non_human_cube_bin_reset_skill(self) -> GroundSkill:
        """Automatic reset with the same mechanics as the human reset.

        Its distinct identity gives the belief-space model an independent joint
        competence/learning-rate/cost posterior while preserving identical symbolic
        applicability and outcomes.
        """
        human_reset = self.human_cube_bin_reset_skill()
        return human_reset.model_copy(
            update={
                "skill": human_reset.skill.model_copy(
                    update={
                        "name": "non_human_reset_cube_bin_only",
                        "practice_cost": self.non_human_reset_practice_cost,
                    }
                )
            }
        )

    def movables_reset_skills(self) -> tuple[GroundSkill, ...]:
        return (self.human_cube_bin_reset_skill(), self.non_human_cube_bin_reset_skill())


class Tossing3DOracle(OraclePolicyProvider):
    """Tossing3D's privileged solver, driving `SkillOracleMethod`.

    Goal-agnostic (one goal family; see `skill_oracle_policy.py`). `throw_standoff` is a
    constructor field rather than a constant read off the policy module because the
    standoff that solves depends on which scene is loaded -- 1.35 on the coincident
    config, 1.55 on stock -- and the CLI has to be able to say which.
    """

    env: Tossing3DEnvironment
    throw_standoff: float = ORACLE_THROW_STANDOFF

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(
            state=state, env=self.env, goal=goal, throw_standoff=self.throw_standoff
        )
