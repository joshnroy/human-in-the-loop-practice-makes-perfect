"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import (
    ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
    OraclePolicyProvider,
    SkillProvider,
)
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import Tossing3DEnvironment
from .layout import Tossing3DLayout
from .predicates import (
    BIN_AT_SIDE,
    CUBE_AT_SIDE,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    ROBOT_AT_SIDE,
)
from .recovery_skills import CLOSED_EMPTY, ON_BIN_RIM, ON_FLOOR, SameSideSkills
from .sides import Tossing3DSide, Tossing3DSides
from .skill_oracle_policy import ORACLE_THROW_STANDOFF, SkillOraclePolicy
from .skills import Tossing3DSkills


class Tossing3DSkillProvider(SkillProvider):
    """Tossing3D's `SkillProvider`, mirroring `TossingRoomSkillProvider`.

    `objects()` is a fixed six: upstream's task JSON names exactly one cube, one bin and
    one barrier, plus the robot; two featureless symbolic side objects let STRIPS actions
    bind a reset destination. There is no configuration that changes the physical cast --
    `o2` would add a second cube, and this domain does not support it (see the README).

    The goal region is not a symbolic object. Its scored box remains in the `State`,
    carried on the bin (see `predicates.py`'s module docstring); it is not something a
    planner binds a variable to because no skill can act on it.
    """

    env: Tossing3DEnvironment
    human_reset_practice_cost: float = Field(default=5.0, ge=0.0, allow_inf_nan=False)

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
                NOT_HOLDING,
                CLOSED_EMPTY,
                ON_BIN_RIM,
                ROBOT_AT_SIDE,
                CUBE_AT_SIDE,
                BIN_AT_SIDE,
            )
        return (
            IN_BIN,
            HAND_EMPTY,
            HOLDING,
            NOT_HOLDING,
            ON_GROUND,
            ROBOT_AT_SIDE,
            CUBE_AT_SIDE,
            BIN_AT_SIDE,
        )

    def types(self) -> tuple[Type, ...]:
        return (
            Tossing3DEnvironment.robot_type,
            Tossing3DEnvironment.cube_type,
            Tossing3DEnvironment.bin_type,
            Tossing3DEnvironment.barrier_type,
            Tossing3DSides.type,
        )

    def objects(self) -> tuple[Object, ...]:
        env = self.env
        return (env.robot, env.cube, env.bin, env.barrier, *Tossing3DSides.objects())

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
        robot, bin_, *_ = ground_skill.objects
        dx = state.get(obj=bin_, feature_name="x") - state.get(obj=robot, feature_name="pos_base_x")
        dy = state.get(obj=bin_, feature_name="y") - state.get(obj=robot, feature_name="pos_base_y")
        yaw = state.get(obj=robot, feature_name="pos_base_rot")
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        return [1.0, forward, lateral, *(float(param) for param in params)]

    def human_cube_bin_reset_skill(self) -> GroundSkill:
        """Tossing3D's `ask_for_reset_cube_bin_only`: repositions `cube_0`/`bin_0`
        to fresh ground poses via `KinderBackend.reset_cube_and_bin`, robot
        untouched. Effects: `OnGround` (or same-side `OnFloor`) and the selected
        typed side facts become true, `InBin` becomes false; same-side `OnBinRim`
        is cleared too. ``HandEmpty`` stays as it was.

        The static precondition binds the robot-relative side object. The singular API
        preserves each layout's historical bin destination; planners use the plural
        API below to choose either destination."""
        resets = self.human_cube_bin_reset_skills()
        historical_destination = (
            Tossing3DSide.ROBOT.value
            if self.env.layout == Tossing3DLayout.SAME_SIDE
            else Tossing3DSide.OPPOSITE.value
        )
        return next(reset for reset in resets if reset.objects[-1].name == historical_destination)

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """One lifted reset, grounded once for each typed bin destination.

        The cube is always returned to the robot's side so practice can continue;
        the final parameter is the bin side selected by the planner. Functional side
        predicates are cleared through ``ignore_effects`` before the two selected
        atoms are added. This is the one lifted STRIPS action; the two choices are
        ordinary ground actions, not separately named skills.
        """
        env = self.env
        robot = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
        cube = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
        bin_ = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
        barrier = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
        robot_side = Variable(name="robot_side", type=Tossing3DSides.type)
        bin_destination = Variable(name="bin_destination", type=Tossing3DSides.type)
        floor = (
            LiftedAtom(predicate=ON_FLOOR, variables=(cube, bin_))
            if env.layout == Tossing3DLayout.SAME_SIDE
            else LiftedAtom(predicate=ON_GROUND, variables=(cube,))
        )
        removed = {LiftedAtom(predicate=IN_BIN, variables=(cube, bin_))}
        if env.layout == Tossing3DLayout.SAME_SIDE:
            removed.add(LiftedAtom(predicate=ON_BIN_RIM, variables=(cube, bin_)))
        skill = Skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            parameters=(
                robot,
                cube,
                bin_,
                barrier,
                robot_side,
                bin_destination,
            ),
            preconditions=frozenset({
                LiftedAtom(
                    predicate=ROBOT_AT_SIDE,
                    variables=(robot, barrier, robot_side),
                ),
                LiftedAtom(predicate=NOT_HOLDING, variables=(robot, cube)),
            }),
            add_effects=frozenset({
                floor,
                LiftedAtom(predicate=NOT_HOLDING, variables=(robot, cube)),
                LiftedAtom(
                    predicate=CUBE_AT_SIDE,
                    variables=(cube, barrier, robot_side),
                ),
                LiftedAtom(
                    predicate=BIN_AT_SIDE,
                    variables=(bin_, barrier, bin_destination),
                ),
            }),
            delete_effects=frozenset(removed),
            ignore_effects=frozenset({CUBE_AT_SIDE, BIN_AT_SIDE}),
            param_dim=0,
            practice_cost=self.human_reset_practice_cost,
        )
        return tuple(
            GroundSkill(
                skill=skill,
                objects=(
                    env.robot,
                    env.cube,
                    env.bin,
                    env.barrier,
                    Tossing3DSides.robot,
                    side,
                ),
            )
            for side in Tossing3DSides.objects()
        )

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        if ground_skill.skill.name != ASK_FOR_RESET_CUBE_BIN_ONLY_NAME:
            return None
        destination = ground_skill.objects[-1]
        Tossing3DSides.parse(name=destination.name)
        return destination.name

    def movables_reset_skills(self) -> tuple[GroundSkill, ...]:
        return self.human_cube_bin_reset_skills()


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
