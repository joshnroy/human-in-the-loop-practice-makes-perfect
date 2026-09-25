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
from .predicates import (
    BIN_AT_SIDE,
    CLOSED_EMPTY,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    PICKUP_UNBLOCKED,
    ROBOT_AT_SIDE,
)
from .recovery_skills import ON_BIN_RIM, ON_FLOOR, SameSideSkills
from .sides import Tossing3DSide, Tossing3DSides
from .skill_oracle_policy import ORACLE_THROW_STANDOFF, SkillOraclePolicy
from .skills import Tossing3DSkills
from .toss import Tossing3DToss


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
                NOT_HOLDING,
                CLOSED_EMPTY,
                ON_BIN_RIM,
                ROBOT_AT_SIDE,
                CUBE_AT_SIDE,
                BIN_AT_SIDE,
                PICKUP_UNBLOCKED,
                # Declared here for the same reason the side atoms are: the shared
                # toss operator forgets GraspClear (ignore_effects), and a forgetting
                # effect must name a declared predicate in every domain that writes
                # the operator. No same-side skill conditions on it.
                GRASP_CLEAR,
            )
        return (
            CLOSED_EMPTY,
            IN_BIN,
            HAND_EMPTY,
            HOLDING,
            NOT_HOLDING,
            ON_GROUND,
            ROBOT_AT_SIDE,
            CUBE_AT_SIDE,
            BIN_AT_SIDE,
            GRASP_CLEAR,
            PICKUP_UNBLOCKED,
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

    # The toss is routed to `Tossing3DToss` BEFORE any layout branch, so both layouts
    # reach the identical callables; only the parameterless skills differ by layout.

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return Tossing3DToss.sample_params(rng=rng)
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.sample_params(ground_skill=ground_skill, rng=rng)
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def sample_params_at_state(
        self, *, ground_skill: GroundSkill, rng: np.random.Generator, state: State
    ) -> np.ndarray:
        if ground_skill.skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return Tossing3DToss.sample_params_at_state(
                rng=rng, ground_skill=ground_skill, state=state
            )
        return self.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        if ground_skill.skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return Tossing3DToss.compute_action(params=params, state=state)
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.compute_action(
                ground_skill=ground_skill, params=params, state=state
            )
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)

    def parameter_rejection_reason(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> str | None:
        """Raises `NoFeasibleTossDirectionError` for a standoff no direction can plan."""
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return None
        return Tossing3DToss.rejection_reason(state=state, params=params)

    def hand_selected_feature_transform(
        self, *, ground_skill: GroundSkill, state: State, params: np.ndarray
    ) -> list[float] | None:
        """`[1, standoff, speed, release]` -- see `toss.py` for why nothing else."""
        del state
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return None
        return Tossing3DToss.feature_row(params=params)

    def action_annotations(self, *, ground_skill: GroundSkill, action: Action) -> dict[str, float]:
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return {}
        return Tossing3DToss.action_annotations(action=action)

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
        ordinary ground actions, not separately named skills. Holding is deleted
        when a reset relocates the cube; a closed gripper is preserved through
        the existing conditional ClosedEmpty effect.
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
        holding = LiftedAtom(predicate=HOLDING, variables=(robot, cube))
        removed = {LiftedAtom(predicate=IN_BIN, variables=(cube, bin_)), holding}
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
            }),
            add_effects=frozenset({
                floor,
                LiftedAtom(predicate=NOT_HOLDING, variables=(robot, cube)),
                # The reset's cube region (blocks_init_region, x in [0.5, 0.75]) sits
                # >= 1.7 m from either bin destination region, so a reset cube clears
                # the walls by construction -- this add effect is what lets a plan
                # recover from a not-GraspClear landing by paying for the reset.
                LiftedAtom(predicate=GRASP_CLEAR, variables=(cube, bin_)),
                # Relocating the cube physically heals every observed refusal
                # instance, so the reset may promise the channel clear.
                LiftedAtom(predicate=PICKUP_UNBLOCKED, variables=(cube,)),
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
            conditional_add_effects=frozenset({
                ConditionalAddEffect(
                    conditions=frozenset({holding}),
                    add_effects=frozenset({
                        LiftedAtom(predicate=CLOSED_EMPTY, variables=(robot, cube))
                    }),
                )
            }),
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

    def non_human_cube_bin_reset_skill(self) -> GroundSkill:
        """The historical destination with its independent automatic-reset cost."""
        human = self.human_cube_bin_reset_skill()
        return self._automatic_reset(reset=human)

    def non_human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """Automatic reset destinations share one belief, distinct from human reset."""
        return tuple(
            self._automatic_reset(reset=reset) for reset in self.human_cube_bin_reset_skills()
        )

    def _automatic_reset(self, *, reset: GroundSkill) -> GroundSkill:
        return reset.model_copy(
            update={
                "skill": reset.skill.model_copy(
                    update={
                        "name": "non_human_reset_cube_bin_only",
                        "practice_cost": self.non_human_reset_practice_cost,
                    }
                )
            }
        )

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        if ground_skill.skill.name not in {
            ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            "non_human_reset_cube_bin_only",
        }:
            return None
        destination = ground_skill.objects[-1]
        Tossing3DSides.parse(name=destination.name)
        return destination.name

    def movables_reset_skills(self) -> tuple[GroundSkill, ...]:
        return (*self.human_cube_bin_reset_skills(), *self.non_human_cube_bin_reset_skills())


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
