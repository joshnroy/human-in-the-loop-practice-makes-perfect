"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
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
    BIN_ON_GROUND,
    BIN_ON_OPPOSITE_SIDE,
    BIN_ON_ROBOT_SIDE,
    CLOSED_EMPTY,
    CUBE_ON_OPPOSITE_SIDE,
    CUBE_ON_ROBOT_SIDE,
    GRASP_CLEAR,
    GRIPPER_CLOSED,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    PICK_PLANNABLE,
    PICKUP_UNBLOCKED,
)
from .recovery_skills import ON_BIN_RIM, ON_FLOOR, SameSideSkills
from .sides import Tossing3DSide
from .skill_oracle_policy import ORACLE_THROW_STANDOFF, SkillOraclePolicy
from .skills import CUBE_SIDES, Tossing3DSkills
from .toss import Tossing3DToss

RESET_CUBE_BIN_SAME_SIDE_NAME = "ResetCubeBinSameSide"
RESET_CUBE_BIN_OPPOSITE_SIDE_NAME = "ResetCubeBinOppositeSide"
# Off by default. Both automatic destinations keep the one historical name, so nothing
# keyed on it changes for that arm.
NON_HUMAN_RESET_NAME = "non_human_reset_cube_bin_only"
BIN_SIDES = frozenset({BIN_ON_ROBOT_SIDE, BIN_ON_OPPOSITE_SIDE})


class Tossing3DSkillProvider(SkillProvider):
    """Tossing3D's `SkillProvider`, mirroring `TossingRoomSkillProvider`.

    `objects()` is a fixed four: upstream's task JSON names exactly one cube, one bin and
    one barrier, plus the robot. Sides are robot-relative predicates, not objects, and
    the two reset destinations are two skills. There is no configuration that changes
    the physical cast -- `o2` would add a second cube, and this domain does not support
    it (see the README).

    The goal region is not a symbolic object. Its scored box remains in the `State`,
    carried on the bin (see `predicates.py`'s module docstring); it is not something a
    planner binds a variable to because no skill can act on it.
    """

    env: Tossing3DEnvironment
    # Off by default: the automatic reset is the human reset relabelled -- it raises the
    # same request, the same oracle executes it and it counts as a human intervention --
    # so offering both leaves the planner choosing between two cost posteriors for one
    # mechanism, which is noise rather than a decision.
    offer_non_human_reset: bool = False
    # Off only for the "is the human reset needed?" baseline. Absent, not expensive: no
    # finite reset price removes the skill, because a planner that is otherwise stuck
    # will take it at any price (see `EesMethod.plan_to`).
    offer_human_reset: bool = True

    def skills(self) -> tuple[Skill, ...]:
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return tuple(
                skill
                for skill in SameSideSkills.skills()
                if (self.env.supports_bin_pouring and self.env.enable_bin_pouring)
                or skill != SameSideSkills.POUR_BIN
            )
        return (
            Tossing3DSkills.PICK_CUBE,
            Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
            # Robot-executed, unlike `human_cube_bin_reset_skill` -- always offered to the
            # planner (unconditionally, not gated behind `plan_to`'s `practicing`), since
            # it is a free real action rather than a costed human intervention. See its
            # docstring in skills.py for why it exists.
            Tossing3DSkills.OPEN_GRIPPER,
        ) + (
            (Tossing3DSkills.POUR_BIN,)
            if self.env.supports_bin_pouring and self.env.enable_bin_pouring
            else ()
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
                GRIPPER_CLOSED,
                ON_BIN_RIM,
                CUBE_ON_ROBOT_SIDE,
                CUBE_ON_OPPOSITE_SIDE,
                BIN_ON_ROBOT_SIDE,
                BIN_ON_OPPOSITE_SIDE,
                PICKUP_UNBLOCKED,
                BIN_ON_GROUND,
                PICK_PLANNABLE,
                # Declared here for the same reason the side atoms are: the shared
                # toss operator forgets GraspClear (ignore_effects), and a forgetting
                # effect must name a declared predicate in every domain that writes
                # the operator. No same-side skill conditions on it.
                GRASP_CLEAR,
            )
        return (
            CLOSED_EMPTY,
            GRIPPER_CLOSED,
            IN_BIN,
            HAND_EMPTY,
            HOLDING,
            NOT_HOLDING,
            ON_GROUND,
            CUBE_ON_ROBOT_SIDE,
            CUBE_ON_OPPOSITE_SIDE,
            BIN_ON_ROBOT_SIDE,
            BIN_ON_OPPOSITE_SIDE,
            # Observed state only since PickPlannable replaced it in the pick: no
            # skill conditions on it, the toss still forgets it and the resets still
            # add it.
            GRASP_CLEAR,
            PICKUP_UNBLOCKED,
            BIN_ON_GROUND,
            PICK_PLANNABLE,
        )

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

    # The toss is routed to `Tossing3DToss` BEFORE any layout branch, so both layouts
    # reach the identical callables; only the parameterless skills differ by layout.

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return Tossing3DToss.sample_params(
                rng=rng, parameters=Tossing3DToss.parameters_of(env=self.env)
            )
        if self.env.layout == Tossing3DLayout.SAME_SIDE:
            return SameSideSkills.sample_params(ground_skill=ground_skill, rng=rng)
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def sample_params_at_state(
        self, *, ground_skill: GroundSkill, rng: np.random.Generator, state: State
    ) -> np.ndarray:
        if ground_skill.skill == Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return Tossing3DToss.sample_params_at_state(
                rng=rng,
                ground_skill=ground_skill,
                state=state,
                parameters=Tossing3DToss.parameters_of(env=self.env),
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
        """Rejects a standoff no direction can plan, rather than raising."""
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return None
        return Tossing3DToss.rejection_reason(
            ground_skill=ground_skill,
            state=state,
            params=params,
            parameters=Tossing3DToss.parameters_of(env=self.env),
        )

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

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        """Tossing3D's same-side human reset: repositions `cube_0`/`bin_0` to fresh
        ground poses via `KinderBackend.reset_cube_and_bin`, robot untouched.
        Effects: `OnGround` (or same-side `OnFloor`) and the destination's side facts
        become true, `InBin` becomes false; same-side `OnBinRim` is cleared too.
        ``HandEmpty`` stays as it was.

        Planners use the plural API below, which offers both destinations as two
        skills and lets the planner choose. `None` when `offer_human_reset` is off."""
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """The reset skills practice may choose, or none when `offer_human_reset`
        is off."""
        return self._cube_bin_reset_groundings() if self.offer_human_reset else ()

    def _cube_bin_reset_groundings(self) -> tuple[GroundSkill, ...]:
        """Two human reset skills, one per bin destination, same side first.

        The cube is always returned to the robot's side so practice can continue;
        the skill names where the bin goes. The costs file prices both alike,
        so nothing here prefers a side: the choice is the planner's. The functional
        side predicates are cleared through ``ignore_effects`` before the
        destination's atoms are added. Holding is deleted when a reset relocates
        the cube; a closed gripper is preserved through the conditional ClosedEmpty
        effect.
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
        removed = {
            LiftedAtom(predicate=IN_BIN, variables=(cube, bin_)),
            holding,
        }
        if env.layout == Tossing3DLayout.SAME_SIDE:
            removed.add(LiftedAtom(predicate=ON_BIN_RIM, variables=(cube, bin_)))
        common = frozenset({
            floor,
            LiftedAtom(predicate=NOT_HOLDING, variables=(robot, cube)),
            # The reset's cube region (blocks_init_region, x in [0.5, 0.75]) sits
            # >= 1.7 m from either bin destination region, so a reset cube clears
            # the walls by construction.
            LiftedAtom(predicate=GRASP_CLEAR, variables=(cube, bin_)),
            # The reset puts the cube in its spawn strip, in the open and on the
            # robot's side. This add effect is what lets a plan recover from a
            # cube the pick controller cannot plan for by paying for the reset.
            LiftedAtom(predicate=PICK_PLANNABLE, variables=(robot, cube)),
            # Relocating the cube physically heals every observed refusal
            # instance, so the reset may promise the channel clear.
            LiftedAtom(predicate=PICKUP_UNBLOCKED, variables=(cube,)),
            # The reset re-places the bin upright on the floor, which is what
            # lets a plan recover from a tipped bin by paying for it.
            LiftedAtom(predicate=BIN_ON_GROUND, variables=(bin_,)),
            LiftedAtom(predicate=CUBE_ON_ROBOT_SIDE, variables=(cube, barrier)),
        })
        return tuple(
            GroundSkill(
                skill=Skill(
                    name=name,
                    parameters=(robot, cube, bin_, barrier),
                    preconditions=frozenset(),
                    add_effects=common
                    | {LiftedAtom(predicate=bin_side, variables=(bin_, barrier))},
                    delete_effects=frozenset(removed),
                    conditional_add_effects=frozenset({
                        ConditionalAddEffect(
                            conditions=frozenset({holding}),
                            add_effects=frozenset({
                                LiftedAtom(predicate=CLOSED_EMPTY, variables=(robot, cube)),
                                LiftedAtom(predicate=GRIPPER_CLOSED, variables=(robot,)),
                            }),
                        )
                    }),
                    ignore_effects=CUBE_SIDES | BIN_SIDES,
                    param_dim=0,
                ),
                objects=(env.robot, env.cube, env.bin, env.barrier),
            )
            for name, bin_side in (
                (RESET_CUBE_BIN_SAME_SIDE_NAME, BIN_ON_ROBOT_SIDE),
                (RESET_CUBE_BIN_OPPOSITE_SIDE_NAME, BIN_ON_OPPOSITE_SIDE),
            )
        )

    def non_human_cube_bin_reset_skill(self) -> GroundSkill:
        """The historical destination of the automatic reset."""
        return self.non_human_cube_bin_reset_skills()[0]

    def non_human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        """The automatic counterparts of both human resets, under their own name."""
        return tuple(
            self._automatic_reset(reset=reset) for reset in self._cube_bin_reset_groundings()
        )

    def _automatic_reset(self, *, reset: GroundSkill) -> GroundSkill:
        return reset.model_copy(
            update={"skill": reset.skill.model_copy(update={"name": NON_HUMAN_RESET_NAME})}
        )

    def movables_reset_destination(self, *, ground_skill: GroundSkill) -> str | None:
        if ground_skill.skill.name not in {
            RESET_CUBE_BIN_SAME_SIDE_NAME,
            RESET_CUBE_BIN_OPPOSITE_SIDE_NAME,
            NON_HUMAN_RESET_NAME,
        }:
            return None
        # Read off the effect rather than the name: both automatic resets share one.
        added = {atom.predicate for atom in ground_skill.skill.add_effects}
        if BIN_ON_ROBOT_SIDE in added:
            return Tossing3DSide.ROBOT.value
        if BIN_ON_OPPOSITE_SIDE in added:
            return Tossing3DSide.OPPOSITE.value
        raise ValueError(f"reset {ground_skill.skill.name} adds no bin side")

    def human_skill_name(self, *, ground_skill: GroundSkill) -> str | None:
        destination = self.movables_reset_destination(ground_skill=ground_skill)
        if destination is None:
            return None
        return (
            "reset_cube_and_bin_near"
            if destination == Tossing3DSide.ROBOT.value
            else "reset_cube_far"
        )

    def movables_reset_skills(self) -> tuple[GroundSkill, ...]:
        automatic = self.non_human_cube_bin_reset_skills() if self.offer_non_human_reset else ()
        return (*self.human_cube_bin_reset_skills(), *automatic)


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
