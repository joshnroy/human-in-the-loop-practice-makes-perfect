"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np

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
from .predicates import HAND_EMPTY, HOLDING, IN_BIN, ON_GROUND, REACHABLE
from .recovery_skills import CLOSED_EMPTY, ON_FLOOR, SameSideSkills
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
            return (IN_BIN, HAND_EMPTY, HOLDING, ON_FLOOR, REACHABLE, CLOSED_EMPTY)
        return (IN_BIN, HAND_EMPTY, HOLDING, ON_GROUND, REACHABLE)

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

    def human_cube_bin_reset_skill(self) -> GroundSkill:
        """Tossing3D's `ask_for_reset_cube_bin_only`: repositions `cube_0`/`bin_0`
        to fresh ground poses via `KinderBackend.reset_cube_and_bin`, robot
        untouched. Effects: `OnGround`/`Reachable` become true, `InBin` becomes
        false; everything unnamed (`HandEmpty`, `Holding`) stays as it was.

        No precondition -- callable from any state. Used to require
        `HandEmpty(robot)`, on the reasoning that without it the operator would
        claim `Holding` is unaffected even while the robot holds the cube, which
        repositioning it out from under a closed gripper doesn't actually
        describe. That guarded a real correctness gap, but it also made the
        rescue mechanism unreachable from the one state it exists to rescue:
        `HandEmpty` is a *command* read (gripper commanded open), not "nothing is
        genuinely held", and it is never the *effect* of any operator in this
        domain -- so a gripper that closes without actually grasping anything
        (`Holding` false, `HandEmpty` also false, since the command is still
        "closed") reaches a dead end no plan can escape: `PickCube` needs
        `HandEmpty`, `MoveToTossLocationAndToss` needs `Holding`, and the old
        precondition meant the reset needed `HandEmpty` too. Nothing in the
        model can ever produce `HandEmpty` from that state, so the episode raised
        `InteractionComplete` with a rescue mechanism configured and available,
        just unreachable.

        The right precondition is really `not Holding` (dropping a genuinely
        held cube out from under the gripper is the actual problem; an empty,
        commanded-closed gripper isn't), but this framework's `LiftedAtom`
        preconditions are positive-only -- no negation. Dropping the
        precondition to none is what "not Holding" degrades to given that
        constraint, since `Holding` is true only rarely (mid-carry) and this
        skill is otherwise always safe to offer. The one residual risk: a
        *hypothetical* multi-step plan built by the classical planner that
        chains this skill before `MoveToTossLocationAndToss` would internally
        assume `Holding` survives the reset, which is false. Nothing in the
        current domain builds a plan of that shape, and live execution always
        re-observes predicates fresh from the real simulator rather than
        carrying planning-time predictions forward -- but a future skill or
        planner change that did chain them this way would need to account for
        it."""
        env = self.env
        robot = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
        cube = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
        bin_ = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
        barrier = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
        skill = Skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            parameters=(robot, cube, bin_, barrier),
            preconditions=frozenset(),
            add_effects=frozenset({
                LiftedAtom(predicate=ON_GROUND, variables=(cube,)),
                LiftedAtom(predicate=REACHABLE, variables=(cube, barrier)),
            }),
            delete_effects=frozenset({
                LiftedAtom(predicate=IN_BIN, variables=(cube, bin_)),
            }),
            param_dim=0,
        )
        return GroundSkill(skill=skill, objects=(env.robot, env.cube, env.bin, env.barrier))


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
