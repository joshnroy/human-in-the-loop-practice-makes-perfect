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
from .predicates import HAND_EMPTY, HOLDING, IN_BIN, ON_GROUND, REACHABLE
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
        return (Tossing3DSkills.PICK_CUBE, Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS)

    def predicates(self) -> tuple[Predicate, ...]:
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
        return Tossing3DSkills.sample_params(ground_skill=ground_skill, rng=rng)

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)

    def human_cube_bin_reset_skill(self) -> GroundSkill:
        """Tossing3D's `ask_for_reset_cube_bin_only`: repositions `cube_0`/`bin_0`
        to fresh ground poses via `KinderBackend.reset_cube_and_bin`, robot
        untouched. Effects: `OnGround`/`Reachable` become true, `InBin` becomes
        false; everything unnamed (`HandEmpty`, `Holding`) stays as it was.

        `HandEmpty(robot)` is a real precondition, not decoration: without it the
        operator would claim `Holding` is unaffected even while the robot holds
        the cube, which repositioning it out from under a closed gripper doesn't
        actually describe (`skills.py`'s "never permit more than the raw dynamics
        allow" rule). Requiring it first makes that claim true by construction."""
        env = self.env
        robot = Variable(name="robot", type=Tossing3DEnvironment.robot_type)
        cube = Variable(name="cube", type=Tossing3DEnvironment.cube_type)
        bin_ = Variable(name="bin", type=Tossing3DEnvironment.bin_type)
        barrier = Variable(name="barrier", type=Tossing3DEnvironment.barrier_type)
        skill = Skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            parameters=(robot, cube, bin_, barrier),
            preconditions=frozenset({
                LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
            }),
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
