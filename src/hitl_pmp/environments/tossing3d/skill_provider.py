"""The two injection seams a domain-agnostic `Method` needs from Tossing3D."""

import numpy as np

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Predicate

from .environment import Tossing3DEnvironment
from .predicates import Tossing3DPredicates
from .skill_oracle_policy import ORACLE_PARAMETER_SEED, SkillOraclePolicy
from .skills import Tossing3DSkills


class Tossing3DSkillProvider(SkillProvider):
    """Tossing3D's `SkillProvider`, mirroring `TossingRoomSkillProvider`.

    `objects()` is a fixed four: upstream's task JSON names exactly one cube, one bin and
    one barrier, plus the robot. There is no configuration that changes the cast -- `o2`
    would add a second cube, and this domain does not support it (see the README).

    **`skills()` and `predicates()` now need a live simulator**, because both are built
    over the abstraction of one live scene. `Tossing3DEnvironment.abstraction()` starts
    MuJoCo if it has not already, so calling either on a fresh provider is a scene build.
    That is the documented cost of consuming upstream's symbolic layer rather than
    reimplementing it; see `predicates.py`.

    `types()` is two rather than four. `bin_0`, `cube_0` and `cuboid_barrier` are all
    `MujocoMovableObjectType` upstream, and the operators bind over that one type.
    """

    env: Tossing3DEnvironment

    def skills(self) -> tuple[Skill, ...]:
        return Tossing3DSkills.all(abstraction=self.env.abstraction())

    def predicates(self) -> tuple[Predicate, ...]:
        return Tossing3DPredicates.all(abstraction=self.env.abstraction())

    def types(self) -> tuple[Type, ...]:
        return (Tossing3DEnvironment.robot_type, Tossing3DEnvironment.movable_type)

    def objects(self) -> tuple[Object, ...]:
        env = self.env
        return (env.robot, env.cube, env.bin, env.barrier)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """Delegated to the controller's own sampler.

        The `SkillProvider` contract asks for a *state-independent* draw and this reads
        the environment's current state, because `PickCubeController.sample_parameters`
        does. See `Tossing3DSkills.sample_params` for why that deviation is the safe
        direction and what pins its practical effect on this scene.
        """
        return Tossing3DSkills.sample_params(
            ground_skill=ground_skill,
            rng=rng,
            controllers=self.env.controllers(),
            state=self.env.get_current_state(),
        )

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        return Tossing3DSkills.compute_action(ground_skill=ground_skill, params=params, state=state)


class Tossing3DOracle(OraclePolicyProvider):
    """Tossing3D's privileged solver, driving `SkillOracleMethod`.

    Goal-agnostic (one goal family; see `skill_oracle_policy.py`). `parameter_seed`
    replaces the old `throw_standoff` field: with the base move and the throw fused into
    one controller, the oracle draws all four continuous parameters from that controller's
    own sampler, so what is configurable is which draw it takes rather than which standoff
    it stops at.
    """

    env: Tossing3DEnvironment
    parameter_seed: int = ORACLE_PARAMETER_SEED

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        return SkillOraclePolicy.get_labeled_action(
            state=state, env=self.env, goal=goal, parameter_seed=self.parameter_seed
        )
