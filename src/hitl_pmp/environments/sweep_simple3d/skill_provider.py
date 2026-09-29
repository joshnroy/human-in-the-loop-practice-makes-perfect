"""Two learned task samplers and independently selectable fixed floor recoveries."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate
from hitl_pmp.planning.grounding import SkillGrounder

from .environment import SweepSimpleEnvironment
from .symbolic import SIMPLE_PREDICATES, SimpleSymbols


class SweepSimpleSkillProvider(SkillProvider):
    env: SweepSimpleEnvironment
    human_reset_enabled: bool = True
    human_reset_practice_cost: float = Field(default=1.0, ge=0.0)
    robot_practice_costs: dict[str, float] = Field(default_factory=dict)
    stock_parameter_bounds: dict[str, tuple[tuple[float, float], tuple[float, float]]] = Field(
        default_factory=lambda: {
            "PickFloorWiper": ((0.55, 0.85), (-np.pi / 12, np.pi / 12)),
            "SweepCubeToGoal": ((0.40, 0.70), (-np.pi / 12, np.pi / 12)),
        }
    )

    def validate_trainable_support(self) -> None:
        if set(self.stock_parameter_bounds) != set(SimpleSymbols.TRAINABLE):
            raise ValueError("Expected exactly two approved Simple sampler supports")
        for bounds in self.stock_parameter_bounds.values():
            if any(not np.isfinite([lo, hi]).all() or lo >= hi for lo, hi in bounds):
                raise ValueError("Nonfinite or degenerate Simple sampler support")

    def skills(self) -> tuple[Skill, ...]:
        return SimpleSymbols.skills(costs=self.robot_practice_costs)

    def deployment_skills(self) -> tuple[Skill, ...]:
        return tuple(s for s in self.skills() if s.name in SimpleSymbols.TRAINABLE)

    def predicates(self) -> tuple[Predicate, ...]:
        return tuple(SIMPLE_PREDICATES.values())

    def types(self) -> tuple[Type, ...]:
        return SimpleSymbols.SCENE_TYPE, SimpleSymbols.CUBE_TYPE

    def objects(self) -> tuple[Object, ...]:
        return SimpleSymbols.objects()

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if ground_skill.skill.param_dim == 0:
            return np.empty(0)
        self.validate_trainable_support()
        return np.array([
            rng.uniform(*interval)
            for interval in self.stock_parameter_bounds[ground_skill.skill.name]
        ])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        if params.shape != (ground_skill.skill.param_dim,):
            raise ValueError("Invalid Simple skill parameter dimensions")
        cube = (
            -1
            if len(ground_skill.objects) == 1
            else SimpleSymbols.CUBES.index(ground_skill.objects[1])
        )
        action = np.array([
            float(self.env.ACTION_NAMES.index(ground_skill.skill.name)),
            float(cube),
            0.0,
            0.0,
        ])
        action[2 : 2 + len(params)] = params
        return action

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        return (
            (SimpleSymbols.human_reset(cost=self.human_reset_practice_cost),)
            if self.human_reset_enabled
            else ()
        )

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    @staticmethod
    def deployment_initial_atoms() -> frozenset[GroundAtom]:
        return SimpleSymbols.initial_atoms()

    @staticmethod
    def deployment_goal_atoms() -> frozenset[GroundAtom]:
        return SimpleSymbols.goal_atoms()


class SweepSimpleOracle(OraclePolicyProvider):
    env: SweepSimpleEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        provider = SweepSimpleSkillProvider(env=self.env, human_reset_enabled=False)
        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        grounded = SkillGrounder.applicable_ground_skills(
            skills=provider.deployment_skills(), objects=provider.objects(), true_atoms=atoms
        )
        if not grounded:
            return LabeledAction(action=self.env.noop_action(), label="No applicable task skill")
        skill = sorted(grounded, key=str)[0]
        params = provider.sample_params(ground_skill=skill, rng=np.random.default_rng(0))
        return LabeledAction(
            action=provider.compute_action(ground_skill=skill, params=params, state=state),
            label=skill.skill.name,
        )
