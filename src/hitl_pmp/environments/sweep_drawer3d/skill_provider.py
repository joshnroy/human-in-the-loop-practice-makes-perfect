"""Planner-selected Sweep skills, with learned stock parameter proposals."""

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.skill_provider import OraclePolicyProvider, SkillProvider
from hitl_pmp.core.method.types import GroundSkill, LabeledAction, Skill
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, GroundAtom, Predicate

from .environment import SweepDrawerEnvironment
from .symbolic import SWEEP_PREDICATES, SweepSymbols


class SweepDrawerSkillProvider(SkillProvider):
    env: SweepDrawerEnvironment
    human_reset_enabled: bool = True
    human_reset_practice_cost: float = Field(default=1.0, ge=0.0)
    robot_practice_costs: dict[str, float] = Field(default_factory=dict)
    stock_parameter_bounds: dict[str, tuple[tuple[float, float], tuple[float, float]]] = Field(
        default_factory=lambda: {
            "OpenDrawer": ((0.65, 0.95), (-13 * np.pi / 12, -11 * np.pi / 12)),
            "PickWiper": ((0.55, 0.85), (-13 * np.pi / 12, -11 * np.pi / 12)),
            "Sweep": ((0.40, 0.70), (-13 * np.pi / 12, -11 * np.pi / 12)),
        }
    )

    def validate_trainable_support(self) -> None:
        """Fail closed on missing or degenerate approved learning supports."""
        if set(self.stock_parameter_bounds) != set(SweepSymbols.TRAINABLE):
            raise ValueError("Expected exactly the three trainable stock supports")
        for name, bounds in self.stock_parameter_bounds.items():
            if any(not np.isfinite([lo, hi]).all() or lo >= hi for lo, hi in bounds):
                raise ValueError(f"Nonfinite or degenerate support for {name}")
            if bounds[0][0] <= 0:
                raise ValueError(f"Nonpositive distance support for {name}")

    def skills(self) -> tuple[Skill, ...]:
        return SweepSymbols.skills(costs=self.robot_practice_costs)

    def deployment_skills(self) -> tuple[Skill, ...]:
        """Same three physical task controllers used by the deployment model."""
        return tuple(s for s in self.skills() if s.name in SweepSymbols.TRAINABLE)

    def predicates(self) -> tuple[Predicate, ...]:
        return SweepSymbols.predicates()

    def types(self) -> tuple[Type, ...]:
        return (SweepSymbols.SCENE_TYPE,)

    def objects(self) -> tuple[Object, ...]:
        return (SweepSymbols.SCENE,)

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        name = ground_skill.skill.name
        if name not in SweepSymbols.TRAINABLE:
            return np.empty(0)
        self.validate_trainable_support()
        bounds = self.stock_parameter_bounds[name]
        return np.array([rng.uniform(*interval) for interval in bounds])

    def compute_action(
        self, *, ground_skill: GroundSkill, params: np.ndarray, state: State
    ) -> Action:
        del state
        if params.shape != (ground_skill.skill.param_dim,):
            raise ValueError(f"Wrong parameter shape for {ground_skill.skill.name}")
        index = self.env.ACTION_NAMES.index(ground_skill.skill.name)
        action = np.array([float(index), 0.0, 0.0])
        action[1 : 1 + len(params)] = params
        return action

    def human_cube_bin_reset_skills(self) -> tuple[GroundSkill, ...]:
        if not self.human_reset_enabled:
            return ()
        return (SweepSymbols.human_reset(cost=self.human_reset_practice_cost),)

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        resets = self.human_cube_bin_reset_skills()
        return resets[0] if resets else None

    @staticmethod
    def deployment_initial_atoms() -> frozenset[GroundAtom]:
        names = (
            "HandEmpty",
            "RecoveryHandEmpty",
            "OnTableWiper",
            *(f"OnTable{i}" for i in range(5)),
            "WiperHome",
            "ResetDrawerClosed",
            "DrawerClosed",
            "DrawerNotOpen",
            "RobotHome",
            "AnyCubeInPile",
            *(f"InPile{i}" for i in range(5)),
            *(f"SweepReachable{i}" for i in range(5)),
        )
        return frozenset(
            GroundAtom(predicate=SWEEP_PREDICATES[n], objects=(SweepSymbols.SCENE,)) for n in names
        )

    @staticmethod
    def deployment_goal_atoms() -> frozenset[GroundAtom]:
        return frozenset(
            GroundAtom(predicate=SWEEP_PREDICATES[name], objects=(SweepSymbols.SCENE,))
            for name in ("HoldingWiper", "DrawerOpen", *(f"InDrawer{i}" for i in range(5)))
        )


class SweepDrawerOracle(OraclePolicyProvider):
    env: SweepDrawerEnvironment

    def get_labeled_action(self, *, state: State, goal: Goal) -> LabeledAction:
        del goal
        provider = SweepDrawerSkillProvider(env=self.env, human_reset_enabled=False)
        for skill in provider.skills():
            if skill.name not in SweepSymbols.TRAINABLE:
                continue
            ground = GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))
            if all(a.predicate.holds(state, a.objects) for a in ground.preconditions):
                params = provider.sample_params(ground_skill=ground, rng=np.random.default_rng(0))
                return LabeledAction(
                    action=provider.compute_action(ground_skill=ground, params=params, state=state),
                    label=skill.name,
                )
        return LabeledAction(action=self.env.noop_action(), label="No applicable stock skill")
